import json
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from pandera.typing import Series
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from tqdm import tqdm as stqdm
from transformers import AutoModel, set_seed

from act_bench.metrics import average_displacement_error, final_displacement_error
from act_bench.utils import send_batch_to

tqdm = partial(stqdm, dynamic_ncols=True)
np.set_printoptions(precision=4, suppress=True)

ACT_BENCH_DATA_PATH = Path(__file__).parents[1] / "data"
SEED = 42  # do not change this for reproducibility
LABELS = [
    "curving_to_left",
    "curving_to_right",
    "straight_constant_high_speed",
    "straight_constant_low_speed",
    "straight_accelerating",
    "straight_decelerating",
    "starting",
    "stopping",
    "stopped",
]


@dataclass
class ActBenchConfig:
    input_dir: Path = field(metadata={"help": "Input directory containing generated videos"})
    output_dir: Path = field(metadata={"help": "Output directory"})
    batch_size: int = field(default=10, metadata={"help": "Batch size"})
    num_workers: int = field(default=32, metadata={"help": "Number of workers for data loader"})
    overwrite: bool = field(default=False, metadata={"help": "Overwrite existing output files"})
    verbose: bool = field(default=True, metadata={"help": "Verbose mode"})

    def __post_init__(self):
        self.labels = LABELS
        self.label_to_class_id = {l: i for i, l in enumerate(self.labels)}
        self.class_id_to_label = {i: l for i, l in enumerate(self.labels)}
        self.num_frames = 44  # (25 - 3) * 2 round
        self.img_size = 224

        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if self.verbose:
            print(f"Found {len(list(self.input_dir.glob('*.mp4')))} videos at {self.input_dir}")
            print(f"Output dir: {self.output_dir}")


@dataclass
class ActBenchResults:
    accuracy: float
    ade: float
    fde: float
    summary: dict


class DataSchema(pa.DataFrameModel):
    sample_id: Series[int] = pa.Field(unique=True)
    label: Series[str]
    context_frames: Series[list[str]]
    instruction_trajs: Series[list[list[list[float]]]]
    reference_traj: Series[list[list[float]]]
    reference_traj_transformed: Series[list[list[float]]]
    cond_label: Series[str] = pa.Field(isin=LABELS)
    cond_class: Series[int] = pa.Field(isin=list(range(8)))
    video_path: Series[str]


@pa.check_types
def prepare_data(config: ActBenchConfig, allow_missing_videos: bool = False) -> DataSchema:
    # Load Act-Bench dataset
    dataset = load_dataset("turing-motors/ACT-Bench", data_files="act_bench.jsonl", split="train")
    df_act_bench = pd.DataFrame(dataset.to_dict())  # This may take a while

    # Transform reference trajectories to match the coordinate system of act-estimator
    def transform_traj(traj):
        traj = np.array(traj)
        waypoints = traj[:, :2]
        waypoints = waypoints[:, ::-1].copy()
        waypoints[:, 0] *= -1
        return waypoints[:44].tolist()

    df_act_bench["reference_traj_transformed"] = df_act_bench.reference_traj.apply(transform_traj)

    # Get condition label
    def get_cond_label(r):
        category, subcategory = r.label.split("/")
        if category == "straight_constant_speed":
            speed = int(re.search(r"(\d+)kmph$", subcategory).group(1))
            if speed <= 20:
                return "straight_constant_low_speed"
            else:
                return "straight_constant_high_speed"
        return category

    df_act_bench["cond_label"] = df_act_bench.apply(get_cond_label, axis=1)
    df_act_bench = df_act_bench[df_act_bench.cond_label.isin(config.labels)]  # Exclude commands not defined in LABELS
    df_act_bench["cond_class"] = df_act_bench.cond_label.apply(lambda x: config.label_to_class_id[x])
    df_act_bench = df_act_bench.reset_index(drop=True)

    # Load generated videos
    df_video = pd.DataFrame({"video_path": sorted(config.input_dir.resolve().glob("*.mp4"))})
    df_video["sample_id"] = df_video.video_path.apply(lambda x: re.search(r"_(\d+)$", x.stem).group(1)).astype(int)

    # Merge generated videos paths to act_bench dataset
    if allow_missing_videos:
        df_act_bench = df_act_bench.merge(df_video, how="right", left_on="sample_id", right_on="sample_id")
    else:
        df_act_bench_size = len(df_act_bench)
        df_act_bench = df_act_bench.merge(df_video, how="left", left_on="sample_id", right_on="sample_id")
        assert len(df_act_bench) == df_act_bench_size

    assert not df_act_bench.video_path.isna().any(), "Some videos are not found."
    assert df_act_bench.video_path.apply(lambda x: x.is_file()).all(), "Some mp4 files do not exist."

    df_act_bench.video_path = df_act_bench.video_path.apply(str)

    return df_act_bench


class GeneratedVideosDataset(Dataset):
    def __init__(
        self,
        video_paths: list[str],
        num_frames: int = 44,
    ) -> None:
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.base_transforms = T.Resize((256, 256))
        self.eval_transforms = T.Compose(
            [
                self.base_transforms,
                T.CenterCrop((224, 224)),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        path = self.video_paths[idx]
        frames, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
        if frames.size(0) > self.num_frames:
            frames = frames[-self.num_frames :]
        elif frames.size(0) < self.num_frames:
            raise RuntimeError(f"Loaded video does not contain enough frames: got {len(frames)}")

        frames = torch.stack([self.eval_transforms(frame) for frame in frames])
        frames = frames.permute(1, 0, 2, 3).float() / 255

        timestamps = torch.linspace(0, 4.3, 44).view(44, 1)
        return {
            "frames": frames,
            "timestamps": timestamps,
        }


@torch.inference_mode()
def predict_on_dataloader(model: nn.Module, dataloader: DataLoader, device="cuda", verbose=True):
    predictions = []
    logits = []
    waypoints = []
    for inputs in tqdm(dataloader, dynamic_ncols=True, disable=not verbose):
        send_batch_to(inputs, device)

        with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            outputs = model(**inputs)

        logit = outputs["command"]
        logit = logit.cpu().float()
        logits.extend(logit.numpy().tolist())
        probs = F.softmax(logit, dim=1).numpy()
        class_ids = probs.argmax(axis=1)
        predictions.extend(class_ids.tolist())

        wps = outputs["waypoints"]
        waypoints.extend(wps.cpu().float().numpy().tolist())

    return np.asarray(predictions), np.asarray(logits), np.asarray(waypoints)


def plot_confusion_matrix(df_inputs: DataSchema, acc: float, config: ActBenchConfig) -> None:
    conf_matrix = confusion_matrix(df_inputs.cond_label, df_inputs.estimated_label, normalize="true")
    conf_matrix = np.round(conf_matrix, 2)

    short_labels = {
        "curving_to_left": "Curv. Left",
        "curving_to_right": "Curv. Right",
        "straight_constant_high_speed": "Str. Fast",
        "straight_constant_low_speed": "Str. Slow",
        "straight_accelerating": "Str. Accel",
        "straight_decelerating": "Str. Decel",
        "starting": "Starting",
        "stopping": "Stopping",
        "stopped": "Stopped",
    }
    labels = sorted(config.labels.copy())

    _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2g",
        cmap="Purples",
        xticklabels=[f"{short_labels[labels[i]]}" for i in range(conf_matrix.shape[1])],
        yticklabels=[f"{short_labels[labels[i]]}" for i in range(conf_matrix.shape[0])],
        annot_kws={"size": 8.5},  # fontsize in cell
        ax=ax,
        vmax=1.0,
        vmin=0,
    )
    ax.set_aspect("equal")
    ax.set_ylabel("Conditioned Action", fontsize=12)
    ax.set_xlabel("Estimated Action", fontsize=12)
    ax.set_title(f"N={len(df_inputs):,}, Accuracy={acc*100:.2f}%")
    plt.xticks(rotation=45, fontsize=9, ha="right")  # Rotate x-axis labels by 45 degrees and align them to the right
    plt.yticks(fontsize=9)
    plt.tight_layout()

    filename = config.output_dir / "confusion_matrix.pdf"
    if config.overwrite or not filename.exists():
        plt.savefig(filename)
    else:
        print(f"File {filename} already exists.")


def compute_score(config: ActBenchConfig) -> ActBenchResults:
    set_seed(SEED)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("turing-motors/Act-Estimator", trust_remote_code=True)
    model.to(device)
    model.eval()

    # Prepare inputs
    df_inputs: DataSchema = prepare_data(config, allow_missing_videos=False)

    # Predict actions and trajectories
    dataset = GeneratedVideosDataset(video_paths=df_inputs.video_path.tolist())
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    preds, _, trajs = predict_on_dataloader(model, dataloader, device)

    df_inputs["estimated_class"] = preds
    df_inputs["estimated_label"] = [config.class_id_to_label[c] for c in preds]
    df_inputs["estimated_traj"] = [t.tolist() for t in trajs]

    # Compute accuracy and plot confusion matrix
    acc = (df_inputs.cond_class == df_inputs.estimated_class).sum() / len(df_inputs)
    acc = acc.item()
    plot_confusion_matrix(df_inputs, acc, config)

    cols = [
        "sample_id",
        "context_frames",
        "cond_label",
        "cond_class",
        "instruction_trajs",
        "reference_traj",
        "reference_traj_transformed",
        "video_path",
        "estimated_class",
        "estimated_label",
        "estimated_traj",
    ]
    df_results = df_inputs[cols].copy()

    # Compute ADE and FDE
    reference_trajs = np.stack(df_inputs["reference_traj_transformed"])
    est_trajs = np.stack(df_inputs["estimated_traj"])
    ades = average_displacement_error(reference_trajs, est_trajs)
    fdes = final_displacement_error(reference_trajs, est_trajs)
    df_results["ade"] = ades
    df_results["fde"] = fdes

    mean_ade = df_results.ade.mean().item()
    mean_fde = df_results.fde.mean().item()

    # Convert trajectories to lists for JSON serialization
    filename = config.output_dir / "results.jsonl"
    if config.overwrite or not filename.exists():
        df_results.to_json(str(filename), orient="records", lines=True, force_ascii=False)
    else:
        print(f"File {filename} already exists.")

    # Compute summary
    labels = [
        "curving_to_left",
        "curving_to_right",
        "starting",
        "stopped",
        "stopping",
        "straight_accelerating",
        "straight_constant_high_speed",
        "straight_constant_low_speed",
        "straight_decelerating",
    ]

    accuracy_per_cond_class = {}
    ade_per_cond_class = {}
    fde_per_cond_class = {}
    for label in labels:
        df_subset = df_results[df_results.cond_label == label]
        if len(df_subset) == 0:
            accuracy_per_cond_class[label] = "N/A"
            ade_per_cond_class[label] = "N/A"
            fde_per_cond_class[label] = "N/A"
            continue

        correct = (df_subset.cond_label == df_subset.estimated_label).sum().item()
        accuracy_per_cond_class[label] = correct / len(df_subset)
        ade_per_cond_class[label] = df_subset.ade.mean().item()
        fde_per_cond_class[label] = df_subset.fde.mean().item()

    summary = {
        "number_of_samples": len(df_results),
        "number_of_samples_per_action_class": df_results.cond_label.value_counts().sort_index().to_dict(),
        "instruction_fidelity": {"recall": accuracy_per_cond_class, "accuracy": acc},
        "action_fidelity": {
            "ADE_per_cond_class": ade_per_cond_class,
            "ADE": mean_ade,
            "FDE_per_cond_class": fde_per_cond_class,
            "FDE": mean_fde,
        },
    }
    filename = config.output_dir / "summary.json"
    if config.overwrite or not filename.exists():
        with open(filename, "w") as f:
            json.dump(summary, f, indent=4)
    else:
        print(f"File {filename} already exists.")

    return ActBenchResults(
        accuracy=acc,
        ade=mean_ade,
        fde=mean_fde,
        summary=summary,
    )
