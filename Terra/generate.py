import argparse
import json

import random
from pathlib import Path

import imageio
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoModel
from tqdm import tqdm

from video_refiner.util import instantiate_from_config


# Constants
IMAGE_SIZE = (288, 512)
N_FRAMES_PER_ROUND = 25
MAX_NUM_FRAMES = 50
N_TOKENS_PER_FRAME = 576
PATH_START_ID = 9
PATH_POINT_INTERVAL = 10
N_ACTION_TOKENS = 6
VIDEO_PREFIX = "NUSCENES_ACTION_"
VIDEO_REFINER_HEIGHT = 384
VIDEO_REFINER_WIDTH = 640


def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image_root", 
        type=Path,
        required=True,
        help="Path to the directory containing the images."
    )
    parser.add_argument(
        "--annotation_file",
        type=Path,
        required=True,
        help="Path to the annotation file."
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_overlapping_frames", type=int, default=3)
    parser.add_argument(
        "--decoding_method",
        type=str,
        choices=["framewise", "video_refiner"],
        default="framewise",
    )
    parser.add_argument(
        "--video_refiner_config",
        type=Path,
        default="./configs/video_refiner/inference.yaml"
    )
    parser.add_argument(
        "--video_refiner_weights",
        type=Path,
        default="./checkpoints/video_refiner/refiner_model.pt"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="lfq_tokenizer_B_256",
        help="Name of the tokenizer to use. This name corresponds to the name of subfolder in turing-motors/Terra huggingface repository."
    )
    parser.add_argument(
        "--vllm_impl",
        action="store_true",
        help="Whether to use vLLM implementation for faster generation."
    )
    parser.add_argument(

        "--world_model_name",
        type=str,
        default="world_model",
        help="Name of the world model to use. This name corresponds to the name of subfolder in turing-motors/Terra huggingface repository."
    )
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top k tokens to sample from.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p tokens to sample from.")
    parser.add_argument("--penalty_alpha", type=float, default=None, help="Penalty for repetition.")
    args = parser.parse_args()

    assert args.num_frames <= MAX_NUM_FRAMES, f"`num_frames` should be less than or equal to {MAX_NUM_FRAMES}"
    assert args.num_overlapping_frames < N_FRAMES_PER_ROUND, f"`num_overlapping_frames` should be less than {N_FRAMES_PER_ROUND}"

    return args


def set_random_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_image(image: Image.Image, size: tuple[int, int] = (288, 512)) -> torch.Tensor:
    H, W = size
    image = image.convert("RGB")
    image = image.resize((W, H))
    image_array = np.array(image)
    image_array = (image_array / 127.5 - 1.0).astype(np.float32)
    return torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()


def to_np_images(images: torch.Tensor) -> np.ndarray:
    images = images.detach().cpu()
    images = torch.clamp(images, -1., 1.)
    images = (images + 1.) / 2.
    images = images.permute(0, 2, 3, 1).numpy()
    return (255 * images).astype(np.uint8)


def load_images(file_path_list: list[Path], size: tuple[int, int] = (288, 512)) -> torch.Tensor:
    images = []
    for file_path in file_path_list:
        image = Image.open(file_path)
        image = preprocess_image(image, size)
        images.append(image)
    return torch.cat(images, dim=0)


def save_images_to_mp4(images: np.ndarray, output_path: Path, fps: int = 10):
    writer = imageio.get_writer(output_path, fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()


def determine_num_rounds(num_frames: int, num_overlapping_frames: int, n_initial_frames: int) -> int:
    n_rounds = (num_frames - n_initial_frames) // (N_FRAMES_PER_ROUND - num_overlapping_frames)
    if (num_frames - n_initial_frames) % (N_FRAMES_PER_ROUND - num_overlapping_frames) > 0:
        n_rounds += 1
    return n_rounds


def prepare_action(
    trajs: list[list[list[float]]],
    path_start_id: int, 
    path_point_interval: int, 
    n_action_tokens: int = 5, 
    start_index: int = 0, 
    n_frames: int = 25
) -> torch.Tensor:
    actions = []
    timesteps = np.arange(0.0, 3.0, 0.05)
    for i in range(start_index, start_index + n_frames):
        traj = trajs[i][path_start_id::path_point_interval][:n_action_tokens]
        action = np.array(traj)
        timestep = timesteps[path_start_id::path_point_interval][:n_action_tokens]
        action = np.concatenate([
            action[:, [1, 0]],
            timestep.reshape(-1, 1)
        ], axis=1)
        actions.append(torch.tensor(action))
    return actions


def load_video_refiner(config_path: Path, weights_path: Path):
    video_refiner_config = OmegaConf.load(config_path)
    video_refiner = instantiate_from_config(video_refiner_config["model"])
    ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")["module"]
    for key in list(ckpt.keys()):
        if "_forward_module" in key:
            ckpt[key.replace("_forward_module.", "")] = ckpt[key]
        del ckpt[key]
    missing, unexpected = video_refiner.load_state_dict(ckpt, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    return video_refiner


def load_annotations(file_path: Path) -> list[dict]:
    with open(file_path, "r") as f:
        annotations = [json.loads(l) for l in f] 
    return annotations


def prepare_output_dir(output_dir: Path | None):
    if output_dir is None:
        output_dir = Path("../generated_videos/Terra")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_context_frame_tokens(file_path_list: list[Path], tokenizer, device: torch.device) -> torch.Tensor:
    conditioning_frames = load_images(file_path_list, IMAGE_SIZE).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        input_ids = tokenizer.tokenize(conditioning_frames).detach().unsqueeze(0)
    return input_ids


def generate_tokens_for_round(
    round_id: int, 
    model,
    trajs: list[list[list[float]]], 
    input_ids: torch.Tensor, 
    num_overlapping_frames: int, 
    num_frames: int,
    num_conditioning_frames: int,
    generation_configs: dict,
) -> torch.Tensor:
    start_index = round_id * (N_FRAMES_PER_ROUND - num_overlapping_frames)
    num_frames_for_round = min(N_FRAMES_PER_ROUND, num_frames - start_index)
    actions = prepare_action(
        trajs, PATH_START_ID, PATH_POINT_INTERVAL, N_ACTION_TOKENS, start_index, num_frames_for_round
    )
    actions = torch.cat(actions, dim=0).unsqueeze(0).to(input_ids.device).float()
    num_generated_tokens = N_TOKENS_PER_FRAME * (num_frames_for_round - num_conditioning_frames)
    progress_bar = tqdm(total=num_generated_tokens, desc=f"Round {round_id + 1}")
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        output_tokens = model.generate(
            input_ids=input_ids,
            actions=actions,
            do_sample=True,
            max_length=N_TOKENS_PER_FRAME * num_frames_for_round,
            use_cache=True,
            pad_token_id=None,
            eos_token_id=None,
            progress_bar=progress_bar,
            **generation_configs
        )
    progress_bar.close()
    return output_tokens


def framewise_decode(output_tokens: torch.Tensor, tokenizer) -> np.ndarray:
    # calculate the shape of the latent tensor
    downsample_ratio = 1
    for coef in tokenizer.config.encoder_decoder_config["ch_mult"]:
        downsample_ratio *= coef
    h = IMAGE_SIZE[0] // downsample_ratio
    w = IMAGE_SIZE[1] // downsample_ratio
    c = tokenizer.config.encoder_decoder_config["z_channels"]
    latent_shape = (output_tokens.size(0), h, w, c)

    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        output_images = tokenizer.decode_tokens(output_tokens.view(-1), latent_shape)
    return to_np_images(output_images)


def video_refiner_decode(output_tokens: torch.Tensor, video_refiner, min_overlap_frames: int, num_frames: int, device: torch.device) -> np.ndarray:
    # decide which part of generated tokens to be refined in each round
    start_end_indices = []
    start_idx = 0
    num_frames_per_inference = video_refiner.num_frames
    end_idx = num_frames_per_inference
    start_end_indices.append((start_idx, end_idx))
    while True:
        rest_frames = num_frames - end_idx
        if rest_frames <= 0:
            break
        start_idx = min(end_idx - min_overlap_frames, num_frames - num_frames_per_inference)
        end_idx = start_idx + num_frames_per_inference
        start_end_indices.append((start_idx, end_idx))

    # video generation using video refiner
    all_samples = []
    for start_idx, end_idx in start_end_indices:
        batch_tokens = output_tokens[start_idx:end_idx].unsqueeze(0)
        batch = {
            "motion_bucket_id": torch.tensor([127]).unsqueeze(0).to(device),
            "fps_id": torch.tensor([9]).unsqueeze(0).to(device),
            "cond_aug": torch.tensor([0.0]).unsqueeze(0).to(device).bfloat16(),
            "tokens": batch_tokens
        }
        decoded_tokens = video_refiner.decode_tokens(batch)
        batch["cond_frames"] = torch.clamp(
            F.resize(decoded_tokens, (VIDEO_REFINER_HEIGHT, VIDEO_REFINER_WIDTH)),
            -1., 1.,
        ).bfloat16()
        c, uc = video_refiner.conditioner.get_unconditional_conditioning(batch, force_uc_zero_embeddings=[])
        N = decoded_tokens.size(0)
        z = c["concat"].clone()
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            samples = video_refiner.sample(c, cond_frame=z, shape=z.shape[1:], uc=uc, N=N)
        all_samples.append(samples)

    # overlap removal
    overlap_removed = [all_samples[0]]
    if len(all_samples) > 1:
        for i, samples in enumerate(all_samples[1:]):
            num_overlap_frames = start_end_indices[i][1] - start_end_indices[i + 1][0]
            overlap_removed.append(samples[num_overlap_frames:])
    samples = torch.cat(overlap_removed, dim=0)
    samples = video_refiner.decode_first_stage(samples)
    samples = torch.clamp(samples, -1., 1.)
    samples = (samples + 1.) / 2.
    output_images = rearrange(samples.squeeze().cpu().numpy(), "t c h w -> t h w c")
    output_images = (255 * output_images).astype(np.uint8)
    return output_images


def prepare_vllm_model(model_path: str, device: torch.device):
    from vllm import LLM, ModelRegistry
    from vllm_impl.modeling_llama_action import LlamaActionForCausalLM

    ModelRegistry.register_model("LlamaActionForCausalLM", LlamaActionForCausalLM)
    model = LLM(
        model=model_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        trust_remote_code=True,
        max_num_seqs=5,
        device=device,
        gpu_memory_utilization=0.5,
    )
    return model


def generate_with_vllm_model_for_round(
    round_id: int, 
    model,
    trajs: list[list[list[float]]], 
    input_ids: torch.Tensor, 
    num_overlapping_frames: int, 
    num_frames: int,
    num_conditioning_frames: int,
    generation_configs: dict,
):
    import logging
    from vllm import SamplingParams

    logging.getLogger("vllm").setLevel(logging.ERROR)

    start_index = round_id * (N_FRAMES_PER_ROUND - num_overlapping_frames)
    num_frames_for_round = min(N_FRAMES_PER_ROUND, num_frames - start_index)
    actions_list = prepare_action(
        trajs, PATH_START_ID, PATH_POINT_INTERVAL, N_ACTION_TOKENS, start_index, num_frames_for_round
    )
    prompt = torch.ones((N_TOKENS_PER_FRAME + N_ACTION_TOKENS) * num_conditioning_frames, dtype=torch.long) * -3
    for i in range(num_conditioning_frames):
        prompt[i * (N_TOKENS_PER_FRAME + N_ACTION_TOKENS):i * (N_TOKENS_PER_FRAME + N_ACTION_TOKENS) + N_TOKENS_PER_FRAME] = \
            input_ids[0, i * N_TOKENS_PER_FRAME:(i + 1) * N_TOKENS_PER_FRAME]
    actions = torch.cat(actions_list[:num_conditioning_frames], dim=0)
    inputs = [{"prompt_token_ids": prompt.tolist(), "multi_modal_data": {"actions": actions}}]

    sampling_params = SamplingParams(
        detokenize=False,
        max_tokens=N_TOKENS_PER_FRAME,
        stop_token_ids=None,
        **generation_configs
    )

    all_outputs = []
    for step in tqdm(range(num_frames_for_round - num_conditioning_frames), desc=f"Round {round_id + 1}"):
        outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].token_ids
        all_outputs.append(outputs)
        prompt = torch.cat([prompt, torch.tensor(outputs), torch.ones(N_ACTION_TOKENS, dtype=torch.long) * -3])
        actions = torch.cat([actions, actions_list[num_conditioning_frames + step]], dim=0)
        inputs = [{"prompt_token_ids": prompt.tolist(), "multi_modal_data": {"actions": actions}}]
    return torch.cat([
        input_ids,
        torch.tensor(all_outputs).view(-1).unsqueeze(0).to(input_ids.device)
    ], dim=1)

 
if __name__ == "__main__":
    args = prepare_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoModel.from_pretrained("turing-motors/Terra", subfolder=args.tokenizer_name, trust_remote_code=True).to(device).eval()
    if args.vllm_impl:
        model = prepare_vllm_model(args.world_model_name, device)
    else:
        model = AutoModel.from_pretrained("turing-motors/Terra", subfolder=args.world_model_name, trust_remote_code=True).to(device).eval()

    if args.decoding_method == "video_refiner":
        video_refiner = load_video_refiner(args.video_refiner_config, args.video_refiner_weights).to(device).eval()

    set_random_seed(args.seed)

    annotations = load_annotations(args.annotation_file)
    output_dir = prepare_output_dir(args.output_dir)

    for sample in annotations:
        print(f"Generating video for sample: {sample['sample_id']}")
        print(f"Context Frames: {sample['context_frames']}")
        print(f"Label: {sample['label']}")

        file_path_list = [args.image_root / frame_path for frame_path in sample["context_frames"]]
        input_ids = prepare_context_frame_tokens(file_path_list, tokenizer, device)
        num_rounds = determine_num_rounds(args.num_frames, args.num_overlapping_frames, len(file_path_list))
        print(f"Number of rounds: {num_rounds}")

        all_outputs = []
        for round_id in range(num_rounds):
            if args.vllm_impl:
                output_tokens = generate_with_vllm_model_for_round(
                    round_id,
                    model,
                    sample["instruction_trajs"],
                    input_ids,
                    args.num_overlapping_frames,
                    args.num_frames,
                    num_conditioning_frames=len(file_path_list) if round_id == 0 else args.num_overlapping_frames,
                    generation_configs={
                        "temperature": args.temperature,
                        "top_k": args.top_k if args.top_k is not None else -1,
                        "top_p": args.top_p if args.top_p is not None else 1.0,
                    }
                )
            else:
                output_tokens = generate_tokens_for_round(
                    round_id, 
                    model,
                    sample["instruction_trajs"],
                    input_ids,
                    args.num_overlapping_frames,
                    args.num_frames,
                    num_conditioning_frames=len(file_path_list) if round_id == 0 else args.num_overlapping_frames,
                    generation_configs={
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "penalty_alpha": args.penalty_alpha
                    }
                )
            output_tokens = generate_tokens_for_round(
                round_id, 
                model,
                sample["instruction_trajs"],
                input_ids,
                args.num_overlapping_frames,
                args.num_frames,
                num_conditioning_frames=len(file_path_list) if round_id == 0 else args.num_overlapping_frames,
                generation_configs={
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "penalty_alpha": args.penalty_alpha
                }
            )
            if round_id == 0:
                all_outputs.append(output_tokens[0])
            else:
                all_outputs.append(output_tokens[0, args.num_overlapping_frames * N_TOKENS_PER_FRAME:])
            input_ids = output_tokens[:, -args.num_overlapping_frames * N_TOKENS_PER_FRAME:]

        output_tokens = torch.cat(all_outputs).to(device)
        output_tokens = output_tokens.view(-1, N_TOKENS_PER_FRAME)
        num_frames = output_tokens.size(0)
        print(f"number of frames: {num_frames}")

        if args.decoding_method == "framewise":
            output_images = framewise_decode(output_tokens, tokenizer)
        else:
            output_images = video_refiner_decode(output_tokens, video_refiner, args.num_overlapping_frames, args.num_frames, device)
        save_images_to_mp4(output_images, output_dir / f"{VIDEO_PREFIX}{sample['sample_id']:06}.mp4")
