# ACT-Bench

ACT-Bench is a dedicated framework for quantitatively evaluating the action controllability of world models for autonomous driving. It focuses on measuring how well a world model can generate driving scenes conditioned on specified trajectories.

![overview](assets/overview.png)

## Get Started

To begin with, you have to download the benchmark dataset from [somewhere](...), and put it in the `data/` directory.

```bash
git clone https://github.com/turingmotors/ACT-Bench.git
cd ACT-Bench

# Install dependencies via uv
uv sync
# Or, via pip
pip install -e .

source .venv/bin/activate
```

## Evaluate Action Controllability

### 1. Generate Videos with World Model on the Benchmark Dataset

Put your generated videos in `generated_videos/<your_model_name>/`.
The number of mp4 files in the directory must be the same as the number of samples in the benchmark dataset, which is 2286.

**TODO**: Create a script and provide instructions for generating videos using the benchmark dataset.
Note that `instruction_trajs` may not be directly usable as-is; you will need to tailor it to match the input format required by specific world models.

### 2. Compute Scores

```bash
cd wmbench/act_bench
python compute_score.py --input_dir generated_videos/<your_model_name> --output_dir results/<your_model_name>
```

Or, you can write a script to evaluate your generated videos:

```python
from act_bench import ActBenchConfig, compute_score

config = ActBenchConfig(
    input_dir="generated_videos/<your_model_name>",
    output_dir="results/<your_model_name>",
)
results = compute_score(config)
print(f"Accuracy: {results.accuracy*100:.2f}%")
print(f"Mean ADE: {results.ade:.4f}, Mean FDE: {results.fde:.4f}")
```

## Reproduce Numbers in the Paper

We provide the results of the paper in the `results/` directory as well as the generated videos in `generated_videos/`.
The scores of the paper can be reproduced by running the following command:

```bash
# For Terra
./scripts/compute_score_terra_paper.sh

# For Vista
./scripts/compute_score_vista_paper.sh
```

Also, a notebook example is provided to reproduce the numbers: [compute_score.ipynb](notebook/compute_score.ipynb).

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
comming soon
```
