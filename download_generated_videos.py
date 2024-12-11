from huggingface_hub import snapshot_download

# Download generated videos by Vista
snapshot_download(
    repo_id="turing-motors/ACT-Bench",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["generated_videos/vista/*"],
)

# Download generated videos by Terra
snapshot_download(
    repo_id="turing-motors/ACT-Bench",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["generated_videos/terra/*/*"],
)
