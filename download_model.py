from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/deepseek-coder-6.7b-base",
    local_dir="./models/deepseek-coder-6.7b-base",
)
