from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AnonymousUser19/Fragments-dataset",
    repo_type="dataset",
    local_dir="./Fragments-dataset",
    resume_download=True,
    force_download=True
)