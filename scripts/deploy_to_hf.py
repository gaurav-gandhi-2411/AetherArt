"""Deploy AetherArt to HF Space.

Order of operations:
  1. Upload spaces/README.md as README.md first (sets YAML frontmatter).
  2. Upload the repo with upload_folder, explicitly excluding root README.md
     so the frontmatter is never overwritten by the GitHub-style README.

Run:
    python scripts/deploy_to_hf.py
"""

from pathlib import Path

from huggingface_hub import HfApi, upload_file

REPO_ID = "gauravgandhi2411/AetherArt"
REPO_ROOT = Path(__file__).resolve().parent.parent
SPACES_README = REPO_ROOT / "spaces" / "README.md"

api = HfApi()

# Step 1: push the Space-specific README (with YAML frontmatter) first.
# This MUST happen before upload_folder so it doesn't get overwritten.
print("[1/2] Uploading Space README (frontmatter)...")
upload_file(
    path_or_fileobj=str(SPACES_README),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="chore: update Space README",
)

# Step 2: upload the rest of the repo, skipping files that must not appear in Space.
print("[2/2] Uploading app code, modules, assets...")
api.upload_folder(
    folder_path=str(REPO_ROOT),
    path_in_repo=".",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="chore: deploy latest AetherArt to Space",
    ignore_patterns=[
        # --- CRITICAL: never overwrite Space frontmatter ---
        "README.md",
        # --- git internals ---
        ".git/**",
        ".gitignore",
        ".gitattributes",
        # --- Space-specific subfolder (already uploaded above) ---
        "spaces/**",
        # --- Python bytecode / caches ---
        "**/__pycache__/**",
        "**/*.pyc",
        ".mypy_cache/**",
        ".pytest_cache/**",
        ".coverage",
        # --- research narrative docs (GitHub-only) ---
        "reports/**",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CITATIONS.bib",
        "PLAN.md",
        "docs/lab_notebook.md",
        "docs/gallery_candidates/**",
        "docs/hero_candidates/**",
        "docs/hero_tiles/**",
        "images/**",
        # --- dev/CI tooling (no runtime use on Space) ---
        "tests/**",
        ".github/**",
        "scripts/**",
        ".pre-commit-config.yaml",
        ".flake8",
        "pyproject.toml",
        "setup.py",
        "environment.yml",
        "requirements-dev.txt",
        "requirements-lock.txt",
        # --- large local-only model/data files ---
        "models/**",
        "outputs/**",
        # LoRA training artifacts — Space only needs the adapter safetensors,
        # which is already present from prior deploys; exclude to avoid LFS
        # batch upload against the 1 GB Space LFS cap.
        "data/lora/ukiyo-e/*.safetensors",
        "data/lora/ukiyo-e/images/**",
        "data/lora/ukiyo-e/dataset/**",
        "data/lora/ukiyo-e/training_output/**",
        "data/lora/ukiyo-e-data20/**",
        "data/lora/ukiyo-e-data40/**",
        "data/lora/ukiyo-e-rank16/**",
        "data/lora/ukiyo-e-rank4/**",
        # --- misc transient files ---
        "*.log",
        "_probe_out.txt",
    ],
)

print(f"\nDone. Space: https://huggingface.co/spaces/{REPO_ID}")
print("Wait ~60 s for Space rebuild, then check status:")
print(
    f'  python -c "from huggingface_hub import HfApi; '
    f"print(HfApi().space_info('{REPO_ID}').runtime.stage)\""
)
