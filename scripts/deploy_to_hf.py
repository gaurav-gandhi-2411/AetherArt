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
        # --- large local-only files ---
        "models/**",
        "outputs/**",
        "data/lora/ukiyo-e/dataset/**",
        "reports/lora_comparison_gallery/**",
        # --- Python bytecode ---
        "**/__pycache__/**",
        "**/*.pyc",
        # --- temp/dev scripts ---
        "scripts/_*.py",
        # --- misc ---
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
