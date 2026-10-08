"""Download CuiSync's dataset files during the Render build.

The .json / .npy data files are not in GitHub (see .gitignore), so Render
can't get them from the repo. This script downloads any that are missing
from a private Hugging Face dataset repo:

    HF_DATA_REPO = your-username/cuisync-data
    HF_TOKEN     = a Hugging Face token that can read it

If all the files are already here (like on your own computer), it does nothing.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES = ("ingredients", "actions", "cookware", "utensils")
REQUIRED = ["recipes.json"] + [
    f"{feature}_{suffix}" for feature in FEATURES for suffix in ("item_vecs.npy", "items.json")
]
OPTIONAL = ["embeddings_order.json"]  # app.py uses it as a safety check if it's there


def main():
    missing = [name for name in REQUIRED + OPTIONAL if not os.path.exists(os.path.join(ROOT, name))]
    if not missing:
        print("[fetch_data] All data files are already here.")
        return

    repo = os.environ.get("HF_DATA_REPO", "").strip()
    if not repo:
        sys.exit("[fetch_data] Missing data files and HF_DATA_REPO is not set: " + ", ".join(missing))

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    for name in missing:
        try:
            hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, local_dir=ROOT, token=token)
            print(f"[fetch_data] Downloaded {name}")
        except Exception as error:
            if name in OPTIONAL:
                print(f"[fetch_data] Skipped optional {name}: {error}")
            else:
                sys.exit(f"[fetch_data] Could not download {name} from {repo}: {error}")


if __name__ == "__main__":
    main()
