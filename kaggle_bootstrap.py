"""Kaggle bootstrap cell — paste/run this as the FIRST cell of any notebook on Kaggle.

Why this file exists: the repo only ships Colab `.ipynb` bootstraps (see
`colab/Lab21_RUN_ALL.ipynb`). Kaggle notebooks don't read a `.env` file
automatically (labkit.env.load_dotenv only looks at os.environ + the nearest
`.env` on disk — fine if the repo's `.env` is present on disk, but Kaggle
sessions often start from a fresh clone/dataset mount without it), and the
Kaggle GPU image preinstalls an older `torchao` that breaks `get_peft_model()`.
This cell makes those two gaps explicit instead of failing 10 minutes into NB3.

Usage on Kaggle:
    1. New Notebook -> Settings -> Accelerator = GPU T4 x2, Internet = On.
    2. First cell:
       !git clone <your-repo-url> repo
       %cd repo
       %run kaggle_bootstrap.py
    3. Then run notebooks/01_data_and_mask.py ... 05_evaluate_and_verdict.py
       cells (or `!python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5`).
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.getcwd()

# 1. Env vars BEFORE `import labkit` — config.get_tier() reads os.environ directly,
#    and Kaggle does not source a .env file for you.
os.environ.setdefault("COMPUTE_TIER", "T4")
os.environ.setdefault("MASK_MODE", "assistant-only")
os.environ.setdefault("EPOCHS", "2")
# Leave EVAL_LIMIT unset for a submission-grade run.

# 2. Make the repo's own .env (if present on disk) override nothing we already set
#    above, but fill in anything we didn't (keeps a single source of truth in .env
#    if you later edit it and re-run this cell).
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
try:
    from labkit import env as labkit_env
    labkit_env.load_dotenv(override=False)
except ImportError:
    pass  # labkit not installed yet on first run — fine, env vars above still apply.

# 3. Only use ONE of Kaggle's two T4 GPUs — the lab does not need multi-GPU and
#    letting accelerate/transformers see both can silently shard across them.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 4. Install deps. torch is preinstalled on Kaggle -> skip it. torchao must be
#    upgraded explicitly or the preinstalled 0.10.0 breaks get_peft_model().
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers>=5.15,<6", "trl>=1.10,<2", "peft>=0.20,<1",
     "accelerate>=1.14,<2", "datasets>=5,<6", "torchao>=0.16",
     "jinja2>=3.1,<4", "tokenizers>=0.22,<1", "bitsandbytes>=0.50,<1",
     "jupytext>=1.17,<2", "pytest>=8.3,<10"],
    check=True,
)

print("COMPUTE_TIER =", os.environ.get("COMPUTE_TIER"))
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

import torch  # noqa: E402
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
