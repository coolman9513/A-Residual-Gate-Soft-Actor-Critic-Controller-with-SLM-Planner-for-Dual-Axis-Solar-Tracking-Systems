"""
Solar Tracker SLM-RL project setup.
Target environment: sllm_rl (Python 3.7, CUDA 11.7)

IMPORTANT — PyTorch with CUDA must be installed first:
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

Then install the rest:
    pip install -e .

Or use requirements.txt directly:
    pip install -r requirements.txt
"""

from pathlib import Path

from setuptools import setup, find_packages

_HERE = Path(__file__).resolve().parent

setup(
    name="solar-tracker-sllm-rl",
    version="1.0.0",
    description="Hierarchical SLM-guided goal-conditioned RL for dual-axis solar tracking",
    python_requires=">=3.7",
    packages=find_packages(),
    # Read from requirements.txt so the two cannot drift apart. torch and
    # torchvision are skipped: they need the CUDA extra-index-url and must be
    # installed separately, as requirements.txt explains.
    install_requires=[
        line.split("#")[0].strip()
        for line in (_HERE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#")[0].strip()
        and not line.lstrip().startswith("#")
        and not line.split("#")[0].strip().startswith(("torch==", "torchvision=="))
    ],
)
