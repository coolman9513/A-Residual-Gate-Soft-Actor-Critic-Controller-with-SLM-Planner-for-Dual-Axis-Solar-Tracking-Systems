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

from setuptools import setup, find_packages

setup(
    name="solar-tracker-sllm-rl",
    version="1.0.0",
    description="Hierarchical SLM-guided goal-conditioned RL for dual-axis solar tracking",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        # Reinforcement Learning
        "stable-baselines3==2.0.0",
        "gym==0.26.2",
        "gymnasium==0.28.1",
        "Shimmy==1.1.0",
        "CityLearn==2.1.2",
        "cloudpickle==2.2.1",

        # Scientific / Data
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scipy==1.7.3",
        "scikit-learn==1.0.2",
        "joblib==1.3.2",
        "threadpoolctl==3.1.0",
        "h5py==3.8.0",

        # Visualisation
        "matplotlib==3.5.3",
        "seaborn==0.12.2",
        "Pillow==9.5.0",

        # Solar / Weather
        "pvlib==0.10.4",

        # LLM / API (OpenAI-compatible local inference via LM Studio)
        "openai==1.39.0",
        "httpx==0.24.1",
        "httpcore==0.17.3",
        "pydantic==2.5.3",
        "pydantic_core==2.14.6",
        "annotated-types==0.5.0",
        "anyio==3.7.1",
        "sniffio==1.3.1",
        "distro==1.9.0",
        "h11==0.14.0",

        # Jupyter / Notebook
        "ipywidgets==8.1.8",
        "jupyterlab_widgets==3.0.16",
        "widgetsnbextension==4.0.15",

        # Utilities
        "tqdm==4.67.3",
        "requests==2.31.0",
        "urllib3==1.26.13",
        "charset-normalizer==3.4.7",
        "idna==3.10",
        "certifi",
        "simplejson==3.20.2",
        "beautifulsoup4==4.14.3",
        "soupsieve==2.4.1",
        "pytz==2026.2",
        "typing_extensions==4.7.1",
        "importlib-metadata==6.7.0",
        "zipp==3.15.0",
        "exceptiongroup==1.3.1",
        "cached-property==1.5.2",
        "packaging",
        "pyparsing==3.1.4",
        "cycler==0.11.0",
        "kiwisolver==1.4.5",
        "fonttools==4.38.0",
        "python-dateutil",
        "six",
        "colorama==0.4.6",
        "pygame==2.6.1",
    ],
)
