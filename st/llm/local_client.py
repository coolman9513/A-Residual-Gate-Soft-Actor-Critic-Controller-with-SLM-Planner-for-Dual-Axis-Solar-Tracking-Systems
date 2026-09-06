"""Local inference client for the fine-tuned Qwen2.5-0.5B-Instruct LoRA adapter.

Drop-in replacement for QwenClient — same chat_json() interface, no LM Studio needed.

Uses a persistent subprocess running finetune/serve_inference.py in the sllm_finetune
conda environment (which has peft + transformers).  The model loads once on first call;
subsequent calls in the same process are fast.  The guidance cache in LLMGoalGuidance
means the model is queried only ~240 times per 10-day episode, so startup overhead
is negligible across 100 training episodes.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SERVE_SCRIPT = _PROJECT_ROOT / "finetune" / "serve_inference.py"

# Path to the Python interpreter that has peft + transformers installed.
# Resolved automatically; override by passing python_exe to __init__.
def _find_sllm_python() -> str:
    candidates = [
        Path("C:/Users/mrcoo/anaconda3/envs/sllm_finetune/python.exe"),
        Path(r"C:\Users\alli13\AppData\Local\anaconda3\envs\sllm_finetune\python.exe"),
        Path(r"C:\Users\alli13\AppData\Local\anaconda3\envs\stracker\python.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable


class LocalQwenClient:
    """Runs the fine-tuned LoRA adapter in a background subprocess.

    Parameters
    ----------
    python_exe:
        Path to the Python interpreter with peft/transformers installed.
        Defaults to the sllm_finetune conda env python.
    serve_script:
        Path to finetune/serve_inference.py.
    max_new_tokens:
        Hard cap on generated tokens (overridden per-call by max_tokens).
    """

    def __init__(
        self,
        python_exe: Optional[str] = None,
        serve_script: Optional[Path | str] = None,
        max_new_tokens: int = 200,
    ):
        self.python_exe = python_exe or _find_sllm_python()
        self.serve_script = Path(serve_script) if serve_script else _SERVE_SCRIPT
        self.max_new_tokens = max_new_tokens
        self._proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _stderr_target():
        """Return a usable stderr target for subprocess.Popen.

        In Jupyter notebooks sys.stderr is a custom wrapper that does NOT
        support fileno(), so passing it directly to Popen raises an OSError.
        Fall back to subprocess.PIPE (output is discarded) in that case.
        """
        try:
            sys.stderr.fileno()
            return sys.stderr
        except (AttributeError, io.UnsupportedOperation):
            return subprocess.PIPE

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [self.python_exe, str(self.serve_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_target(),
            text=True,
            bufsize=1,
            cwd=str(_PROJECT_ROOT),
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> str:
        """Generate a response and return the raw assistant text."""

        if self._proc is None or self._proc.poll() is not None:
            self._start()

        req = json.dumps({
            "messages": messages,
            "max_tokens": min(max_tokens, self.max_new_tokens),
            "temperature": float(temperature),
        })
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("serve_inference subprocess closed stdout unexpectedly")

        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(f"serve_inference error: {result['error']}")
        return result["text"]

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
