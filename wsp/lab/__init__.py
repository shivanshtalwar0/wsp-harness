"""wsp.lab: a live Weight-Space Programming lab for bf16 models.

This is the Experiment-8 harness from the paper, rebuilt so a frontier model
(Claude Opus / Fable) can play the analyst:

  * a Python sandbox with the model's weights loaded (read and write, no grads)
  * a metered budget of forward passes over *practice* data
  * a held-out eval split the analyst never sees
  * checkpoints, diffs, and a ledger of every call
  * an MCP server so the analyst drives it, and a 3D viewer so you can watch

The 4-bit tooling in ``wsp`` (quant, weights, verify) targets an MLX checkpoint.
The lab targets a full-precision Hugging Face model (default Qwen/Qwen3-4B), where
an edit lands exactly as written instead of being rounded onto a 4-bit grid.
"""

from .session import Lab, LabConfig

__all__ = ["Lab", "LabConfig"]
