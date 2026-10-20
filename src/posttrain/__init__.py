"""Reference implementations of LLM post-training objectives.

The package is organized by training stage (``sft``, ``preference``, ``rl``) on
top of a small set of masked tensor operations in ``posttrain.ops``. Every
public function works on plain tensors so it can be tested in isolation from
any model or trainer.
"""

__version__ = "0.1.0"
