# post-training-from-scratch

From-scratch, tested LLM post-training objectives in pure PyTorch. Work in progress.

## Development

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```
