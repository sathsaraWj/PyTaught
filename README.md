# pytaught

**pytaught** is a minimal Machine Learning / Deep Learning framework built from scratch using pure Python and NumPy.

**PyPI page:** [https://pypi.org/project/pytaught/1.0.0/](https://pypi.org/project/pytaught/1.0.0/)

## Installation

```bash
pip install pytaught
```

Or build locally:
```bash
python -m build
python -m twine upload dist/*
```

## Usage

```bash
pytaught --csv data.csv --target price --model mlp --hidden-sizes 64 32 --epochs 100
```

If no CSV is given, the framework generates synthetic data automatically.

## Features
- Linear regression and MLP models
- Pure NumPy training loops
- Early stopping and model checkpointing
- CLI interface for flexible training

## License
MIT License © 2025 Sathsara Wijekulasuriya
