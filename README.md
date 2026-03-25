# DeepTimeSeries

[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%E2%80%933.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.enbuild.2023.113027-blue)](https://doi.org/10.1016/j.enbuild.2023.113027)

A deep learning library for time series forecasting, built on **PyTorch** and **PyTorch Lightning**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Overview

DeepTimeSeries provides a logical framework for designing and implementing various deep learning architectures specifically tailored for time series forecasting.

> **We present logical guidelines for designing various deep learning models for time series forecasting.**

The library targets intermediate-level users who need to develop deep learning models for time series prediction. It solves common challenges unique to time series data—variable-length sequences, encoding/decoding windows, multi-feature handling, and probabilistic forecasting—while its high-level API allows beginners to use pre-implemented models with minimal configuration.

## Key Features

- **Modular Architecture** — Clean separation between encoding, decoding, and prediction components
- **Multiple Model Support** — Pre-implemented models including MLP, Dilated CNN, RNN variants (LSTM, GRU), and Transformer
- **Flexible Data Handling** — Pandas DataFrame-based data processing with chunk-based extraction
- **Data Preprocessing** — Built-in `ColumnTransformer` for feature scaling and transformation
- **Probabilistic Forecasting** — Support for both deterministic and probabilistic predictions
- **PyTorch Lightning Integration** — Seamless training, validation, and testing workflows

## Installation

```bash
# Using pip
pip install .

# Using uv (recommended)
uv sync

# For development with dev dependencies
uv sync --all-groups
```

<details>
<summary><strong>Requirements</strong></summary>

| Package | Version |
|---------|---------|
| Python | ≥ 3.10, < 3.12 |
| PyTorch | ≥ 2.0.0 |
| PyTorch Lightning | ≥ 2.0.0 |
| NumPy | ≥ 1.24.2 |
| Pandas | ≥ 1.5.3 |
| XArray | ≥ 2023.2.0 |

See [`pyproject.toml`](pyproject.toml) for the complete list of dependencies.

</details>

## Quick Start

```python
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import deep_time_series as dts
from deep_time_series.model import MLP
from sklearn.preprocessing import StandardScaler

# Prepare data
data = pd.DataFrame({
    'target': np.sin(np.arange(100)),
    'feature': np.cos(np.arange(100))
})

# Preprocess data
transformer = dts.ColumnTransformer(
    transformer_tuples=[
        (StandardScaler(), ['target', 'feature'])
    ]
)
data = transformer.fit_transform(data)

# Create model
model = MLP(
    hidden_size=64,
    encoding_length=10,
    decoding_length=5,
    target_names=['target'],
    nontarget_names=['feature'],
    n_hidden_layers=2,
)

# Create dataset and dataloader
dataset = dts.TimeSeriesDataset(
    data_frames=data,
    chunk_specs=model.make_chunk_specs()
)
dataloader = DataLoader(dataset, batch_size=32)

# Train model
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
```

## Supported Models

| Model | Target Features | Non-target Features | Deterministic | Probabilistic |
|-------|:-:|:-:|:-:|:-:|
| MLP | ✓ | ✓ | ✓ | ✓ |
| Dilated CNN | ✓ | ✓ | ✓ | ✓ |
| Vanilla RNN | ✓ | ✓ | ✓ | ✓ |
| LSTM | ✓ | ✓ | ✓ | ✓ |
| GRU | ✓ | ✓ | ✓ | ✓ |
| Transformer | ✓ | ✓ | ✓ | ✓ |

## Project Structure

```
deep_time_series/
├── core.py          # ForecastingModule, Head, BaseHead, etc.
├── chunk.py         # Chunk specification and extraction
├── dataset.py       # TimeSeriesDataset
├── transform.py     # ColumnTransformer for preprocessing
├── plotting.py      # Visualization utilities
├── layer.py         # Custom neural network layers
├── util.py          # Utility functions
└── model/           # Pre-implemented forecasting models
    ├── mlp.py
    ├── dilated_cnn.py
    ├── rnn.py
    └── single_shot_transformer.py
```

## Core Concepts

### Chunk Specification

The library uses a chunk-based approach for handling time series data:

| Chunk | Purpose |
|-------|---------|
| `EncodingChunkSpec` | Defines the input window for the encoder |
| `DecodingChunkSpec` | Defines the input window for the decoder |
| `LabelChunkSpec` | Defines the target window for prediction |

### Forecasting Module

All models inherit from `ForecastingModule`, which provides:
- Automatic training / validation / test step implementations
- Metric tracking and logging
- Loss calculation with multiple heads
- Chunk specification generation

### Data Flow

```
DataFrame → ColumnTransformer → TimeSeriesDataset → Lightning Trainer → ChunkInverter → DataFrame
```

## Documentation

Full documentation: **https://bet-lab.github.io/DeepTimeSeries/**

- **User Guide** — Design concepts and usage patterns
- **Tutorials** — Step-by-step examples
- **API Reference** — Complete API documentation

## Citation

If you use DeepTimeSeries in your research, please cite:

> Choi, W., & Lee, S. (2023). Performance evaluation of deep learning architectures for load and temperature forecasting under dataset size constraints and seasonality. *Energy and Buildings*, 288, 113027. https://doi.org/10.1016/j.enbuild.2023.113027

```bibtex
@article{choi2023performance,
  author  = {Choi, W. and Lee, S.},
  title   = {Performance evaluation of deep learning architectures for load
             and temperature forecasting under dataset size constraints
             and seasonality},
  journal = {Energy and Buildings},
  volume  = {288},
  pages   = {113027},
  year    = {2023},
  doi     = {10.1016/j.enbuild.2023.113027}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Authors

- [Sangwon Lee](https://github.com/swlee-bet)
- [Wonjun Choi](https://github.com/wjchoi-bet)
