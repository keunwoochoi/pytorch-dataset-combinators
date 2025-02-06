# PyTorch Dataset Combinators

A lightweight library providing flexible dataset combinators for PyTorch's IterableDataset. This library enables you to easily combine multiple datasets through multiplexing (weighted sampling) or concatenation.

## Features

- **MultiplexedDataset**: Sample from multiple datasets according to specified weights
- **ConcatenatedDataset**: Combine multiple datasets sequentially in a round-robin fashion
- Distributed training support with proper worker sharding
- Reproducible sampling with optional seed control

## Installation

For now, you can copy the code directly into your project.

## Usage

### MultiplexedDataset

Sample from multiple datasets according to specified weights:

```python
from torch.utils.data import IterableDataset
from dataset_combinators import MultiplexedDataset

# Create your datasets
dataset1 = YourIterableDataset1()
dataset2 = YourIterableDataset2()

# Define sampling weights
datasets = {
    dataset1: 0.7,  # 70% samples from dataset1
    dataset2: 0.3   # 30% samples from dataset2
}

# Create multiplexed dataset
multiplexed = MultiplexedDataset(datasets, seed=42)  # seed is optional

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(multiplexed, batch_size=32, num_workers=4)
```

### ConcatenatedDataset

Combine multiple datasets sequentially:

```python
from dataset_combinators import ConcatenatedDataset

# Create your datasets
dataset1 = YourIterableDataset1()
dataset2 = YourIterableDataset2()

# Create concatenated dataset
concatenated = ConcatenatedDataset([dataset1, dataset2])

# Use with DataLoader
loader = DataLoader(concatenated, batch_size=32, num_workers=4)
```

## Features in Detail

### MultiplexedDataset

- Samples from multiple datasets according to specified weights
- Supports reproducible sampling with optional seed parameter
- Handles distributed training with proper worker sharding
- Automatically normalizes weights to sum to 1.0
- Stops iteration when any dataset is exhausted

### ConcatenatedDataset

- Combines multiple datasets sequentially in round-robin fashion
- Handles distributed training with proper worker sharding
- Continues until all datasets are exhausted
- Automatically replaces exhausted iterators


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

it's okay...

