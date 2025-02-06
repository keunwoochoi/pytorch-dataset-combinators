import random
from typing import Dict, List, Iterator, Optional
from torch.utils.data import IterableDataset, get_worker_info


class MultiplexedDataset(IterableDataset):
    """Dataset that samples from multiple datasets according to specified weights."""

    def __init__(
        self,
        datasets: Dict[IterableDataset, float],
        seed: Optional[int] = None,
    ):
        """Initialize multiplexed dataset.

        Args:
            datasets: Dict mapping datasets to their sampling weights
            seed: Optional random seed for reproducibility
        """
        self.datasets = list(datasets.keys())
        total_weight = sum(datasets.values())
        self.weights = [w / total_weight for w in datasets.values()]
        self.seed = seed

        # For worker sharding
        self._worker_info = None
        self._shared_seed = None

    def _get_shared_seed(self) -> int:
        """Get seed shared across workers for consistent sampling."""
        if self._shared_seed is None:
            # Use provided seed or random value
            base_seed = self.seed if self.seed is not None else random.randint(0, 2**32-1)
            # Add epoch to seed for different permutations per epoch
            self._shared_seed = base_seed
        return self._shared_seed

    def _get_iterator(self, dataset: IterableDataset) -> Iterator:
        """Get iterator for a dataset with proper worker sharding."""
        if self._worker_info is None:  # Single worker
            return iter(dataset)

        # Multiple workers: each worker gets a different slice of data
        worker_id = self._worker_info.id
        num_workers = self._worker_info.num_workers

        # Set worker seed for reproducibility
        worker_seed = self._get_shared_seed() + worker_id
        random.seed(worker_seed)

        # Get iterator and advance to worker's section
        it = iter(dataset)
        for _ in range(worker_id):
            next(it)
        return it

    def __iter__(self):
        """Iterate over samples from datasets according to weights."""
        self._worker_info = get_worker_info()

        # Create iterators for each dataset
        iterators = {
            dataset: self._get_iterator(dataset)
            for dataset in self.datasets
        }

        # Set random seed for sampling
        random_seed = self._get_shared_seed()
        rng = random.Random(random_seed)

        # Sample from datasets according to weights
        while True:
            try:
                chosen_dataset = rng.choices(self.datasets, weights=self.weights, k=1)[0]
                yield next(iterators[chosen_dataset])
            except StopIteration:
                # If any dataset is exhausted, stop iteration
                break


class ConcatenatedDataset(IterableDataset):
    """Dataset that concatenates multiple datasets sequentially."""

    def __init__(self, datasets: List[IterableDataset]):
        """Initialize concatenated dataset.

        Args:
            datasets: List of datasets to concatenate
        """
        self.datasets = datasets
        self._worker_info = None

    def _get_iterator(self, dataset: IterableDataset) -> Iterator:
        """Get iterator for a dataset with proper worker sharding."""
        if self._worker_info is None:  # Single worker
            return iter(dataset)

        # Multiple workers: each worker gets a different slice of data
        worker_id = self._worker_info.id
        num_workers = self._worker_info.num_workers

        # Get iterator and advance to worker's section
        it = iter(dataset)
        for _ in range(worker_id):
            next(it)
        return it

    def __iter__(self):
        """Iterate over datasets in round-robin fashion."""
        self._worker_info = get_worker_info()

        # Create iterators for each dataset
        iterators = [self._get_iterator(dataset) for dataset in self.datasets]

        while iterators:  # Continue until all iterators are exhausted
            # Try getting one item from each dataset in order
            for i, iterator in enumerate(iterators):
                try:
                    yield next(iterator)
                except StopIteration:
                    # Replace exhausted iterator with a new one
                    iterators[i] = self._get_iterator(self.datasets[i])
                    try:
                        yield next(iterators[i])
                    except StopIteration:
                        # If we can't get any items from this dataset, remove it
                        del iterators[i]
                        if not iterators:  # If all datasets are exhausted
                            return
