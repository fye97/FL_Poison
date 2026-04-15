import math
import random
from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from flpoison.aggregators import aggregator_registry


@aggregator_registry
class Bucketing(AggregatorBase):
    """
    [Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](https://openreview.net/forum?id=jXKKDEi5vJt) - ICLR '22
    Bucketing aggregates the updates by first shuffling the updates, then dividing the updates into buckets of size bucket_size, and finally aggregating the updates in each bucket using the given aggregator, e.g., Krum, Mean, etc.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """"
        bucket_size (int): the size of each bucket, normally can be 2,5,10
        selected_aggregator (str): the aggregator used to aggregate the updates in each bucket
        """
        self.default_defense_params = {
            "bucket_size": 2, "selected_aggregator": "Krum"}
        self.update_and_set_attr()
        self.a_aggregator = aggregator_registry[self.selected_aggregator](
            args)
        self.algorithm = "FedSGD"
        self.supports_torch_updates = bool(
            getattr(self.a_aggregator, "supports_torch_updates", False))

    def aggregate(self, updates, **kwargs):
        num_updates = len(updates)
        if num_updates == 0:
            return self.a_aggregator.aggregate(updates, **kwargs)

        if torch.is_tensor(updates):
            shuffled = updates.index_select(
                0, torch.randperm(num_updates, device=updates.device))
            bucket_avg_updates = torch.stack(
                [
                    shuffled[start:start + self.bucket_size].mean(dim=0)
                    for start in range(0, num_updates, self.bucket_size)
                ],
                dim=0,
            )
        else:
            if isinstance(updates, np.ndarray):
                shuffled = updates[np.random.permutation(num_updates)]
            else:
                shuffled = list(updates)
                random.shuffle(shuffled)
            num_buckets = math.ceil(num_updates / self.bucket_size)
            bucket_avg_updates = np.stack(
                [np.mean(shuffled[bucket_id * self.bucket_size:(bucket_id + 1) * self.bucket_size], axis=0)
                 for bucket_id in range(num_buckets)],
                axis=0,
            )

        return self.a_aggregator.aggregate(bucket_avg_updates, **kwargs)
