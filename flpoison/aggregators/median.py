from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch

from flpoison.aggregators import aggregator_registry


@aggregator_registry
class Median(AggregatorBase):
    """
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Coordinated Median computes the median of the updates coordinate-wisely.
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        with self.profile_substage("aggregate"):
            if torch.is_tensor(updates):
                return torch.median(updates, dim=0).values
            return np.median(updates, axis=0)
