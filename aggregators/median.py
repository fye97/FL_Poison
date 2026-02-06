import numpy as np
import torch
from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase


@aggregator_registry
class Median(AggregatorBase):
    """
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Coordinated Median computes the median of the updates coordinate-wisely.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.use_torch = True

    def aggregate(self, updates, **kwargs):
        if torch.is_tensor(updates):
            return torch.median(updates, dim=0).values
        return np.median(updates, axis=0)
