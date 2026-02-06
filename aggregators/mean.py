import numpy as np
import torch
from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase


@aggregator_registry
class Mean(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.use_torch = True

    def aggregate(self, updates, **kwargs):
        if torch.is_tensor(updates):
            return torch.mean(updates, dim=0)
        return np.mean(updates, axis=0)
