from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from aggregators import aggregator_registry


@aggregator_registry
class Mean(AggregatorBase):
    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        with self.profile_substage("aggregate"):
            if torch.is_tensor(updates):
                return torch.mean(updates, dim=0)
            return np.mean(updates, axis=0)
