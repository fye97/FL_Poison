from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from flpoison.aggregators import aggregator_registry


@aggregator_registry
class Mean(AggregatorBase):
    supports_torch_updates = True
    accepts_unstacked_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        with self.profile_substage("aggregate"):
            if torch.is_tensor(updates):
                return torch.mean(updates, dim=0)
            if isinstance(updates, (list, tuple)) and updates and all(torch.is_tensor(update) for update in updates):
                aggregated = updates[0].detach().clone()
                for update in updates[1:]:
                    aggregated.add_(update)
                aggregated.div_(len(updates))
                return aggregated
            return np.mean(updates, axis=0)
