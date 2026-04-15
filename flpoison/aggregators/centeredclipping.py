from copy import deepcopy
from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from flpoison.aggregators import aggregator_registry


@aggregator_registry
class CenteredClipping(AggregatorBase):
    """
    [Learning from History for Byzantine Robust Optimization](https://arxiv.org/abs/2012.10333) - ICML '21
    It assumes worker use momentum, and the server aggregates the momentum updates by clipping that to the last round one, and then clip the aggregated update to a threshold.
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = "FedSGD"
        """
        norm_threshold (float): the threshold for clipping the aggregated update
        num_iters (int): the number of iterations for clipping the aggregated update
        """
        self.default_defense_params = {
            "norm_threshold": 100, "num_iters": 1}
        self.update_and_set_attr()
        self.momentum = None

    def aggregate(self, updates, **kwargs):
        if self.momentum is None:
            if torch.is_tensor(updates):
                self.momentum = torch.zeros_like(updates[0])
            else:
                self.momentum = np.zeros_like(updates[0], dtype=np.float32)

        for _ in range(self.num_iters):
            clipped_sum = torch.zeros_like(self.momentum) if torch.is_tensor(
                self.momentum) else np.zeros_like(self.momentum)
            for update in updates:
                clipped_sum += self.clip(update - self.momentum)
            self.momentum = (
                clipped_sum / len(updates)
                + self.momentum
            )

        if torch.is_tensor(self.momentum):
            return self.momentum.clone()
        return self.momentum.copy()

    def clip(self, v):
        if torch.is_tensor(v):
            norm = torch.linalg.vector_norm(v)
            if float(norm.item()) == 0.0:
                return v
            scale = min(1.0, float(self.norm_threshold) / float(norm.item()))
            return v * scale
        scale = min(1, self.norm_threshold / np.linalg.norm(v, ord=2))
        return v * scale
