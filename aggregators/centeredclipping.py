from copy import deepcopy

import numpy as np
import torch
from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase


@aggregator_registry
class CenteredClipping(AggregatorBase):
    """
    [Learning from History for Byzantine Robust Optimization](https://arxiv.org/abs/2012.10333) - ICML '21
    It assumes worker use momentum, and the server aggregates the momentum updates by clipping that to the last round one, and then clip the aggregated update to a threshold.
    """

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
        self.use_torch = True

    def aggregate(self, updates, **kwargs):
        if torch.is_tensor(updates):
            if self.momentum is None or not torch.is_tensor(self.momentum):
                self.momentum = torch.zeros_like(updates[0])

            for _ in range(self.num_iters):
                self.momentum = (
                    sum(self.clip(v - self.momentum)
                        for v in updates) / len(updates)
                    + self.momentum
                )
            return self.momentum.clone()

        if self.momentum is None:
            self.momentum = np.zeros_like(updates[0], dtype=np.float32)

        for _ in range(self.num_iters):
            self.momentum = (
                sum(self.clip(v - self.momentum)
                    for v in updates) / len(updates)
                + self.momentum
            )

        return deepcopy(self.momentum)

    def clip(self, v):
        if torch.is_tensor(v):
            norm = torch.linalg.norm(v)
            scale = torch.clamp(self.norm_threshold / (norm + 1e-12), max=1.0)
            return v * scale
        scale = min(1, self.norm_threshold / np.linalg.norm(v, ord=2))
        return v * scale
