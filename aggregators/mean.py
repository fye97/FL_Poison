from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class Mean(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        with self.profile_substage("aggregate"):
            return np.mean(updates, axis=0)
