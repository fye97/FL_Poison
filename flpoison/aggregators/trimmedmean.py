from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from flpoison.aggregators import aggregator_registry


@aggregator_registry
class TrimmedMean(AggregatorBase):
    """
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Trimmed Mean exludes the smallest and largest beta fraction coordiantes of the updates and averages the rest coordiantes.
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        beta (float): fraction of updates to exclude, both from the top and the bottom
        """
        self.default_defense_params = {"beta": 0.1}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        with self.profile_substage("defense"):
            num_excluded = int(self.beta * len(updates))
            if torch.is_tensor(updates):
                if num_excluded == 0:
                    trimmed_updates = updates
                else:
                    sorted_updates = torch.sort(updates, dim=0).values
                    trimmed_updates = sorted_updates[num_excluded: len(updates) - num_excluded]
            else:
                if num_excluded == 0:
                    trimmed_updates = updates
                else:
                    smallest_excluded = np.partition(
                        updates, kth=num_excluded, axis=0)[:num_excluded]
                    biggest_excluded = np.partition(
                        updates, kth=-num_excluded, axis=0)[-num_excluded:]
                    trimmed_updates = np.concatenate(
                        (updates, -smallest_excluded, -biggest_excluded)
                    )

        with self.profile_substage("aggregate"):
            if torch.is_tensor(trimmed_updates):
                return torch.mean(trimmed_updates, dim=0)
            if num_excluded == 0:
                return np.mean(trimmed_updates, axis=0)
            weights = trimmed_updates.sum(0)
            weights /= len(updates) - 2 * num_excluded
            return weights


def trimmed_mean(updates, filter_frac):
    num_excluded = int(filter_frac * len(updates))
    if num_excluded == 0:
        return np.mean(updates, axis=0)
    smallest_excluded = np.partition(
        updates, kth=num_excluded, axis=0)[:num_excluded]
    biggest_excluded = np.partition(
        updates, kth=-num_excluded, axis=0)[-num_excluded:]

    # fast way: add and substract. here directly add the negative values of smallest_excluded and biggest_excluded for counterbalance
    weights = np.concatenate(
        (updates, -smallest_excluded, -biggest_excluded)).sum(0)
    weights /= len(updates) - 2 * num_excluded
    return weights
