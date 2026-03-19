import numpy as np
import torch

from flpoison.aggregators import aggregator_registry
from flpoison.aggregators.aggregatorbase import AggregatorBase
from flpoison.aggregators.krum import krum_scores


@aggregator_registry
class MultiKrum(AggregatorBase):
    """
    [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17

    Multi-Krum is a variant of Krum that selects the m updates with the smallest scores, rather than just the single update chosen by Krum, where the score is the sum of the n-f-1 smallest Euclidean distances to the other updates. Then it verages these selected updates to produce the final aggregated update.
    """
    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        avg_percentage (float): the percentage of clients to be selected for averaging
        """
        self.default_defense_params = {
            "avg_percentage": 0.2, "enable_check": False}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        num_clients = len(updates)
        m_avg = int(self.avg_percentage * num_clients)
        if self.enable_check:
            if num_clients <= 2 * self.args.num_adv + 2:
                raise ValueError(
                    f"num_byzantine should be meet 2f+2 < n, got 2*{self.args.num_adv}+2 >= {num_clients}."
                )
        with self.profile_substage("defense"):
            scores = krum_scores(updates, self.args.num_adv)
            if torch.is_tensor(updates):
                selected_idx = torch.argsort(scores)[:m_avg]
                selected_updates = updates.index_select(0, selected_idx)
            else:
                selected_idx = np.argsort(scores)[:m_avg]
                selected_updates = updates[selected_idx]
        with self.profile_substage("aggregate"):
            if torch.is_tensor(selected_updates):
                return torch.mean(selected_updates, dim=0)
            return np.mean(selected_updates, axis=0)


def multi_krum(updates, num_byzantine, avg_percentage, enable_check=False):
    """
    m_avg: select smallest m scores for averaging
    """
    num_clients = len(updates)
    m_avg = int(avg_percentage * num_clients)
    if enable_check:
        if num_clients <= 2 * num_byzantine+2:
            raise ValueError(
                f"num_byzantine should be meet 2f+2 < n, got 2*{num_byzantine}+2 >= {num_clients}."
            )
    scores = krum_scores(updates, num_byzantine)
    if torch.is_tensor(updates):
        selected_idx = torch.argsort(scores)[:m_avg]
        return torch.mean(updates.index_select(0, selected_idx), dim=0)
    selected_idx = np.argsort(scores)[:m_avg]
    return np.mean(updates[selected_idx], axis=0)
