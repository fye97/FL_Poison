import numpy as np
import torch

from flpoison.aggregators import aggregator_registry
from flpoison.aggregators.aggregatorbase import AggregatorBase
from flpoison.aggregators.krum import (
    _numpy_krum_scores_from_distance_matrix,
    _numpy_pairwise_distances,
    _torch_krum_scores_from_distance_matrix,
)


def _select_bulyan_indices_numpy(updates, num_byzantine, set_size):
    pairwise_distances = _numpy_pairwise_distances(updates)
    active_idx = np.arange(len(updates), dtype=np.int64)
    selected_idx = []

    while len(selected_idx) < set_size and active_idx.size > 0:
        active_distances = pairwise_distances[np.ix_(active_idx, active_idx)]
        local_scores = _numpy_krum_scores_from_distance_matrix(
            active_distances, num_byzantine
        )
        local_idx = int(np.argmin(local_scores))
        selected_idx.append(int(active_idx[local_idx]))
        active_idx = np.delete(active_idx, local_idx)

    return np.array(selected_idx, dtype=np.int64)


def _select_bulyan_indices_torch(updates, num_byzantine, set_size):
    pairwise_distances = torch.cdist(updates, updates)
    active_idx = torch.arange(updates.shape[0], device=updates.device, dtype=torch.long)
    selected_idx = []

    while len(selected_idx) < set_size and active_idx.numel() > 0:
        active_distances = pairwise_distances.index_select(0, active_idx).index_select(1, active_idx)
        local_scores = _torch_krum_scores_from_distance_matrix(
            active_distances, num_byzantine
        )
        local_idx = int(torch.argmin(local_scores).item())
        selected_idx.append(int(active_idx[local_idx].item()))
        active_idx = torch.cat((active_idx[:local_idx], active_idx[local_idx + 1:]))

    return torch.tensor(selected_idx, device=updates.device, dtype=torch.long)


@aggregator_registry
class Bulyan(AggregatorBase):
    """[The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/abs/1802.07927)
    Bulyan first select a subset of updates via Krum or other norm-based aggregation rules and then computes the coordinate-wise robust aggregation of the remaining updates
    For coordinate-wise robust aggregation, original paper use coordinate-wise closest beta median, other coordinate-wise method, e.g., trimmed mean, can also be used
    """
    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        enable_check (bool): whether to enable the check of the number of Byzantine clients
        """
        self.default_defense_params = {"enable_check": False}
        self.update_and_set_attr()

        # with prior knowledge of the number of adversaries
        self.beta = self.args.num_clients - 2 * self.args.num_adv

    def aggregate(self, updates, **kwargs):
        """
        Bulyan condition check
        """
        if self.enable_check:
            if 4*self.args.num_adv + 3 > self.args.num_clients:
                raise ValueError(
                    f"num_adv should be meet 4f+3 <= n, got {4*self.args.num_adv+3} > {self.args.num_clients}.")

        with self.profile_substage("defense"):
            # 1. get the selection set by krum
            set_size = self.args.num_clients - 2 * self.args.num_adv
            if torch.is_tensor(updates):
                selected_idx = _select_bulyan_indices_torch(
                    updates, self.args.num_adv, set_size
                )
                selected_updates = updates.index_select(0, selected_idx)
            else:
                selected_idx = _select_bulyan_indices_numpy(
                    updates, self.args.num_adv, set_size
                )
                selected_updates = updates[selected_idx]

            # for the case of NoAttack, otherwise, argpartition will raise error
            if self.beta == self.args.num_clients or self.beta == len(selected_idx):
                bening_updates = selected_updates
            else:
                # 2. compute the robust aggregation via coordiante-wise method in selection set
                # return trimmed_mean(updates[selected_idx], self.args.num_adv)# if use trimmed mean as the coordinate-wise aggregation method
                if torch.is_tensor(selected_updates):
                    median = torch.quantile(selected_updates, 0.5, dim=0)
                    abs_dist = torch.abs(selected_updates - median)
                    beta_idx = torch.topk(
                        abs_dist, k=self.beta, largest=False, dim=0
                    ).indices
                    bening_updates = torch.gather(selected_updates, 0, beta_idx)
                else:
                    median = np.median(selected_updates, axis=0)
                    abs_dist = np.abs(selected_updates - median)

                    # get the smallest beta-closest-median number of elements in axis=0
                    beta_idx = np.argpartition(
                        abs_dist, self.beta, axis=0)[:self.beta]
                    bening_updates = np.take_along_axis(
                        selected_updates, beta_idx, axis=0)
        with self.profile_substage("aggregate"):
            if torch.is_tensor(bening_updates):
                return torch.mean(bening_updates, dim=0)
            return np.mean(bening_updates, axis=0)
