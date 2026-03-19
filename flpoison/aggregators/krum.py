import numpy as np
import torch

from flpoison.aggregators import aggregator_registry
from flpoison.aggregators.aggregatorbase import AggregatorBase


def _torch_krum_scores_from_distance_matrix(distances, num_byzantine):
    keep = distances.shape[0] - num_byzantine
    if keep <= 0:
        return torch.zeros(distances.shape[0], device=distances.device, dtype=distances.dtype)
    return torch.topk(distances, k=keep, largest=False, dim=1).values.sum(dim=1)


def _torch_krum_scores(updates, num_byzantine):
    return _torch_krum_scores_from_distance_matrix(
        torch.cdist(updates, updates), num_byzantine
    )


def _numpy_pairwise_distances(updates):
    num_clients = len(updates)
    if num_clients == 0:
        return np.empty((0, 0), dtype=np.float32)
    flat_updates = np.asarray(updates).reshape(num_clients, -1)
    work = flat_updates.astype(np.float64, copy=False)

    # For the small client counts used by Krum-family defenses, a Gram-matrix
    # formulation is faster than repeated model-sized subtractions.
    gram = work @ work.T
    norms = np.diag(gram)
    distances_sq = norms[:, None] + norms[None, :] - 2.0 * gram
    np.maximum(distances_sq, 0.0, out=distances_sq)
    np.sqrt(distances_sq, out=distances_sq)
    np.fill_diagonal(distances_sq, 0.0)
    return distances_sq


def _numpy_krum_scores_from_distance_matrix(distances, num_byzantine):
    keep = distances.shape[0] - num_byzantine
    if keep <= 0:
        return np.zeros(distances.shape[0], dtype=distances.dtype)
    return np.partition(distances, keep - 1, axis=1)[:, :keep].sum(axis=1)


def _numpy_krum_scores(updates, num_byzantine):
    return _numpy_krum_scores_from_distance_matrix(
        _numpy_pairwise_distances(updates), num_byzantine
    )


def krum_scores(updates, num_byzantine=0):
    if isinstance(updates, (list, tuple)) and updates and all(torch.is_tensor(update) for update in updates):
        updates = torch.stack([update.reshape(-1) for update in updates], dim=0)
    if torch.is_tensor(updates):
        return _torch_krum_scores(updates, num_byzantine)
    return _numpy_krum_scores(updates, num_byzantine)


def _torch_krum_selected_index(updates, num_byzantine):
    return int(torch.argmin(_torch_krum_scores(updates, num_byzantine)).item())


@aggregator_registry
class Krum(AggregatorBase):
    """
    [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17
    Krum first compute the score of each update, which is the sum of the n-f-1 smallest Euclidean distances to the other updates. Then it selects the update with the smallest score as the aggregated update.
    """
    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        enable_check (bool): whether to enable the check of the number of Byzantine clients
        """
        self.default_defense_params = {"enable_check": False}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        num_clients = len(updates)
        if self.enable_check:
            if 2 * self.args.num_adv + 2 >= num_clients:
                raise ValueError(
                    f"num_byzantine should be meet 2f+2 < n, got 2*{self.args.num_adv}+2 >= {num_clients}."
                )
        with self.profile_substage("defense"):
            if torch.is_tensor(updates):
                selected_idx = _torch_krum_selected_index(updates, self.args.num_adv)
            else:
                selected_idx = int(krum_scores(updates, self.args.num_adv).argmin())
        with self.profile_substage("aggregate"):
            return updates[selected_idx]


def krum(updates, num_byzantine=0, return_index=False, enable_check=False):
    if isinstance(updates, (list, tuple)) and updates and all(torch.is_tensor(update) for update in updates):
        updates = torch.stack([update.reshape(-1) for update in updates], dim=0)
    num_clients = len(updates)
    if enable_check:
        if 2 * num_byzantine + 2 >= num_clients:
            raise ValueError(
                f"num_byzantine should be meet 2f+2 < n, got 2*{num_byzantine}+2 >= {num_clients}."
            )
    if torch.is_tensor(updates):
        selected_idx = _torch_krum_selected_index(updates, num_byzantine)
        if return_index:
            return selected_idx
        return updates[selected_idx]
    selected_idx = int(krum_scores(updates, num_byzantine).argmin())
    if return_index:
        return selected_idx
    else:
        return updates[selected_idx]
