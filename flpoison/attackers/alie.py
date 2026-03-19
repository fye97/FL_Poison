import numpy as np
import torch
from scipy.stats import norm
from flpoison.fl.client import Client
from flpoison.attackers.pbases.mpbase import MPBase
from flpoison.utils.global_utils import actor
from flpoison.attackers import attacker_registry


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class ALIE(MPBase, Client):
    """
    [A Little Is Enough: Circumventing Defenses For Distributed Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html) - NeurIPS '19
    apply small but well-crafted perturbations to the model weight updates via mean and std of benign updates based on normal distribution
    """

    supports_torch_updates = True
    shared_omniscient_update = True

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            'z_max': None, "attack_start_epoch": None}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"
        self._resolved_z_max = None

    def resolve_z_max(self):
        if self._resolved_z_max is None:
            if self.z_max is None:
                s = np.floor(self.args.num_clients / 2 + 1) - self.args.num_adv
                cdf_value = (self.args.num_clients - self.args.num_adv - s) / \
                    (self.args.num_clients - self.args.num_adv)
                self._resolved_z_max = float(norm.ppf(cdf_value))
            else:
                self._resolved_z_max = float(self.z_max)
        return self._resolved_z_max

    def omniscient(self, clients):
        if self.attack_start_epoch is not None and self.global_epoch <= 2 + self.attack_start_epoch:
            return None
        benign_updates = [i.update for i in clients if i.category == "benign"]
        return craft_alie_attack(benign_updates, self.resolve_z_max())


def _numpy_mean_std_from_updates(updates):
    first = np.asarray(updates[0]).reshape(-1)
    mean = np.zeros_like(first, dtype=first.dtype)
    m2 = np.zeros_like(first, dtype=first.dtype)

    for idx, update in enumerate(updates, start=1):
        arr = np.asarray(update).reshape(-1)
        delta = arr - mean
        mean += delta / idx
        delta2 = arr - mean
        m2 += delta * delta2

    return mean, np.sqrt(np.maximum(m2 / len(updates), 0.0))


def _torch_mean_std_from_updates(updates):
    first = updates[0].detach().reshape(-1)
    mean = torch.zeros_like(first)
    m2 = torch.zeros_like(first)

    for idx, update in enumerate(updates, start=1):
        tensor = update.detach().reshape(-1)
        delta = tensor - mean
        mean = mean + delta / idx
        delta2 = tensor - mean
        m2 = m2 + delta * delta2

    return mean, torch.sqrt((m2 / len(updates)).clamp_min_(0.0))


def craft_alie_attack(benign_updates, z_max):
    if not benign_updates:
        raise ValueError("ALIE requires at least one benign client update")
    if torch.is_tensor(benign_updates):
        std, mean = torch.std_mean(benign_updates, dim=0, correction=0)
        return mean + float(z_max) * std
    if isinstance(benign_updates, (list, tuple)) and all(torch.is_tensor(update) for update in benign_updates):
        mean, std = _torch_mean_std_from_updates(benign_updates)
        return mean + float(z_max) * std
    if isinstance(benign_updates, (list, tuple)):
        mean, std = _numpy_mean_std_from_updates(benign_updates)
        return mean + float(z_max) * std
    arr = np.asarray(benign_updates)
    return np.mean(arr, axis=0) + float(z_max) * np.std(arr, axis=0)
