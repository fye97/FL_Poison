import numpy as np
import torch
from attackers import attacker_registry
from attackers.pbases.mpbase import MPBase
from fl.client import Client
from global_utils import actor
from scipy.stats import norm


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class ALIE(MPBase, Client):
    """
    [A Little Is Enough: Circumventing Defenses For Distributed Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html) - NeurIPS '19
    apply small but well-crafted perturbations to the model weight updates via mean and std of benign updates based on normal distribution
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            'z_max': None, "attack_start_epoch": None}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"

    def omniscient(self, clients):
        if self.attack_start_epoch is not None and self.global_epoch <= 2 + self.attack_start_epoch:
            return None
        if self.z_max is None:
            s = np.floor(self.args.num_clients / 2 + 1) - self.args.num_adv
            cdf_value = (self.args.num_clients - self.args.num_adv - s) / \
                (self.args.num_clients - self.args.num_adv)
            z_max = norm.ppf(cdf_value)
        else:
            z_max = self.z_max
        benign_updates = []
        for client in clients:
            if client.category != "benign":
                continue
            update = client.update
            if torch.is_tensor(update):
                update = update.detach().cpu().numpy()
            benign_updates.append(update)
        mean = np.mean(benign_updates, axis=0)
        std = np.std(benign_updates, axis=0)
        attack_vec = mean + z_max * std
        # repeat attack vector for all attackers
        return np.tile(attack_vec, (self.args.num_adv, 1))
