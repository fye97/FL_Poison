import numpy as np
import torch
from flpoison.utils.global_utils import actor
from flpoison.attackers.pbases.mpbase import MPBase
from flpoison.attackers import attacker_registry
from flpoison.fl.client import Client


class MinBase(MPBase, Client):
    """
    [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) - NDSS '21
    """
    supports_torch_updates = True

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            'gamma_init': 10, 'stop_threshold': 1.0e-5}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def omniscient(self, clients):
        # self.__class__.__name__ is the string-type subclass name when inherited. here is MinSum or MinMax
        attack_vec = Min(clients, self.__class__.__name__,
                         'unit_vec', self.gamma_init, self.stop_threshold)
        # repeat attack vector for all attackers
        if torch.is_tensor(attack_vec):
            return attack_vec.unsqueeze(0).repeat(self.args.num_adv, 1)
        return np.tile(attack_vec, (self.args.num_adv, 1))


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class MinMax(MinBase):
    """
    MinMax attack aims to find a malicious gradient, whose maximum distance from other benign gradient updates is smaller than the maximum distance between any two benign gradient updates via finding a optimal gamma
    """
    pass


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class MinSum(MinBase):
    """
    MinSum seeks a malicious gradient whose sum of distances from other benign gradient updates is smaller than the sum of distances of any benign gradient updates from other benign updates via finding a optimal gamma
    """
    pass


def get_metrics(metric_type):
    if metric_type == 'MinMax':
        def metric(x): return np.linalg.norm(x, axis=1).max()
    elif metric_type == 'MinSum':
        def metric(x): return np.square(np.linalg.norm(x, axis=1)).sum()
    return metric


def _benign_updates(clients):
    benign_updates = [client.update for client in clients if client.category == "benign"]
    if not benign_updates:
        raise ValueError("Min-based attacks require at least one benign client update")
    return benign_updates


def _numpy_deviation(updates, benign_mean, dev_type):
    if dev_type == 'unit_vec':
        mean_norm = np.linalg.norm(benign_mean)
        return np.zeros_like(benign_mean) if mean_norm == 0 else benign_mean / mean_norm
    if dev_type == 'sign':
        return np.sign(benign_mean)
    if dev_type == 'std':
        sum_sq = np.zeros_like(benign_mean)
        for update in updates:
            sum_sq += np.square(update)
        variance = np.maximum(sum_sq / len(updates) - np.square(benign_mean), 0.0)
        return np.sqrt(variance)
    raise ValueError(f"Unsupported deviation type: {dev_type}")


def _torch_deviation(updates, benign_mean, dev_type):
    if dev_type == 'unit_vec':
        mean_norm = torch.linalg.vector_norm(benign_mean)
        if float(mean_norm.item()) == 0.0:
            return torch.zeros_like(benign_mean)
        return benign_mean / mean_norm
    if dev_type == 'sign':
        return torch.sign(benign_mean)
    if dev_type == 'std':
        sum_sq = torch.zeros_like(benign_mean)
        for update in updates:
            sum_sq.add_(update.square())
        variance = (sum_sq / len(updates) - benign_mean.square()).clamp_min_(0.0)
        return torch.sqrt(variance)
    raise ValueError(f"Unsupported deviation type: {dev_type}")


def _min_numpy(updates, metric_type, dev_type, gamma_init, stop_threshold):
    updates = [np.asarray(update).reshape(-1) for update in updates]
    benign_mean = np.zeros_like(updates[0])
    norm_sqs = []
    for update in updates:
        benign_mean += update
        norm_sqs.append(float(np.dot(update, update)))
    benign_mean /= len(updates)

    deviation = _numpy_deviation(updates, benign_mean, dev_type)
    deviation_norm_sq = float(np.dot(deviation, deviation))
    mean_norm_sq = float(np.dot(benign_mean, benign_mean))
    mean_dot_deviation = float(np.dot(benign_mean, deviation))
    total_sum = np.zeros_like(benign_mean)
    for update in updates:
        total_sum += update
    total_norm_sq = float(sum(norm_sqs))

    centered_norm_sqs = []
    centered_dot_deviations = []
    for update, norm_sq in zip(updates, norm_sqs):
        update_dot_mean = float(np.dot(update, benign_mean))
        centered_norm_sqs.append(max(norm_sq + mean_norm_sq - 2.0 * update_dot_mean, 0.0))
        centered_dot_deviations.append(float(np.dot(update, deviation)) - mean_dot_deviation)

    if metric_type == 'MinMax':
        upper_bound = 0.0
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                dot_ij = float(np.dot(updates[i], updates[j]))
                pair_dist_sq = max(norm_sqs[i] + norm_sqs[j] - 2.0 * dot_ij, 0.0)
                upper_bound = max(upper_bound, pair_dist_sq)
    elif metric_type == 'MinSum':
        upper_bound = 0.0
        num_updates = len(updates)
        for update, norm_sq in zip(updates, norm_sqs):
            metric_value = total_norm_sq + num_updates * norm_sq - 2.0 * float(np.dot(update, total_sum))
            upper_bound = max(upper_bound, metric_value)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    lamda, step, lamda_succ = float(gamma_init), float(gamma_init) / 2.0, 0.0
    while abs(lamda_succ - lamda) > stop_threshold:
        lambda_sq = lamda * lamda
        if metric_type == 'MinMax':
            mal_metric_value = max(
                max(centered_norm_sq + lambda_sq * deviation_norm_sq + 2.0 * lamda * centered_dot_deviation, 0.0)
                for centered_norm_sq, centered_dot_deviation in zip(centered_norm_sqs, centered_dot_deviations)
            )
        else:
            mal_metric_value = sum(
                max(centered_norm_sq + lambda_sq * deviation_norm_sq + 2.0 * lamda * centered_dot_deviation, 0.0)
                for centered_norm_sq, centered_dot_deviation in zip(centered_norm_sqs, centered_dot_deviations)
            )

        if mal_metric_value <= upper_bound:
            lamda_succ = lamda
            lamda += step
        else:
            lamda -= step
        step /= 2.0

    return benign_mean - lamda_succ * deviation


def _min_torch(updates, metric_type, dev_type, gamma_init, stop_threshold):
    updates = [update.detach().reshape(-1) for update in updates]
    benign_mean = torch.zeros_like(updates[0])
    norm_sqs = []
    for update in updates:
        benign_mean.add_(update)
        norm_sqs.append(torch.dot(update, update))
    benign_mean.div_(len(updates))

    deviation = _torch_deviation(updates, benign_mean, dev_type)
    deviation_norm_sq = torch.dot(deviation, deviation)
    mean_norm_sq = torch.dot(benign_mean, benign_mean)
    mean_dot_deviation = torch.dot(benign_mean, deviation)
    total_sum = torch.zeros_like(benign_mean)
    for update in updates:
        total_sum.add_(update)
    total_norm_sq = sum(norm_sqs, torch.zeros((), device=updates[0].device, dtype=updates[0].dtype))

    centered_norm_sqs = []
    centered_dot_deviations = []
    for update, norm_sq in zip(updates, norm_sqs):
        update_dot_mean = torch.dot(update, benign_mean)
        centered_norm_sqs.append((norm_sq + mean_norm_sq - 2.0 * update_dot_mean).clamp_min_(0.0))
        centered_dot_deviations.append(torch.dot(update, deviation) - mean_dot_deviation)

    if metric_type == 'MinMax':
        upper_bound = torch.zeros((), device=updates[0].device, dtype=updates[0].dtype)
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                pair_dist_sq = (norm_sqs[i] + norm_sqs[j] - 2.0 * torch.dot(updates[i], updates[j])).clamp_min_(0.0)
                upper_bound = torch.maximum(upper_bound, pair_dist_sq)
    elif metric_type == 'MinSum':
        upper_bound = torch.zeros((), device=updates[0].device, dtype=updates[0].dtype)
        num_updates = len(updates)
        for update, norm_sq in zip(updates, norm_sqs):
            metric_value = total_norm_sq + num_updates * norm_sq - 2.0 * torch.dot(update, total_sum)
            upper_bound = torch.maximum(upper_bound, metric_value)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    lamda, step, lamda_succ = float(gamma_init), float(gamma_init) / 2.0, 0.0
    while abs(lamda_succ - lamda) > stop_threshold:
        lambda_sq = lamda * lamda
        if metric_type == 'MinMax':
            mal_metric_value = torch.zeros((), device=updates[0].device, dtype=updates[0].dtype)
            for centered_norm_sq, centered_dot_deviation in zip(centered_norm_sqs, centered_dot_deviations):
                dist_sq = (centered_norm_sq + lambda_sq * deviation_norm_sq + 2.0 * lamda * centered_dot_deviation).clamp_min_(0.0)
                mal_metric_value = torch.maximum(mal_metric_value, dist_sq)
        else:
            mal_metric_value = torch.zeros((), device=updates[0].device, dtype=updates[0].dtype)
            for centered_norm_sq, centered_dot_deviation in zip(centered_norm_sqs, centered_dot_deviations):
                mal_metric_value = mal_metric_value + (
                    centered_norm_sq + lambda_sq * deviation_norm_sq + 2.0 * lamda * centered_dot_deviation
                ).clamp_min_(0.0)

        if bool((mal_metric_value <= upper_bound).item()):
            lamda_succ = lamda
            lamda += step
        else:
            lamda -= step
        step /= 2.0

    return benign_mean - lamda_succ * deviation


def Min(clients, type, dev_type, gamma_init, stop_threshold):
    benign_updates = _benign_updates(clients)
    if torch.is_tensor(benign_updates[0]):
        return _min_torch(benign_updates, type, dev_type, gamma_init, stop_threshold)
    return _min_numpy(benign_updates, type, dev_type, gamma_init, stop_threshold)
