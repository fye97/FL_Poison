import numpy as np
import torch

from flpoison.attackers import attacker_registry
from flpoison.attackers.pbases.mpbase import MPBase
from flpoison.fl.client import Client
from flpoison.utils.global_utils import actor


def _as_update_list(attacker_updates):
    if torch.is_tensor(attacker_updates):
        return [
            attacker_updates[idx].detach().reshape(-1)
            for idx in range(attacker_updates.shape[0])
        ]
    return [
        update.detach().reshape(-1) if torch.is_tensor(update) else np.asarray(update).reshape(-1)
        for update in attacker_updates
    ]


def _numpy_attacker_distance_rows(updates):
    num_attackers = len(updates)
    rows = []
    for i in range(num_attackers):
        row = np.empty(num_attackers - 1, dtype=np.float32)
        col = 0
        for j in range(num_attackers):
            if i == j:
                continue
            row[col] = np.linalg.norm(updates[i] - updates[j])
            col += 1
        rows.append(row)
    return rows


def _torch_attacker_distance_rows(updates):
    num_attackers = len(updates)
    device = updates[0].device
    dtype = updates[0].dtype
    rows = []
    for i in range(num_attackers):
        row = torch.empty(num_attackers - 1, device=device, dtype=dtype)
        col = 0
        for j in range(num_attackers):
            if i == j:
                continue
            row[col] = torch.linalg.vector_norm(updates[i] - updates[j])
            col += 1
        rows.append(row)
    return rows


def _candidate_selected_numpy(attacker_distance_rows, candidate_distances, num_supporters):
    num_attackers = len(attacker_distance_rows)
    keep = num_attackers - 1
    candidate_keep = num_attackers - num_supporters
    if candidate_keep > 0:
        candidate_score = float(
            np.partition(candidate_distances, candidate_keep - 1)[:candidate_keep].sum()
        )
    else:
        candidate_score = 0.0

    for row, candidate_distance in zip(attacker_distance_rows, candidate_distances):
        combined = np.empty(keep + num_supporters, dtype=row.dtype)
        combined[:keep] = row
        combined[keep:] = candidate_distance
        attacker_score = float(np.partition(combined, keep - 1)[:keep].sum())
        if attacker_score <= candidate_score:
            return False
    return True


def _candidate_selected_torch(attacker_distance_rows, candidate_distances, num_supporters):
    num_attackers = len(attacker_distance_rows)
    keep = num_attackers - 1
    candidate_keep = num_attackers - num_supporters
    if candidate_keep > 0:
        candidate_score = torch.topk(
            candidate_distances, k=candidate_keep, largest=False
        ).values.sum()
    else:
        candidate_score = torch.zeros(
            (), device=candidate_distances.device, dtype=candidate_distances.dtype
        )

    for row, candidate_distance in zip(attacker_distance_rows, candidate_distances):
        combined = torch.cat((row, candidate_distance.repeat(num_supporters)))
        attacker_score = torch.topk(combined, k=keep, largest=False).values.sum()
        if bool((attacker_score <= candidate_score).item()):
            return False
    return True


def _fang_attack_numpy(attacker_updates, perturbation_base, stop_threshold):
    updates = [np.asarray(update).reshape(-1) for update in attacker_updates]
    num_attackers = len(updates)
    mean_update = np.zeros_like(updates[0])
    for update in updates:
        mean_update += update
    mean_update /= num_attackers
    est_direction = np.sign(mean_update)

    if np.isscalar(perturbation_base):
        base = (
            np.zeros_like(updates[0])
            if float(perturbation_base) == 0.0
            else np.full_like(updates[0], perturbation_base)
        )
    else:
        base = np.asarray(perturbation_base, dtype=updates[0].dtype).reshape(-1)

    attacker_distance_rows = _numpy_attacker_distance_rows(updates)
    direction_norm_sq = float(np.dot(est_direction, est_direction))
    base_norm_sq = float(np.dot(base, base))
    base_dot_direction = float(np.dot(base, est_direction))
    offset_norm_sqs = np.empty(num_attackers, dtype=np.float64)
    offset_dot_directions = np.empty(num_attackers, dtype=np.float64)

    for idx, update in enumerate(updates):
        update_norm_sq = float(np.dot(update, update))
        update_dot_base = float(np.dot(update, base))
        update_dot_direction = float(np.dot(update, est_direction))
        offset_norm_sqs[idx] = update_norm_sq + base_norm_sq - 2.0 * update_dot_base
        offset_dot_directions[idx] = update_dot_direction - base_dot_direction

    lambda_value = 1.0
    selected = False
    for num_supporters in range(1, num_attackers):
        lambda_value = 1.0
        while True:
            lambda_sq = lambda_value * lambda_value
            candidate_distances = np.sqrt(
                np.maximum(
                    offset_norm_sqs
                    + 2.0 * lambda_value * offset_dot_directions
                    + lambda_sq * direction_norm_sq,
                    0.0,
                )
            )
            selected = _candidate_selected_numpy(
                attacker_distance_rows, candidate_distances, num_supporters
            )
            if selected or lambda_value <= stop_threshold:
                break
            lambda_value *= 0.5
        if selected:
            break

    return base - lambda_value * est_direction


def _fang_attack_torch(attacker_updates, perturbation_base, stop_threshold):
    updates = [update.detach().reshape(-1) for update in attacker_updates]
    num_attackers = len(updates)
    mean_update = torch.zeros_like(updates[0])
    for update in updates:
        mean_update.add_(update)
    mean_update.div_(num_attackers)
    est_direction = torch.sign(mean_update)

    if torch.is_tensor(perturbation_base):
        base = perturbation_base.detach().reshape(-1).to(
            device=updates[0].device, dtype=updates[0].dtype
        )
    elif np.isscalar(perturbation_base) and float(perturbation_base) == 0.0:
        base = torch.zeros_like(updates[0])
    else:
        base = torch.as_tensor(
            perturbation_base, device=updates[0].device, dtype=updates[0].dtype
        ).reshape(-1)

    attacker_distance_rows = _torch_attacker_distance_rows(updates)
    direction_norm_sq = torch.dot(est_direction, est_direction)
    base_norm_sq = torch.dot(base, base)
    base_dot_direction = torch.dot(base, est_direction)
    offset_norm_sqs = torch.empty(
        num_attackers, device=updates[0].device, dtype=updates[0].dtype
    )
    offset_dot_directions = torch.empty_like(offset_norm_sqs)

    for idx, update in enumerate(updates):
        update_norm_sq = torch.dot(update, update)
        update_dot_base = torch.dot(update, base)
        update_dot_direction = torch.dot(update, est_direction)
        offset_norm_sqs[idx] = update_norm_sq + base_norm_sq - 2.0 * update_dot_base
        offset_dot_directions[idx] = update_dot_direction - base_dot_direction

    lambda_value = 1.0
    selected = False
    for num_supporters in range(1, num_attackers):
        lambda_value = 1.0
        while True:
            lambda_sq = lambda_value * lambda_value
            candidate_distances = torch.sqrt(
                (
                    offset_norm_sqs
                    + 2.0 * lambda_value * offset_dot_directions
                    + lambda_sq * direction_norm_sq
                ).clamp_min_(0.0)
            )
            selected = _candidate_selected_torch(
                attacker_distance_rows, candidate_distances, num_supporters
            )
            if selected or lambda_value <= stop_threshold:
                break
            lambda_value *= 0.5
        if selected:
            break

    return base - lambda_value * est_direction


def craft_fang_attack(attacker_updates, perturbation_base, stop_threshold):
    updates = _as_update_list(attacker_updates)
    if not updates:
        raise ValueError("FangAttack requires at least one attacker update")
    if torch.is_tensor(updates[0]):
        return _fang_attack_torch(updates, perturbation_base, stop_threshold)
    return _fang_attack_numpy(updates, perturbation_base, stop_threshold)


@attacker_registry
@actor('attacker', 'omniscient')
class FangAttack(MPBase, Client):
    """
    [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/abs/1911.11815) - USENIX Security '20
    Fang's attack is a aggregator-specific attack with the knowledge of each adversaries (called partial knowledge in paper). If aggregator is unknown, assume one. here we assume krum.
    1. it treat the before-attack weights/gradients of attackers as benign updates, and get their mean as the update direction
    2. it crafts the attacker[0]'s weights/gradients to be selected by Krum via adding more malicious supporters around the crafted attacker's weights via binary search optimization
    """

    supports_torch_updates = True

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'stop_threshold': 1.0e-5}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"

    # after fetching updates from each client
    def omniscient(self, clients):
        attacker_updates = [
            client.update for client in clients if client.category == "attacker"
        ]
        num_attackers = len(attacker_updates)
        assert num_attackers > 1, "FangAttack requires more than 1 attacker"

        # global_weights_vec for FedAvg comes from paper, 0 for FedSGD comes from FLDetector code
        perturbation_base = self.global_weights_vec if self.args.algorithm == "FedAvg" else 0
        crafted_update = craft_fang_attack(
            attacker_updates, perturbation_base, self.stop_threshold
        )

        # Reusing the crafted point for all supporters matches the previous implementation
        # while avoiding extra model-sized copies around the selected attack vector.
        if torch.is_tensor(crafted_update):
            return crafted_update.unsqueeze(0).repeat(num_attackers, 1)
        return np.repeat(crafted_update.reshape(1, -1), num_attackers, axis=0)

    def sample_vectors(self, epsilon, w0_prime, num_byzantine):
        """Sphere generation method
        1. keep sampling random pertubation vectors and add it to the crafted attacker's weights,attacker_weights[0]
        2. if the pertubation vector is within the epsilon ball, then add attacker_weights[0] as the crafted attacker's weights for attacker_weights[i]
        return the other crafted attackers' updates around [attacker_updates[0]-epsilon, attacker_updates[0]+epsilon]
        """
        # store vectors that meet the conditions
        nearby_vectors = []
        while (len(nearby_vectors) < num_byzantine - 1):
            # generate random vector in the range of [w0_prime-epsilon, w0_prime+epsilon]
            random_vector = w0_prime + \
                np.random.uniform(-epsilon, epsilon, w0_prime.shape)
            if np.linalg.norm(random_vector - w0_prime) <= epsilon:
                nearby_vectors.append(random_vector)
        return np.stack(nearby_vectors, axis=0)
