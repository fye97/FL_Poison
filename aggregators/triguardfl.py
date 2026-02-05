from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import ttest_1samp
from torch.utils.data import DataLoader, Dataset

from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from fl.models.model_utils import vec2model


class _DatasetSplit(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[int(self.indices[item])]


class _ServerEvaluator:
    def __init__(self, args, dataset: Dataset, indices: np.ndarray, batch_size: int):
        self.args = args
        self.device = args.device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.class_loaders = self._create_class_loaders(dataset, indices, batch_size)

    def _create_class_loaders(self, dataset: Dataset, indices: np.ndarray, batch_size: int):
        labels = _get_dataset_labels(dataset)
        if len(indices) == 0:
            return {}
        unique_classes = np.unique(labels[indices])
        indices_by_class = {cls: [] for cls in unique_classes}
        for idx in indices:
            label = labels[int(idx)]
            if label in indices_by_class:
                indices_by_class[label].append(int(idx))

        class_loaders = {}
        for cls, cls_indices in indices_by_class.items():
            if not cls_indices:
                continue
            class_dataset = _DatasetSplit(dataset, np.array(cls_indices))
            class_loaders[int(cls)] = DataLoader(
                class_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
        return class_loaders

    def evaluate_by_class(self, model) -> Tuple[List[float], List[float]]:
        model.to(self.device)
        model.eval()
        all_accuracies, all_losses = [], []
        with torch.no_grad():
            for _, loader in self.class_loaders.items():
                class_loss = 0.0
                all_labels, all_preds = [], []
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = model(images)
                    class_loss += self.loss_fn(logits, labels).item()
                    preds = torch.argmax(logits, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                if not all_labels:
                    continue
                accuracy = float((np.array(all_preds) == np.array(all_labels)).mean())
                loss = class_loss / len(loader)
                all_accuracies.append(accuracy)
                all_losses.append(loss)
        return all_accuracies, all_losses


def _get_dataset_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "train_labels"):
        labels = dataset.train_labels
    else:
        labels = np.array([label for _, label in dataset])
        return labels
    if isinstance(labels, list):
        return np.array(labels)
    if torch.is_tensor(labels):
        return labels.cpu().numpy()
    return np.array(labels)


def _sample_balanced_indices(dataset: Dataset, num_samples: int) -> np.ndarray:
    labels = _get_dataset_labels(dataset)
    total = len(labels)
    if total == 0:
        return np.array([], dtype=np.int64)
    if num_samples <= 0:
        num_samples = total

    num_classes = len(getattr(dataset, "classes", np.unique(labels)))
    per_class = max(1, num_samples // max(1, num_classes))
    leftover = max(0, num_samples - per_class * num_classes)

    indices = []
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        count = per_class + (1 if cls < leftover else 0)
        replace = count > len(cls_indices)
        chosen = np.random.choice(cls_indices, size=count, replace=replace)
        indices.extend(chosen.tolist())
    np.random.shuffle(indices)
    return np.array(indices, dtype=np.int64)


def _median_cosine_similarities(vectors: np.ndarray) -> List[float]:
    if vectors.size == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1)
    dot = vectors @ vectors.T
    denom = np.outer(norms, norms)
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    cosine_similarity = []
    for i in range(cosine.shape[0]):
        s = sorted(cosine[:, i].tolist())
        if len(s) > 0:
            s = s[:-1]
        if len(s) == 0:
            cosine_similarity.append(0.0)
            continue
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            median_value = s[mid]
        else:
            median_value = (s[mid - 1] + s[mid]) / 2
        cosine_similarity.append(median_value)
    return cosine_similarity


def _malicious_detection_candidate(
    cosine_similar: List[float], chosen_users: List[int], cos_threshold: float
) -> Tuple[List[int], List[int]]:
    if cosine_similar is None or len(cosine_similar) == 0:
        return [], []
    cosine_arr = np.asarray(cosine_similar, dtype=float)
    avg_cosine_similarity = np.mean(cosine_arr)
    malicious_indices = np.where(
        (cosine_arr < cos_threshold) & (cosine_arr < avg_cosine_similarity)
    )[0].tolist()
    if len(malicious_indices) == 0:
        return [], []
    malicious_ids = [chosen_users[i] for i in malicious_indices]
    return malicious_ids, malicious_indices


def _perform_t_test(
    data_vector: np.ndarray, target_indices: List[int], significance_level: float = 0.05
) -> Dict[int, Tuple[float, bool]]:
    data_vector = np.array(data_vector)
    all_indices = set(range(len(data_vector)))
    comparison_indices = list(all_indices - set(target_indices))
    comparison_group = data_vector[comparison_indices]

    results = {}
    for i in target_indices:
        target_value = data_vector[i]
        _, p_value = ttest_1samp(comparison_group, target_value, nan_policy="omit")
        results[i] = (p_value, p_value < significance_level)
    return results


def _client_detection(
    list_acc: List[List[float]],
    list_loss: List[List[float]],
    chosen_users: List[int],
    significance: float,
) -> List[int]:
    if not list_loss or len(list_loss) < 2:
        return []
    benign_loss = np.array(list_loss[-1])
    malicious_indices_rel = []
    for i in range(len(list_loss) - 1):
        cand_loss = np.array(list_loss[i])
        diff = cand_loss - benign_loss
        if diff.size == 0:
            continue
        if np.min(diff) > 0:
            malicious_indices_rel.append(i)
        else:
            pos_indices = np.where(diff > 0)[0]
            pos_res = _perform_t_test(list_loss[i], pos_indices, significance)
            pos_has_significant = any(sig for (_, sig) in pos_res.values())
            if pos_has_significant:
                malicious_indices_rel.append(i)
    if not malicious_indices_rel:
        return []
    malicious_ids = [chosen_users[i] for i in malicious_indices_rel]
    return malicious_ids


def _client_reputation(
    discount: float,
    malicious: List[int],
    alpha: np.ndarray,
    beta: np.ndarray,
    chosen_users: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_last = np.array(alpha, copy=True)
    beta_last = np.array(beta, copy=True)
    alpha_update = alpha_last.copy()
    beta_update = beta_last.copy()

    num_chosen = len(chosen_users)
    rep = np.ones(num_chosen, dtype=float)
    malicious_set = set(malicious or [])

    for i, uid in enumerate(chosen_users):
        if uid in malicious_set:
            alpha_update[uid] = discount * alpha_last[uid]
            beta_update[uid] = discount * beta_last[uid] + 1.0
            rep[i] = 0.0
        else:
            alpha_update[uid] = discount * alpha_last[uid] + 1.0
            beta_update[uid] = discount * beta_last[uid]
            denom = alpha_update[uid] + beta_update[uid]
            rep[i] = (alpha_update[uid] / denom) if denom > 0 else 0.0
    return rep, alpha_update, beta_update


def _weighted_average_vectors(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or len(vectors) == 0:
        return np.mean(vectors, axis=0)
    total = np.sum(weights)
    if total <= 0:
        return np.mean(vectors, axis=0)
    return np.average(vectors, axis=0, weights=weights)


@aggregator_registry
class TriGuardFL(AggregatorBase):
    """
    TriGuardFL: Byzantine-robust FL with cosine-similarity filtering,
    class-wise evaluation, and reputation-weighted aggregation.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "cos_threshold": 1.01,
            "significance": 0.02,
            "discount": 0.9,
            "num_items_test": 512,
            "eval_batch_size": 128,
        }
        self.update_and_set_attr()

        train_dataset = kwargs.get("train_dataset")
        if train_dataset is None:
            raise ValueError("TriGuardFL requires train_dataset for server-side evaluation.")

        self.triguard_indices = _sample_balanced_indices(train_dataset, self.num_items_test)
        eval_bs = max(1, int(self.eval_batch_size))
        if len(self.triguard_indices) > 0:
            eval_bs = min(eval_bs, len(self.triguard_indices))
        self.server_evaluator = _ServerEvaluator(self.args, train_dataset, self.triguard_indices, eval_bs)

        num_clients = int(self.args.num_clients)
        self.alphas = np.ones(num_clients, dtype=float)
        self.betas = np.ones(num_clients, dtype=float)

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs["last_global_model"]
        global_weights_vec = kwargs["global_weights_vec"]
        updates = np.array(updates, dtype=np.float32)
        num_clients = len(updates)
        if num_clients == 0:
            return global_weights_vec

        if len(self.alphas) != num_clients:
            self.alphas = np.ones(num_clients, dtype=float)
            self.betas = np.ones(num_clients, dtype=float)

        chosen_users = list(range(num_clients))
        local_model_vecs, _ = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=True
        )

        cosine_similarity = _median_cosine_similarities(local_model_vecs)
        malicious_candidate, malicious_candidate_index = _malicious_detection_candidate(
            cosine_similarity, chosen_users, self.cos_threshold
        )

        w_locals_malicious_candidate = [local_model_vecs[i] for i in malicious_candidate_index]
        w_locals_benign_candidate = [
            local_model_vecs[i] for i in range(num_clients) if i not in malicious_candidate_index
        ]
        if len(w_locals_benign_candidate) == 0:
            w_locals_benign_candidate = list(local_model_vecs)

        w_glob_benign_candidate = np.mean(np.stack(w_locals_benign_candidate, axis=0), axis=0)

        list_acc_local, list_loss_local = [], []
        for c in range(len(malicious_candidate) + 1):
            net_server_local = deepcopy(self.global_model)
            if c < len(malicious_candidate):
                vec2model(w_locals_malicious_candidate[c], net_server_local)
            else:
                vec2model(w_glob_benign_candidate, net_server_local)
            acc_total, loss_total = self.server_evaluator.evaluate_by_class(net_server_local)
            list_acc_local.append(acc_total)
            list_loss_local.append(loss_total)

        malicious = _client_detection(
            list_acc_local, list_loss_local, malicious_candidate, self.significance
        )
        rep, self.alphas, self.betas = _client_reputation(
            self.discount, malicious, self.alphas, self.betas, chosen_users
        )

        aggregated_model_vec = _weighted_average_vectors(local_model_vecs, rep)
        aggregated_grad = aggregated_model_vec - global_weights_vec
        return wrapup_aggregated_grads(
            aggregated_grad, self.args.algorithm, self.global_model, aggregated=True
        )
