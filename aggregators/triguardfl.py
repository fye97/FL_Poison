from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from math import erfc, sqrt
from scipy.stats import ttest_1samp
from torch.utils.data import DataLoader, Dataset

from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from datapreprocessor.data_utils import subset_by_idx
from fl.models.model_utils import vec2model

from tqdm import tqdm


def _robust_median_mad(x: np.ndarray) -> Tuple[float, float]:
    """Return (median, MAD) on finite values; MAD is unscaled."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def _clip_vectors_by_l2_norm(vectors: np.ndarray, clip_norm: float, eps: float = 1e-12) -> np.ndarray:
    """Clip row-wise vectors so that ||v_i||_2 <= clip_norm."""
    if clip_norm is None or not np.isfinite(clip_norm) or clip_norm <= 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1) + eps
    scales = np.minimum(1.0, float(clip_norm) / norms).astype(vectors.dtype, copy=False)
    return vectors * scales.reshape(-1, 1)


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
            class_dataset = subset_by_idx(self.args, dataset, np.array(cls_indices), train=False)
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


def _median_cosine_similarities(
    vectors: np.ndarray, reference: np.ndarray, learning_rate: float
) -> List[float]:
    if vectors.size == 0:
        return []
    deltas = vectors - reference
    lr = float(learning_rate) if learning_rate else 0.0
    scaled = deltas if lr == 0.0 else deltas / lr
    norms = np.linalg.norm(scaled, axis=1)
    dot = scaled @ scaled.T
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
    cosine_similar: List[float],
    cos_threshold: float,
    method: str = "threshold",
    significance_level: float = 0.02,
) -> List[int]:
    if cosine_similar is None or len(cosine_similar) == 0:
        return []
    cosine_arr = np.asarray(cosine_similar, dtype=float)
    avg_cosine_similarity = np.mean(cosine_arr)
    method = (method or "threshold").lower()

    # Backward-compatible behavior: manual thresholding.
    if method == "threshold":
        malicious_indices = np.where(
            (cosine_arr < float(cos_threshold)) & (cosine_arr < avg_cosine_similarity)
        )[0].tolist()
        return malicious_indices

    # Robust outlier test using median/MAD with a normal tail approximation.
    if method == "mad":
        sig = float(significance_level) if significance_level is not None else 0.02
        med = float(np.median(cosine_arr))
        mad = float(np.median(np.abs(cosine_arr - med)))
        if not np.isfinite(mad) or mad <= 1e-12:
            return []
        scale = 1.4826 * mad  # consistent with std under normality
        malicious = []
        for i, s in enumerate(cosine_arr.tolist()):
            if not np.isfinite(s):
                continue
            if s >= med:
                continue
            z = (med - float(s)) / scale  # larger => more extreme low outlier
            # one-sided p-value for low tail under N(0,1)
            p_one = 0.5 * erfc(z / sqrt(2.0))
            if (p_one < sig) and (float(s) < avg_cosine_similarity):
                malicious.append(i)
        return malicious

    raise ValueError(f"Unknown cosine filter method: {method}")


def _norm_outlier_candidates(
    norms: np.ndarray,
    method: str = "none",
    k: float = 3.5,
    percentile: float = 0.95,
) -> List[int]:
    method = (method or "none").lower()
    x = np.asarray(norms, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or method == "none":
        return []

    if method == "percentile":
        q = float(percentile)
        q = min(max(q, 0.0), 1.0)
        thr = float(np.quantile(x, q))
        return np.where(norms >= thr)[0].tolist()

    if method == "mad":
        med, mad = _robust_median_mad(x)
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            return []
        thr = med + float(k) * scale
        return np.where(norms >= thr)[0].tolist()

    raise ValueError(f"Unknown norm filter method: {method}")


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
    malicious_candidates: List[int],
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
    return [malicious_candidates[i] for i in malicious_indices_rel]


def _client_reputation(
    discount: float,
    malicious: List[int],
    alpha: np.ndarray,
    beta: np.ndarray,
    *,
    alpha_inc: float = 1.0,
    beta_inc: float = 1.0,
    hard_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_last = np.array(alpha, copy=True)
    beta_last = np.array(beta, copy=True)
    alpha_update = alpha_last.copy()
    beta_update = beta_last.copy()

    num_clients = len(alpha_update)
    rep = np.ones(num_clients, dtype=float)
    malicious_set = set(malicious or [])

    for uid in range(num_clients):
        if uid in malicious_set:
            alpha_update[uid] = discount * alpha_last[uid]
            beta_update[uid] = discount * beta_last[uid] + float(beta_inc)
        else:
            alpha_update[uid] = discount * alpha_last[uid] + float(alpha_inc)
            beta_update[uid] = discount * beta_last[uid]
        denom = alpha_update[uid] + beta_update[uid]
        rep[uid] = (alpha_update[uid] / denom) if denom > 0 else 0.0
        if hard_zero and uid in malicious_set:
            rep[uid] = 0.0
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
            "cos_threshold": 0.0,
            # Step-1 candidate detection on cosine similarities:
            # - threshold: original rule using cos_threshold
            # - mad: robust significance test via MAD
            "cos_filter_method": "threshold",
            "cos_significance": 0.02,
            # Optional additional candidate signal: update norm outliers.
            # - none: disable
            # - mad: median + k*MAD
            # - percentile: keep those above quantile
            "norm_filter_method": "mad",
            "norm_filter_k": 3.5,
            "norm_filter_percentile": 0.95,
            # How to combine cosine/norm candidates.
            # - cos: only cosine candidates
            # - cos_or_norm: union(cos, norm)
            "candidate_rule": "cos_or_norm",
            # Aggregation: clip per-client delta norms before averaging.
            # If delta_clip_value > 0, it is used as a fixed clip norm.
            # Otherwise, it is derived from the current-round norm stats.
            "delta_clip_method": "mad",  # mad, percentile, none
            "delta_clip_k": 3.0,
            "delta_clip_percentile": 0.95,
            "delta_clip_value": 0.0,
            "significance": 0.02,
            "discount": 0.9,
            "reputation_threshold": 0.6,
            "epochs_phase_2": 20,
            "num_items_test": 512,
            "eval_batch_size": 128,
            # Reputation update: soften penalties to reduce clean-scenario drift.
            "rep_alpha_inc": 1.0,
            "rep_beta_inc": 1.0,
            "rep_hard_zero": False,
            # Phase-2 "gating" strategy (replacing the old hard drop + zero update):
            # - soft: don't drop participants; only use reputation as weights
            # - threshold: keep rep >= reputation_threshold
            # - topk: keep top-k fraction by reputation
            "phase2_gating": "soft",
            "phase2_topk_frac": 0.8,
            "phase2_min_keep": 1,
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
        global_epoch = int(kwargs.get("global_epoch", 0) or 0)
        self.global_model = kwargs["last_global_model"]
        global_weights_vec = kwargs["global_weights_vec"]
        updates = np.array(updates, dtype=np.float32)
        num_clients = len(updates)
        if num_clients == 0:
            zero_grad = np.zeros_like(global_weights_vec)
            return wrapup_aggregated_grads(
                zero_grad, self.args.algorithm, self.global_model, aggregated=True
            )

        if len(self.alphas) != num_clients:
            self.alphas = np.ones(num_clients, dtype=float)
            self.betas = np.ones(num_clients, dtype=float)

        reputation_before = self.alphas / (self.alphas + self.betas + 1e-12)
        participating_indices = np.arange(num_clients, dtype=int)
        if global_epoch > int(self.epochs_phase_2):
            gating = (getattr(self, "phase2_gating", "soft") or "soft").lower()
            if gating == "threshold":
                participating_indices = np.where(reputation_before >= float(self.reputation_threshold))[0].astype(int)
            elif gating == "topk":
                k_frac = float(getattr(self, "phase2_topk_frac", 0.8) or 0.8)
                k_frac = min(max(k_frac, 0.0), 1.0)
                k = max(int(getattr(self, "phase2_min_keep", 1) or 1), int(np.ceil(k_frac * num_clients)))
                order = np.argsort(-reputation_before)  # descending
                participating_indices = order[: min(k, num_clients)].astype(int)
            else:
                participating_indices = np.arange(num_clients, dtype=int)

            if participating_indices.size == 0:
                # Never return a zero update; fall back to all clients.
                participating_indices = np.arange(num_clients, dtype=int)

        updates = updates[participating_indices]
        num_participants = len(updates)

        local_model_vecs, _ = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=True
        )

        # Work in delta space for both detection and aggregation.
        deltas = local_model_vecs - global_weights_vec
        delta_norms = np.linalg.norm(deltas, axis=1)

        cosine_similarity = _median_cosine_similarities(
            local_model_vecs, global_weights_vec, self.args.learning_rate
        )
        cos_candidates = _malicious_detection_candidate(
            cosine_similarity,
            self.cos_threshold,
            method=getattr(self, "cos_filter_method", "threshold"),
            significance_level=getattr(self, "cos_significance", 0.02),
        )
        norm_candidates = _norm_outlier_candidates(
            delta_norms,
            method=getattr(self, "norm_filter_method", "none"),
            k=getattr(self, "norm_filter_k", 3.5),
            percentile=getattr(self, "norm_filter_percentile", 0.95),
        )
        rule = (getattr(self, "candidate_rule", "cos_or_norm") or "cos_or_norm").lower()
        if rule == "cos":
            malicious_candidate_index = cos_candidates
        elif rule == "cos_or_norm":
            malicious_candidate_index = sorted(set(cos_candidates).union(norm_candidates))
        else:
            raise ValueError(f"Unknown candidate_rule: {rule}")

        print("\nCosine Similarity:", cosine_similarity)
        malicious_candidate_global = (
            participating_indices[np.asarray(malicious_candidate_index, dtype=int)].tolist()
            if malicious_candidate_index
            else []
        )
        print("Detected Potential Attackers:", malicious_candidate_global)

        w_locals_malicious_candidate = [local_model_vecs[i] for i in malicious_candidate_index]
        w_locals_benign_candidate = [
            local_model_vecs[i] for i in range(num_participants) if i not in malicious_candidate_index
        ]
        if len(w_locals_benign_candidate) == 0:
            w_locals_benign_candidate = list(local_model_vecs)

        w_glob_benign_candidate = np.mean(np.stack(w_locals_benign_candidate, axis=0), axis=0)

        list_acc_local, list_loss_local = [], []
        for c in range(len(malicious_candidate_index) + 1):
            net_server_local = deepcopy(self.global_model)
            if c < len(malicious_candidate_index):
                vec2model(w_locals_malicious_candidate[c], net_server_local)
            else:
                vec2model(w_glob_benign_candidate, net_server_local)
            acc_total, loss_total = self.server_evaluator.evaluate_by_class(net_server_local)
            list_acc_local.append(acc_total)
            list_loss_local.append(loss_total)

        malicious = _client_detection(
            list_acc_local, list_loss_local, malicious_candidate_index, self.significance
        )

        malicious_global = [int(participating_indices[i]) for i in (malicious or [])]

        alpha_part = self.alphas[participating_indices]
        beta_part = self.betas[participating_indices]
        rep, alpha_part, beta_part = _client_reputation(
            self.discount,
            malicious,
            alpha_part,
            beta_part,
            alpha_inc=getattr(self, "rep_alpha_inc", 1.0),
            beta_inc=getattr(self, "rep_beta_inc", 1.0),
            hard_zero=bool(getattr(self, "rep_hard_zero", False)),
        )
        self.alphas[participating_indices] = alpha_part
        self.betas[participating_indices] = beta_part

        print("Detected Attackers:", malicious_global)
        reputation_after = self.alphas / (self.alphas + self.betas + 1e-12)
        print("Reputation:", reputation_after)

        # Aggregate in delta space with optional norm clipping.
        clip_value = float(getattr(self, "delta_clip_value", 0.0) or 0.0)
        if clip_value > 0:
            clip_norm = clip_value
        else:
            method = (getattr(self, "delta_clip_method", "mad") or "mad").lower()
            if method == "none":
                clip_norm = 0.0
            elif method == "percentile":
                q = float(getattr(self, "delta_clip_percentile", 0.95) or 0.95)
                q = min(max(q, 0.0), 1.0)
                clip_norm = float(np.quantile(delta_norms, q)) if delta_norms.size else 0.0
            elif method == "mad":
                med, mad = _robust_median_mad(delta_norms)
                scale = 1.4826 * mad
                k = float(getattr(self, "delta_clip_k", 3.0) or 3.0)
                clip_norm = float(med + k * scale) if scale > 0 else 0.0
            else:
                raise ValueError(f"Unknown delta_clip_method: {method}")

        deltas_clipped = _clip_vectors_by_l2_norm(deltas, clip_norm)
        aggregated_delta = _weighted_average_vectors(deltas_clipped, rep)
        aggregated_grad = aggregated_delta
        return wrapup_aggregated_grads(
            aggregated_grad, self.args.algorithm, self.global_model, aggregated=True
        )
