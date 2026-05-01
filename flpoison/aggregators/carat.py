import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from flpoison.aggregators import aggregator_registry
from flpoison.aggregators.aggregator_utils import (
    prepare_updates,
    wrapup_aggregated_grads,
)
from flpoison.aggregators.aggregatorbase import AggregatorBase
from flpoison.datapreprocessor.data_utils import subset_by_idx
from flpoison.fl.models import get_model
from flpoison.fl.models.model_utils import vec2model
from flpoison.utils.global_utils import log_file_only


def _to_numpy(vector) -> np.ndarray:
    if torch.is_tensor(vector):
        return vector.detach().cpu().numpy()
    return np.asarray(vector)


def _nan_to_num(vector):
    if torch.is_tensor(vector):
        return torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _get_dataset_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "train_labels"):
        labels = dataset.train_labels
    else:
        return np.array([label for _, label in dataset])

    if isinstance(labels, list):
        return np.asarray(labels)
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def _sample_balanced_indices(dataset: Dataset, num_samples: int) -> np.ndarray:
    labels = _get_dataset_labels(dataset)
    if labels.size == 0:
        return np.array([], dtype=np.int64)
    if num_samples <= 0:
        return np.arange(len(labels), dtype=np.int64)

    classes = np.unique(labels)
    per_class = max(1, int(num_samples // max(1, len(classes))))
    remainder = max(0, int(num_samples) - per_class * len(classes))

    sampled = []
    for cls_rank, cls in enumerate(classes):
        cls_indices = np.where(labels == cls)[0]
        if cls_indices.size == 0:
            continue
        take = per_class + (1 if cls_rank < remainder else 0)
        replace = take > cls_indices.size
        chosen = np.random.choice(cls_indices, size=take, replace=replace)
        sampled.extend(chosen.tolist())
    np.random.shuffle(sampled)
    return np.asarray(sampled, dtype=np.int64)


def _robust_median_mad(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def _compute_clip_norm(norms: np.ndarray, method: str, k: float, percentile: float) -> float:
    x = np.asarray(norms, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    method = (method or "mad").lower()
    if method == "none":
        return 0.0
    if method == "percentile":
        q = min(max(float(percentile), 0.0), 1.0)
        return float(np.quantile(x, q))
    if method == "mad":
        med, mad = _robust_median_mad(x)
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1.0e-12:
            return float(np.max(x))
        return float(med + float(k) * scale)
    raise ValueError(f"Unknown clipping method: {method}")


def _clip_vectors_by_l2_norm(vectors, clip_norm: float, eps: float = 1.0e-12):
    if clip_norm <= 0.0 or not np.isfinite(clip_norm):
        return vectors
    if torch.is_tensor(vectors):
        norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        scales = torch.clamp(
            torch.as_tensor(float(clip_norm), device=vectors.device, dtype=vectors.dtype)
            / (norms + eps),
            max=1.0,
        )
        return vectors * scales
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scales = np.minimum(1.0, float(clip_norm) / (norms + eps))
    return vectors * scales


def _weighted_average_vectors(vectors, weights):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if torch.is_tensor(vectors):
        weight_tensor = torch.as_tensor(weights, device=vectors.device, dtype=vectors.dtype)
        total = float(weight_tensor.sum().item())
        if total <= 0.0:
            return torch.mean(vectors, dim=0)
        return torch.sum(vectors * weight_tensor.unsqueeze(1), dim=0) / weight_tensor.sum()
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.mean(vectors, axis=0)
    return np.average(vectors, axis=0, weights=weights)


def _normalize_to_radius(vectors, radius: float, eps: float = 1.0e-12):
    if radius <= 0.0 or not np.isfinite(radius):
        return vectors * 0.0
    if torch.is_tensor(vectors):
        norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        return vectors * (float(radius) / (norms + eps))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * (float(radius) / (norms + eps))


def _resolve_rho_hat(num_clients: int, num_adv, default_rho_hat: float) -> float:
    if num_clients <= 1:
        return 0.0
    if num_adv is None:
        rho_hat = float(default_rho_hat)
    elif float(num_adv) < 1.0:
        rho_hat = float(num_adv)
    else:
        rho_hat = float(num_adv) / float(num_clients)
    return min(max(rho_hat, 0.0), 0.95)


def _project_capped_simplex(y: np.ndarray, upper: float, tol: float = 1.0e-9, max_iter: int = 80) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    if n == 0:
        return y

    upper = min(max(float(upper), 1.0 / n), 1.0)
    if upper * n < 1.0:
        return np.full(n, 1.0 / n, dtype=np.float64)

    lo = float(np.min(y - upper))
    hi = float(np.max(y))
    projected = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        theta = 0.5 * (lo + hi)
        projected = np.clip(y - theta, 0.0, upper)
        total = float(np.sum(projected))
        if abs(total - 1.0) <= tol:
            break
        if total > 1.0:
            lo = theta
        else:
            hi = theta

    total = float(np.sum(projected))
    if total <= 0.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return projected / total


def _softmin_value_and_task_weights(task_scores: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    scores = np.asarray(task_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return 0.0, np.zeros(0, dtype=np.float64)

    beta = max(float(beta), 1.0e-6)
    logits = -beta * scores
    shift = float(np.max(logits))
    exp_logits = np.exp(logits - shift)
    probs = exp_logits / np.sum(exp_logits)
    logsumexp = shift + np.log(np.sum(exp_logits))
    softmin = -(logsumexp - np.log(scores.size)) / beta
    return float(softmin), probs


def _robust_standardize_columns(matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    center = np.median(arr, axis=0)
    scale = 1.4826 * np.median(np.abs(arr - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)
    return (arr - center) / scale


def _robust_positive_penalty(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    med, mad = _robust_median_mad(arr)
    scale = max(1.4826 * mad, eps)
    return np.maximum((arr - med) / scale, 0.0)


def _sample_rank_penalties_once(deltas: np.ndarray, num_coords: int) -> np.ndarray:
    num_clients, dim = deltas.shape
    if num_clients <= 1 or dim == 0:
        return np.zeros(num_clients, dtype=np.float64)

    coord_count = min(max(1, int(num_coords)), dim)
    if coord_count == dim:
        coord_idx = np.arange(dim)
    else:
        coord_idx = np.random.choice(dim, size=coord_count, replace=False)

    sampled = deltas[:, coord_idx]
    ranks = np.argsort(np.argsort(sampled, axis=0), axis=0).astype(np.float64)
    denom = max(1, num_clients - 1)
    normalized_ranks = ranks / denom
    return np.mean(np.abs(normalized_ranks - 0.5), axis=1)


def _ensemble_rank_penalties(deltas: np.ndarray, num_coords: int, num_subsamples: int) -> np.ndarray:
    if deltas.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    penalties = [
        _sample_rank_penalties_once(deltas, num_coords)
        for _ in range(max(1, int(num_subsamples)))
    ]
    return np.median(np.stack(penalties, axis=0), axis=0)


def _mean_ce_loss(model, loader, device, pin_memory: bool) -> float:
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            logits = model(inputs)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            total_examples += int(labels.numel())
    if total_examples <= 0:
        return 0.0
    return total_loss / total_examples


@aggregator_registry
class CARAT(AggregatorBase):
    """
    CARAT: Class-Aware Robust Aggregation via hidden Task certificates.

    It solves a capped-simplex weight optimization problem whose objective
    encourages strong worst-task hidden probe descent, penalizes rank-space
    outliers, and regularizes weights towards the previous round.
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "num_probe_pool": 512,
            "num_probe_tasks": 8,
            "probe_samples_per_class": 4,
            "probe_task_batch_size": 128,
            "probe_num_workers": 0,
            "probe_radius_quantile": 0.5,
            "clip_method": "mad",
            "clip_k": 2.5,
            "clip_percentile": 0.9,
            "rho_hat": None,
            "certificate_beta": 12.0,
            "rank_weight": 0.20,
            "prior_weight": 0.05,
            "rank_num_coords": 4096,
            "rank_num_subsamples": 5,
            "optimizer_steps": 60,
            "optimizer_lr": 0.5,
        }
        self.update_and_set_attr()

        train_dataset = kwargs.get("train_dataset")
        if train_dataset is None:
            raise ValueError("CARAT requires train_dataset to construct hidden probe tasks.")

        self.train_dataset = train_dataset
        self.probe_pool_indices = _sample_balanced_indices(train_dataset, int(self.num_probe_pool))
        labels = _get_dataset_labels(train_dataset)
        self.probe_class_indices = {}
        for cls in range(int(self.args.num_classes)):
            cls_indices = self.probe_pool_indices[labels[self.probe_pool_indices] == cls]
            if cls_indices.size > 0:
                self.probe_class_indices[cls] = cls_indices

        self.probe_model = get_model(self.args)
        num_clients = max(1, int(self.args.num_clients))
        self.previous_alpha = np.full(num_clients, 1.0 / num_clients, dtype=np.float64)

    def _sample_task_indices(self) -> List[np.ndarray]:
        tasks = []
        samples_per_class = max(1, int(self.probe_samples_per_class))
        for _ in range(max(1, int(self.num_probe_tasks))):
            indices = []
            for cls in range(int(self.args.num_classes)):
                cls_pool = self.probe_class_indices.get(cls)
                if cls_pool is None or cls_pool.size == 0:
                    continue
                replace = cls_pool.size < samples_per_class
                chosen = np.random.choice(cls_pool, size=samples_per_class, replace=replace)
                indices.extend(chosen.tolist())
            if indices:
                np.random.shuffle(indices)
                tasks.append(np.asarray(indices, dtype=np.int64))

        if tasks:
            return tasks
        if self.probe_pool_indices.size == 0:
            return []

        fallback_size = min(
            int(self.probe_pool_indices.size),
            max(1, int(self.probe_samples_per_class) * max(1, int(self.args.num_classes))),
        )
        fallback = np.random.choice(self.probe_pool_indices, size=fallback_size, replace=False)
        return [np.asarray(fallback, dtype=np.int64)]

    def _build_task_loaders(self):
        task_indices_list = self._sample_task_indices()
        if not task_indices_list:
            return []

        pin_memory = str(getattr(self.args.device, "type", self.args.device)).startswith("cuda")
        loaders = []
        for task_indices in task_indices_list:
            task_dataset = subset_by_idx(self.args, self.train_dataset, task_indices, train=False)
            batch_size = min(max(1, int(self.probe_task_batch_size)), len(task_dataset))
            loader = DataLoader(
                task_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=max(0, int(self.probe_num_workers)),
                pin_memory=pin_memory,
            )
            loaders.append(loader)
        return loaders

    def _compute_hidden_loss_certificates(self, global_weights_vec, eval_deltas) -> tuple[np.ndarray, np.ndarray]:
        task_loaders = self._build_task_loaders()
        num_clients = len(eval_deltas)
        if not task_loaders:
            return np.zeros((num_clients, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

        pin_memory = str(getattr(self.args.device, "type", self.args.device)).startswith("cuda")
        vec2model(global_weights_vec, self.probe_model)
        self.probe_model.to(self.args.device)
        self.probe_model.eval()
        baseline_losses = np.array(
            [_mean_ce_loss(self.probe_model, loader, self.args.device, pin_memory) for loader in task_loaders],
            dtype=np.float64,
        )

        certificates = np.zeros((num_clients, len(task_loaders)), dtype=np.float64)
        if torch.is_tensor(global_weights_vec):
            base_vec = global_weights_vec.detach().reshape(-1)
        else:
            base_vec = np.asarray(global_weights_vec).reshape(-1)

        for client_idx in range(num_clients):
            delta = eval_deltas[client_idx]
            candidate_vec = base_vec + delta
            vec2model(candidate_vec, self.probe_model)
            self.probe_model.eval()
            for task_idx, loader in enumerate(task_loaders):
                task_loss = _mean_ce_loss(self.probe_model, loader, self.args.device, pin_memory)
                certificates[client_idx, task_idx] = baseline_losses[task_idx] - task_loss

        return certificates, baseline_losses

    def _optimize_weights(self, certificates: np.ndarray, rank_penalties: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_clients = certificates.shape[0]
        num_tasks = certificates.shape[1]
        if num_clients == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(num_tasks, dtype=np.float64)

        if len(self.previous_alpha) != num_clients:
            self.previous_alpha = np.full(num_clients, 1.0 / num_clients, dtype=np.float64)

        default_rho_hat = 0.2 if self.rho_hat is None else float(self.rho_hat)
        rho_hat = _resolve_rho_hat(num_clients, getattr(self.args, "num_adv", None), default_rho_hat)
        upper = 1.0 / max((1.0 - rho_hat) * num_clients, 1.0)

        alpha = _project_capped_simplex(self.previous_alpha, upper)
        task_weights = np.full(num_tasks, 1.0 / max(1, num_tasks), dtype=np.float64)
        for _ in range(max(1, int(self.optimizer_steps))):
            task_scores = certificates.T @ alpha
            _, task_weights = _softmin_value_and_task_weights(task_scores, float(self.certificate_beta))
            grad = certificates @ task_weights
            grad -= float(self.rank_weight) * rank_penalties
            grad -= 2.0 * float(self.prior_weight) * (alpha - self.previous_alpha)
            alpha = _project_capped_simplex(alpha + float(self.optimizer_lr) * grad, upper)

        return alpha, certificates.T @ alpha

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs["last_global_model"]
        global_weights_vec = kwargs["global_weights_vec"]
        use_torch = torch.is_tensor(updates)
        if not use_torch:
            updates = np.asarray(updates, dtype=np.float32)

        num_clients = len(updates)
        if num_clients == 0:
            zero = torch.zeros_like(global_weights_vec) if torch.is_tensor(global_weights_vec) else np.zeros_like(global_weights_vec)
            return wrapup_aggregated_grads(
                zero,
                self.args.algorithm,
                self.global_model,
                aggregated=True,
                global_weights_vec=global_weights_vec,
            )

        with self.profile_substage("defense"):
            _, deltas = prepare_updates(
                self.args.algorithm,
                updates,
                self.global_model,
                vector_form=True,
                global_weights_vec=global_weights_vec,
            )
            deltas = _nan_to_num(deltas)
            deltas_np = _to_numpy(deltas).astype(np.float64, copy=False)

            norms = np.linalg.norm(deltas_np, axis=1)
            clip_norm = _compute_clip_norm(norms, self.clip_method, float(self.clip_k), float(self.clip_percentile))
            deltas_clipped = _clip_vectors_by_l2_norm(deltas, clip_norm)
            deltas_clipped = _nan_to_num(deltas_clipped)
            deltas_clipped_np = _to_numpy(deltas_clipped).astype(np.float64, copy=False)
            clipped_norms = np.linalg.norm(deltas_clipped_np, axis=1)

            positive_norms = clipped_norms[clipped_norms > 1.0e-12]
            if positive_norms.size:
                radius_q = min(max(float(self.probe_radius_quantile), 0.0), 1.0)
                eval_radius = float(np.quantile(positive_norms, radius_q))
            else:
                eval_radius = 0.0
            eval_deltas = _normalize_to_radius(deltas_clipped, eval_radius)

            certificates_raw, baseline_losses = self._compute_hidden_loss_certificates(global_weights_vec, eval_deltas)
            certificates = _robust_standardize_columns(certificates_raw)
            rank_penalties_raw = _ensemble_rank_penalties(
                deltas_clipped_np,
                int(self.rank_num_coords),
                int(self.rank_num_subsamples),
            )
            rank_penalties = _robust_positive_penalty(rank_penalties_raw)
            alpha, task_scores = self._optimize_weights(certificates, rank_penalties)
            self.previous_alpha = alpha

            softmin_value, task_focus = _softmin_value_and_task_weights(task_scores, float(self.certificate_beta))
            suspicious = np.argsort(alpha)[: max(1, min(num_clients // 5, num_clients))].tolist()
            log_file_only(
                self.args.logger,
                logging.INFO,
                "CARAT alpha=%s rank_penalties=%s raw_rank=%s suspicious=%s",
                np.round(alpha, 4).tolist(),
                np.round(rank_penalties, 4).tolist(),
                np.round(rank_penalties_raw, 4).tolist(),
                suspicious,
            )
            log_file_only(
                self.args.logger,
                logging.INFO,
                "CARAT task_scores=%s softmin=%.6f task_focus=%s baseline_losses=%s eval_radius=%.6f clip_norm=%.6f",
                np.round(task_scores, 4).tolist(),
                float(softmin_value),
                np.round(task_focus, 4).tolist(),
                np.round(baseline_losses, 4).tolist(),
                float(eval_radius),
                float(clip_norm),
            )

        with self.profile_substage("aggregate"):
            aggregated_delta = _weighted_average_vectors(deltas_clipped, alpha)
            return wrapup_aggregated_grads(
                aggregated_delta,
                self.args.algorithm,
                self.global_model,
                aggregated=True,
                global_weights_vec=global_weights_vec,
            )
