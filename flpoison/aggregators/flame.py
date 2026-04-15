import torch
from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import hdbscan
from flpoison.aggregators import aggregator_registry
from flpoison.aggregators.aggregator_utils import normclipping, prepare_updates
from flpoison.fl.models import get_model
from flpoison.fl.models.model_utils import model2vec, vec2model


@aggregator_registry
class FLAME(AggregatorBase):
    """
    [FLAME: Taming Backdoors in Federated Learning](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen) - USENIX Security '22
    FLAME first clusters the cosine distance between client updates with hdbscan, then clips the benign gradients by the median of norms, and finally adds noise to meet the requirements of differential privacy.
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = "FedAvg"
        self.default_defense_params = {"gamma": 1.2e-5}
        self.update_and_set_attr()
        self.noise_model = None

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs['last_global_model']
        global_weights_vec = kwargs.get("global_weights_vec")
        if global_weights_vec is None:
            global_weights_vec = model2vec(
                self.global_model, return_torch=torch.is_tensor(updates))
        model_updates, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, global_weights_vec=global_weights_vec)
        benign_idx = self.cosine_clustering(model_updates)
        aggregated_model_vec, median_norm = self.adpative_clipping(
            global_weights_vec, gradient_updates, benign_idx)
        if self.noise_model is None:
            self.noise_model = get_model(self.args)
        vec2model(aggregated_model_vec, self.noise_model)
        self.add_noise2model(self.gamma * median_norm, self.noise_model)
        aggregated_model_vec = model2vec(
            self.noise_model, return_torch=torch.is_tensor(global_weights_vec))

        if self.args.algorithm == 'FedAvg':
            return aggregated_model_vec
        if torch.is_tensor(aggregated_model_vec):
            base = global_weights_vec if torch.is_tensor(
                global_weights_vec) else torch.as_tensor(
                global_weights_vec,
                device=aggregated_model_vec.device,
                dtype=aggregated_model_vec.dtype,
            )
            return aggregated_model_vec - base.reshape(-1)
        return aggregated_model_vec - np.asarray(global_weights_vec)

    def cosine_clustering(self, model_updates):
        """
        clustering the cosine distance between client updates with hdbscan
        """
        if torch.is_tensor(model_updates):
            model_updates = model_updates.to(dtype=torch.float64)
            normalized_updates = torch.nn.functional.normalize(
                model_updates, p=2, dim=1, eps=1e-12)
            cosine_dists = (
                1.0 - torch.clamp(normalized_updates @ normalized_updates.T, min=-1.0, max=1.0)
            ).detach().cpu().numpy()
            cluster = hdbscan.HDBSCAN(
                metric="precomputed",
                algorithm="generic",
                min_cluster_size=self.args.num_clients // 2 + 1,
                min_samples=1,
                allow_single_cluster=True,
            )
            cluster.fit(cosine_dists)
        else:
            clustering_input = model_updates.astype(np.float64)
            cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                                      min_cluster_size=self.args.num_clients//2+1, min_samples=1, allow_single_cluster=True)
            cluster.fit(clustering_input)
        # choose which cluster is benign
        return [idx for idx, label in enumerate(cluster.labels_) if label == 0]

    def adpative_clipping(self, last_global_weights_vec, gradient_updates, benign_idx):
        """
        clipping threshold is the median of l2 distance between last global model and current clients updates
        """
        if len(benign_idx) == 0:
            benign_idx = list(range(len(gradient_updates)))
        # 1. get median of l2 norm
        if torch.is_tensor(gradient_updates):
            gradient_norms = torch.linalg.vector_norm(gradient_updates, dim=1)
            median_norm = float(torch.median(gradient_norms).item())
            benign_idx_tensor = torch.as_tensor(
                benign_idx, device=gradient_updates.device, dtype=torch.long)
            benign_updates = gradient_updates.index_select(0, benign_idx_tensor)
        else:
            median_norm = np.median(np.linalg.norm(gradient_updates, axis=1))
            benign_updates = gradient_updates[benign_idx]
        # 2. clip the benign gradients by median of norms
        clipped_gradient_updates = normclipping(
            benign_updates, median_norm)
        # 3. calculate the mean of clipped benign gradient updates and add them to the last global model for aggregation
        if torch.is_tensor(clipped_gradient_updates):
            aggregated_gradient = torch.mean(clipped_gradient_updates, dim=0)
            base = last_global_weights_vec if torch.is_tensor(
                last_global_weights_vec) else torch.as_tensor(
                last_global_weights_vec,
                device=aggregated_gradient.device,
                dtype=aggregated_gradient.dtype,
            )
            aggregated_model_vec = base.reshape(-1) + aggregated_gradient
        else:
            aggregated_gradient = np.mean(clipped_gradient_updates, axis=0)
            aggregated_model_vec = np.asarray(
                last_global_weights_vec).reshape(-1) + aggregated_gradient
        return aggregated_model_vec, median_norm

    def add_noise2model(self, noise_scale, model, only_weights=True):
        # add gaussian noise to the model ignoring bias and batch normalization layers
        with torch.no_grad():
            for key, param in model.state_dict().items():
                if only_weights and any(
                    substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']
                ):
                    continue
                if not torch.is_floating_point(param):
                    continue
                std = float(noise_scale) * float(param.std().item())
                if std <= 0.0:
                    continue
                param.add_(torch.normal(
                    mean=0.0,
                    std=std,
                    size=tuple(param.shape),
                    device=param.device,
                    dtype=param.dtype,
                ))
