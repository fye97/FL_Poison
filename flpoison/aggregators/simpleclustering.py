import numpy as np
import torch
from flpoison.aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from flpoison.aggregators.aggregatorbase import AggregatorBase
from flpoison.aggregators import aggregator_registry
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

@aggregator_registry
class SimpleClustering(AggregatorBase):
    """
    Simple majority clustering based on gradient updates.
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "clustering": "DBSCAN"}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        global_weights_vec = kwargs.get("global_weights_vec")
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model, global_weights_vec=global_weights_vec)
        clustering_input = gradient_updates.detach().cpu().numpy(
        ) if torch.is_tensor(gradient_updates) else gradient_updates

        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(
                clustering_input, quantile=0.5, n_samples=50)
            grad_cluster = MeanShift(bandwidth=bandwidth,
                                     bin_seeding=True, cluster_all=False)
        elif self.clustering == "DBSCAN":
            grad_cluster = DBSCAN(eps=0.05, min_samples=3)

        grad_cluster.fit(clustering_input)
        labels = grad_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        if n_cluster <= 0:
            return wrapup_aggregated_grads(
                gradient_updates,
                self.args.algorithm,
                self.global_model,
                global_weights_vec=global_weights_vec,
            )
        # select the cluster with the majority of benign clients
        benign_label = np.argmax([np.sum(labels == i)
                                 for i in range(n_cluster)])
        benign_idx = np.argwhere(labels == benign_label).reshape(-1).tolist()

        if torch.is_tensor(gradient_updates):
            benign_idx_tensor = torch.as_tensor(
                benign_idx, device=gradient_updates.device, dtype=torch.long)
            benign_updates = gradient_updates.index_select(0, benign_idx_tensor)
        else:
            benign_updates = gradient_updates[benign_idx]
        return wrapup_aggregated_grads(benign_updates, self.args.algorithm, self.global_model, global_weights_vec=global_weights_vec)
