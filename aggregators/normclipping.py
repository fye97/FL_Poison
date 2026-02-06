from copy import deepcopy

import numpy as np
import torch
from aggregators import aggregator_registry
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import addnoise, normclipping, prepare_grad_updates, wrapup_aggregated_grads
from fl.models.model_utils import model2vec


@aggregator_registry
class NormClipping(AggregatorBase):
    """
    [Can You Really Backdoor Federated Learning](https://arxiv.org/abs/1911.07963) - NeurIPS '20
    It clips the norm of each client gradient updates by a threshold
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        norm_threshold (float): the threshold for clipping the norm of the updates
        """
        self.default_defense_params = {
            "weakDP": False, "norm_threshold": 3, "noise_mean": 0, "noise_std": 0.002}
        self.update_and_set_attr()

        self.algorithm = 'FedOpt'
        self.use_torch = True

    def aggregate(self, updates, **kwargs):
        # 1. prepare model updates, gradient updates, output layers of gradient updates
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        if torch.is_tensor(updates):
            if self.args.algorithm == "FedAvg":
                global_vec = model2vec(self.global_model, return_torch=True)
                gradient_updates = updates - global_vec
            else:
                gradient_updates = updates

            norms = torch.linalg.norm(gradient_updates, dim=1)
            scales = torch.clamp(self.norm_threshold / (norms + 1e-12), max=1.0)
            normed_updates = gradient_updates * scales.unsqueeze(1)
            if self.weakDP:
                noise = torch.normal(
                    mean=self.noise_mean,
                    std=self.noise_std,
                    size=normed_updates.shape,
                    device=normed_updates.device,
                    dtype=normed_updates.dtype,
                )
                normed_updates = normed_updates + noise

            aggregated_gradient = torch.mean(normed_updates, dim=0)
            if self.args.algorithm == "FedAvg":
                return global_vec + aggregated_gradient
            return aggregated_gradient

        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        normed_updates = normclipping(gradient_updates, self.norm_threshold)
        # add noise to clients' updates
        if self.weakDP:
            normed_updates = addnoise(
                normed_updates,  self.noise_mean, self.noise_std)

        return wrapup_aggregated_grads(normed_updates, self.args.algorithm, self.global_model)
