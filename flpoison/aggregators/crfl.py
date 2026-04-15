from flpoison.aggregators.aggregator_utils import addnoise, prepare_grad_updates, wrapup_aggregated_grads
from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from flpoison.aggregators import aggregator_registry


@aggregator_registry
class CRFL(AggregatorBase):
    """
    [CRFL: Certifiably Robust Federated Learning against Backdoor Attacks](http://proceedings.mlr.press/v139/xie21a/xie21a.pdf)
    CRFL apply parameters clipping and perturbing to mean aggregated update
    """

    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = 'FedOpt'
        self.default_defense_params = {
            "norm_threshold": 3, "noise_mean": 0, "noise_std": 0.001}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        # 1. prepare model updates, gradient updates, output layers of gradient updates
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        global_weights_vec = kwargs.get("global_weights_vec")
        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model, global_weights_vec=global_weights_vec)

        # 1. aggregate the gradient updates
        if torch.is_tensor(gradient_updates):
            agg_update = gradient_updates.mean(dim=0)
        else:
            agg_update = np.mean(gradient_updates, axis=0)
        # 2. norm clip the updates
        if torch.is_tensor(agg_update):
            agg_norm = torch.linalg.vector_norm(agg_update)
            if float(agg_norm.item()) > 0.0:
                normed_agg_update = agg_update * min(
                    1.0, float(self.norm_threshold) / (float(agg_norm.item()) + 1e-10)
                )
            else:
                normed_agg_update = agg_update
        else:
            normed_agg_update = agg_update * \
                min(1, self.norm_threshold / (np.linalg.norm(agg_update)+1e-10))

        # 3. add gaussian noise, note that the noise should be float32 to be consistent with the future torch dtype
        return wrapup_aggregated_grads(addnoise(normed_agg_update,  self.noise_mean, self.noise_std), self.args.algorithm, self.global_model, aggregated=True, global_weights_vec=global_weights_vec)
