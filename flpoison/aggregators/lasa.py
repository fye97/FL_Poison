from flpoison.aggregators.aggregatorbase import AggregatorBase
import numpy as np
import torch
from flpoison.aggregators import aggregator_registry
from flpoison.fl.models.model_utils import state2vec, vec2state


@aggregator_registry
class LASA(AggregatorBase):
    supports_torch_updates = True

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "norm_bound": 1, "sign_bound": 1, "sparsity": 0.3}  # CIRAR10/100 1,1, otherwise norm_bound=2
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        num_clients = len(updates)
        self.global_model = kwargs['last_global_model']
        use_torch = torch.is_tensor(updates)
        # state dict form of the updates with corresponding values not flattened
        dict_form_updates = []
        for i in range(num_clients):
            dict_form_updates.append(
                vec2state(updates[i], self.global_model, numpy=not use_torch))

        # 1. clip and scale based on median of norms of clients
        if use_torch:
            client_norms = torch.linalg.vector_norm(updates, dim=1)
            median_norm = torch.median(client_norms)
            grads_clipped_norm = torch.clamp(client_norms, min=0.0, max=median_norm)
            grad_clipped = (updates / client_norms.clamp_min(1e-12).unsqueeze(1)
                            ) * grads_clipped_norm.unsqueeze(1)
        else:
            client_norms = np.linalg.norm(updates, axis=1)
            median_norm = np.median(client_norms)
            grads_clipped_norm = np.clip(client_norms, a_min=0, a_max=median_norm)
            grad_clipped = (updates / client_norms.reshape(-1, 1)
                            ) * grads_clipped_norm.reshape(-1, 1)

        dict_form_grad_clipped = [
            vec2state(grad_clipped[i], self.global_model, numpy=not use_torch) for i in range(num_clients)]

        # 1. Sparse each client's update with top-k largest strategy individually before aggregation
        for i in range(len(dict_form_updates)):
            dict_form_updates[i] = self.sparse_update(dict_form_updates[i])

        # for each layer
        key_mean_weight = {}
        for key in dict_form_updates[0].keys():
            if 'num_batches_tracked' in key:
                continue
            # 2. get the flattened gradient updates of the key
            if use_torch:
                key_flattened_updates = torch.stack(
                    [dict_form_updates[i][key].reshape(-1)
                     for i in range(num_clients)],
                    dim=0,
                )
            else:
                key_flattened_updates = np.array([dict_form_updates[i][key].flatten()
                                                  for i in range(num_clients)])

            # 3. magnitude filtering based on norm and MZ-score (Median Z-score)
            grad_l2norm = (
                torch.linalg.vector_norm(key_flattened_updates, dim=1)
                if use_torch
                else np.linalg.norm(key_flattened_updates, axis=1)
            )
            S1_benign_idx = self.mz_score(grad_l2norm, self.norm_bound)

            # 4. direction filtering based on sign and  MZ-score (Median Z-score)
            layer_signs = (
                torch.empty(num_clients, device=updates.device, dtype=updates.dtype)
                if use_torch
                else np.empty(num_clients)
            )
            for i in range(num_clients):
                sign_feat = torch.sign(
                    dict_form_updates[i][key]) if use_torch else np.sign(dict_form_updates[i][key])
                if use_torch:
                    denom = torch.sum(torch.abs(sign_feat))
                    if float(denom.item()) == 0.0:
                        layer_signs[i] = 0.0
                    else:
                        layer_signs[i] = 0.5 * torch.sum(sign_feat) / denom * \
                            (1 - self.sparsity)
                else:
                    denom = np.sum(np.abs(sign_feat))
                    layer_signs[i] = 0.0 if denom == 0 else 0.5 * \
                        np.sum(sign_feat) / denom * (1 - self.sparsity)
            S2_benign_idx = self.mz_score(layer_signs, self.sign_bound)
            benign_idx = list(set(S1_benign_idx).intersection(S2_benign_idx))
            benign_idx = benign_idx if len(
                benign_idx) != 0 else list(range(num_clients))
            # layer-wise aggregation
            if use_torch:
                key_mean_weight[key] = torch.mean(
                    torch.stack([dict_form_grad_clipped[i][key]
                                for i in benign_idx], dim=0),
                    dim=0,
                )
            else:
                key_mean_weight[key] = np.mean(
                    [dict_form_grad_clipped[i][key] for i in benign_idx], axis=0)

        return state2vec(key_mean_weight, numpy_flg=not use_torch, return_torch=use_torch)

    def sparse_update(self, update):
        """
        This function sparsifies the convlution and full-connection layer of updates of each client based on the top-k largest sparsification strategy
        """
        use_torch = any(torch.is_tensor(value) for value in update.values())
        # 1. initialize the sparsity mask
        mask = {}
        for key in update.keys():
            if len(update[key].shape) == 4 or len(update[key].shape) == 2:
                # Need to change the dtype, but now only for testing
                mask[key] = torch.ones_like(
                    update[key], dtype=torch.float32) if use_torch else np.ones_like(
                    update[key], dtype=np.float32)
        if self.sparsity == 0.0:
            return update
        # 2. filter the top-k largest values for each key
        weight_abs = [torch.abs(update[key]) if use_torch else np.abs(update[key])
                      for key in update.keys() if key in mask]
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([value.reshape(-1) for value in weight_abs], dim=0) if use_torch else np.concatenate(
            [value.flatten() for value in weight_abs])
        num_topk = int(len(all_scores) * (1 - self.sparsity))
        if num_topk <= 0:
            for key in mask.keys():
                if use_torch:
                    update[key].zero_()
                else:
                    update[key].fill(0)
            return update
        # top-k largest values
        if use_torch:
            kth_largest = torch.topk(
                all_scores, k=num_topk, largest=True).values[-1]
        else:
            kth_largest = np.partition(
                all_scores, -num_topk)[-num_topk]

        # 3. update the mask by setting the values smaller than the threshold to 0
        for key in mask.keys():
            # must be > to prevent acceptable_score is zero, leading to dense tensors
            if use_torch:
                mask[key] = torch.where(
                    torch.abs(update[key]) <= kth_largest,
                    torch.zeros_like(mask[key]),
                    mask[key],
                )
            else:
                mask[key] = np.where(
                    np.abs(update[key]) <= kth_largest, 0, mask[key])

            # 4. apply the mask to the updates
            if use_torch:
                update[key].mul_(mask[key].to(dtype=update[key].dtype))
            else:
                update[key] *= mask[key]

        return update

    def mz_score(self, values, bound):
        if torch.is_tensor(values):
            med = torch.median(values)
            std = torch.std(values, correction=0)
            z_scores = torch.abs((values - med) / (std + 1e-12))
            return torch.nonzero(z_scores < bound, as_tuple=False).squeeze(-1).tolist()
        med, std = np.median(values), np.std(values)
        z_scores = np.abs((values - med) / (std + 1e-12))
        return np.argwhere(z_scores < bound).squeeze(-1).tolist()
