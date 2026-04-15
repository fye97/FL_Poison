import numpy as np
import torch
from flpoison.attackers.pbases.dpbase import DPBase
from flpoison.attackers.pbases.mpbase import MPBase
from .synthesizers import PixelSynthesizer
from flpoison.datapreprocessor.data_utils import subset_by_idx
from flpoison.utils.global_utils import actor
from flpoison.fl.models.model_utils import gradient2vector, model2vec, vec2model
from flpoison.attackers import attacker_registry
from flpoison.fl.client import Client
from flpoison.fl.models import get_model


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning')
class Neurotoxin(MPBase, DPBase, Client):
    """
    [Neurotoxin: Durable Backdoors in Federated Learning](https://proceedings.mlr.press/v162/zhang22w.html) - ICML '22
    Neurotoxin relies on the infrequent updated coordinates by benign clients to hide the backdoor. Specifically, it first get the top-k smallest absolute gradient values of the global model, and then apply the gradient mask to the local model. The gradient mask is used to project the gradient of the local model to the infrequent updated coordinates of the global model. In addition, Neurotoxin applies gradient norm clipping to prevent the model from being updated too much.
    """

    supports_torch_updates = True

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        num_sample: number of benign samples to train for calculating the gradient mask
        topk_ratio: ratio of top-k smallest absolute gradient values
        norm_threshold: clipping threshold of gradient norm 
        """
        self.default_attack_params = {
            'num_sample': 64, 'topk_ratio': 0.1, 'norm_threshold': 0.2, "attack_model": "all2one", "poisoning_ratio": 0.32, "target_label": 6, "source_label": 1, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5}

        self.update_and_set_attr()

        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)
        self.algorithm = 'FedSGD'
        self.mask_model = get_model(args)

    def define_synthesizer(self):
        # for pixel-type trigger, specify the trigger tensor
        self.trigger = torch.ones((1, 5, 5))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=self.poisoning_ratio, source_label=self.source_label, single_epoch=self.single_epoch)

    def local_training(self):
        # 1. get unfrequently-used gradient mask via global model and clean data
        self.grad_mask_vec = self.get_gradient_mask()
        # 2. backdoor training while apply the gradient mask and PGD in step()
        train_acc, train_loss, train_samples = super().local_training()
        return train_acc, train_loss, train_samples

    def get_gradient_mask(self):
        """
        get the coordinates that are not frequently updated by the rest of the benign users with the last global model and sampled local clean data
        return gradient mask
        """
        # benign training to get the frequently-updated coordinates by benign clients
        # 1. sample `self.num_sample` clean data, and do benign training on global model
        sample_indices = np.random.choice(
            range(len(self.train_dataset)), size=self.num_sample, replace=False)
        sampled_dataset = subset_by_idx(
            self.args, self.train_dataset, sample_indices)
        sampled_loader = torch.utils.data.DataLoader(
            sampled_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=getattr(getattr(self.args, "device", None), "type", None) == "cuda",
        )
        new_model = self.mask_model
        vec2model(self.global_weights_tensor, new_model)
        new_optimizer, lr_scheduler = self.get_optimizer_scheduler(new_model)
        super().train(new_model, iter(sampled_loader), new_optimizer)

        # 2 get vectorized gradient from the above training
        grad_vec = gradient2vector(new_model, return_torch=True)

        # 3. get the indices of top-k smallest absolute gradient value as the gradient mask
        k = int(grad_vec.numel() * self.topk_ratio)
        grad_mask_vec = torch.zeros_like(grad_vec)
        if k > 0:
            idx = torch.topk(torch.abs(grad_vec), k=k, largest=False).indices
            grad_mask_vec[idx] = 1.0
        return grad_mask_vec

    def step(self, optimizer, **kwargs):
        # project gradient with gradient mask
        self.apply_grad_mask(self.model.parameters(), self.grad_mask_vec)
        super().step(optimizer)
        # gradient norm clipping
        model_params_vec = model2vec(self.model, return_torch=True)
        weight_diff = model_params_vec - self.global_weights_tensor
        diff_norm = torch.linalg.vector_norm(weight_diff)
        if float(diff_norm.item()) > 0.0:
            scale = min(1.0, float(self.norm_threshold) /
                        float(diff_norm.item()))
            weight_diff.mul_(scale)
        vec2model(self.global_weights_tensor + weight_diff,
                  self.model)

    def apply_grad_mask(self, parameters, grad_mask_vec):
        # .grad
        current_pos = 0
        for param in parameters:
            numel = param.numel()  # get the number of element of param
            if param.grad is None:
                current_pos += numel
                continue
            param.grad.mul_(
                grad_mask_vec[current_pos:current_pos + numel].reshape(param.shape)
            )
            current_pos += numel
