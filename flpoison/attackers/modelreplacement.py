import torch
from flpoison.fl.client import Client
from flpoison.attackers.pbases.mpbase import MPBase
from flpoison.attackers.pbases.dpbase import DPBase
from flpoison.utils.global_utils import actor
from flpoison.fl.models.model_utils import model2vec
from flpoison.attackers import attacker_registry
from sklearn.metrics.pairwise import cosine_distances
from flpoison.attackers.synthesizers.pixel_synthesizer import PixelSynthesizer


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning', 'non_omniscient')
class ModelReplacement(MPBase, DPBase, Client):
    """
    [How to Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html) - AISTATS '20
    Model replacement attack, also known as constrain-and-scale attack and scaling attack, it first trains models with loss=normal_loss + anomaly_loss to avoid backdoor detection, then scales the update (X-G^t) by a factor gamma.
    """

    supports_torch_updates = True

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        scaling_factor: estimated scaling factor, num_clients / global_lr, 50/1=50 in our setting
        alpha: the weight of the classification loss in the total loss
        """
        self.default_attack_params = {
            'scaling_factor': 20, "alpha": 0.5, "attack_model": "all2one",
            "poisoning_ratio": 0.32, "target_label": 7, "source_label": 2, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5}
        self.update_and_set_attr()

        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)
        self.algorithm = "FedOpt"

    def define_synthesizer(self):
        # for pixel-type trigger, specify the trigger tensor
        self.trigger = torch.ones((1, 5, 5))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=self.poisoning_ratio, source_label=self.source_label, single_epoch=self.single_epoch)

    def criterion_fn(self, y_pred, y_true, **kwargs):
        """rewrite the criterion function by adding an anomaly detection term, cosine distance between the local weights and the global weights
        # a L_class + (1-a) L_ano
        """
        local_weights = model2vec(self.model, return_torch=True)
        cosine_dist = 1.0 - torch.nn.functional.cosine_similarity(
            local_weights.unsqueeze(0),
            self.global_weights_tensor.unsqueeze(0),
            dim=1,
            eps=1e-12,
        ).squeeze(0)
        return self.alpha * torch.nn.CrossEntropyLoss()(y_pred, y_true) + (1-self.alpha) * cosine_dist

    def non_omniscient(self):
        # scale
        # gamma = self.args.num_clients/self.optimizer.param_groups[0]['lr'] # however, adversaries don't know num_clients
        # self.update = X - G^t
        scaled_update = self.global_weights_vec + self.scaling_factor * \
            (self.update - self.global_weights_vec) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        return scaled_update
