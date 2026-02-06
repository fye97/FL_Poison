import numpy as np
import torch
from aggregators import get_aggregator
from fl.algorithms import get_algorithm_handler
from fl.models import get_model
from fl.models.model_utils import model2vec
from fl.worker import Worker
from global_utils import TimingRecorder


class Server(Worker):
    def __init__(self, args, clients, test_dataset, train_dataset):
        server_id = -1  # server worker_id = -1
        super().__init__(args, worker_id=server_id)
        self.clients = clients
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset  # it's only used in the defense FLTrust

        # initialize the global model and flattened-model
        self.global_model = get_model(args)
        self.global_weights_vec = model2vec(self.global_model)
        self.aggregated_update = None

        # initialize the aggregator for the server
        self.aggregator = get_aggregator(
            self.args.defense)(self.args, train_dataset=self.train_dataset)

        if self.args.record_time:
            self.time_recorder = TimingRecorder(self.worker_id,
                                                self.args.output)
            self.aggregation = self.time_recorder.timing_decorator(
                self.aggregation)

    def set_algorithm(self, algorithm):
        self.algorithm = get_algorithm_handler(
            algorithm)(self.args, self.global_model)

    def collect_updates(self, global_epoch):
        self.global_epoch = global_epoch
        # get the client update from clients
        if getattr(self.aggregator, "use_torch", False):
            device = self.global_weights_vec.device
            dtype = self.global_weights_vec.dtype
            self.client_updates = torch.stack(
                [self._to_tensor(client.update, device=device, dtype=dtype) for client in self.clients], dim=0)
        else:
            self.client_updates = np.stack(
                [self._to_numpy(client.update) for client in self.clients], axis=0)

    def aggregation(self):
        # aggregate gradient (for fedsgd), model parameters (for fedavg), and return the aggregated model parameters
        global_vec = self.global_weights_vec if getattr(self.aggregator, "use_torch", False) else self._to_numpy(self.global_weights_vec)
        self.aggregated_update = self.aggregator.aggregate(
            self.client_updates,
            last_global_model=self.global_model,
            global_weights_vec=global_vec,
            global_epoch=self.global_epoch,
        )

    def update_global(self):
        # update the global model with the aggregated update w.r.t the algorithm
        aggregated_update = self._to_tensor(
            self.aggregated_update,
            device=self.global_weights_vec.device,
            dtype=self.global_weights_vec.dtype,
        )
        self.global_weights_vec = self.algorithm.update(
            aggregated_update, global_weights_vec=self.global_weights_vec)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _to_tensor(value, device, dtype):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)
