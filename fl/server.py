import numpy as np
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
        self.global_weights_vec = model2vec(
            self.global_model)

        self.aggregated_update = np.zeros_like(
            self.global_weights_vec, dtype=np.float32)

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
        updates = [client.update for client in self.clients]
        if not updates:
            self.client_updates = np.empty((0, 0), dtype=np.float32)
            return
        first = np.asarray(updates[0]).reshape(-1)
        num_clients = len(updates)
        num_params = first.size
        stacked = np.empty((num_clients, num_params), dtype=first.dtype)
        stacked[0] = first
        for idx, update in enumerate(updates[1:], start=1):
            arr = np.asarray(update).reshape(-1)
            if arr.size != num_params:
                raise ValueError(
                    f"Inconsistent update size: client {idx} has {arr.size}, expected {num_params}")
            if arr.dtype != first.dtype:
                arr = arr.astype(first.dtype, copy=False)
            stacked[idx] = arr
        self.client_updates = stacked

    def aggregation(self):
        # aggregate gradient (for fedsgd), model parameters (for fedavg), and return the aggregated model parameters
        self.aggregated_update = self.aggregator.aggregate(
            self.client_updates, last_global_model=self.global_model, global_weights_vec=self.global_weights_vec, global_epoch=self.global_epoch)

    def update_global(self):
        # update the global model with the aggregated update w.r.t the algorithm
        self.global_weights_vec = self.algorithm.update(
            self.aggregated_update, global_weights_vec=self.global_weights_vec)
