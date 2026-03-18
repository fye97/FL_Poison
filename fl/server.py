import numpy as np
import time
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

        # initialize the aggregator for the server
        self.aggregator = get_aggregator(
            self.args.defense)(self.args, train_dataset=self.train_dataset)
        self.use_torch_updates = bool(
            getattr(self.aggregator, "supports_torch_updates", False)
            and getattr(self.args, "attack", "NoAttack") == "NoAttack"
        )

        # Keep Mean-like aggregation on torch to avoid repeated GPU<->CPU vector copies.
        self.global_model = get_model(args)
        self.global_weights_vec = model2vec(
            self.global_model, return_torch=self.use_torch_updates)

        if self.use_torch_updates:
            self.aggregated_update = torch.zeros_like(self.global_weights_vec)
        else:
            self.aggregated_update = np.zeros_like(
                self.global_weights_vec, dtype=np.float32)

        if self.args.record_time:
            self.time_recorder = TimingRecorder(self.worker_id,
                                                self.args.output)
            self.collect_updates = self.time_recorder.timing_decorator(
                self.collect_updates)
            self.aggregation = self.time_recorder.timing_decorator(
                self.aggregation)
            self.update_global = self.time_recorder.timing_decorator(
                self.update_global)

    def set_algorithm(self, algorithm):
        self.algorithm = get_algorithm_handler(
            algorithm)(self.args, self.global_model)

    def _coerce_update_array(self, update):
        if torch.is_tensor(update):
            return update.detach().reshape(-1).cpu().numpy()
        return np.asarray(update).reshape(-1)

    def _coerce_update_tensor(self, update):
        ref = self.global_weights_vec
        if not torch.is_tensor(ref):
            raise TypeError("Torch update path requires tensor global_weights_vec")
        if torch.is_tensor(update):
            return update.detach().reshape(-1).to(device=ref.device, dtype=ref.dtype)
        return torch.as_tensor(update, device=ref.device, dtype=ref.dtype).reshape(-1)

    def collect_updates(self, global_epoch):
        start_time = time.perf_counter()
        self.global_epoch = global_epoch
        # get the client update from clients
        updates = [client.update for client in self.clients]
        if not updates:
            if getattr(self, "use_torch_updates", False):
                self.client_updates = torch.empty(
                    (0, 0),
                    device=self.global_weights_vec.device,
                    dtype=self.global_weights_vec.dtype,
                )
            else:
                self.client_updates = np.empty((0, 0), dtype=np.float32)
            if self.runtime_profiler is not None:
                self.runtime_profiler.add_server_stage(
                    "collect_updates", time.perf_counter() - start_time)
            return

        if getattr(self, "use_torch_updates", False):
            first = self._coerce_update_tensor(updates[0])
            num_params = first.numel()
            tensors = [first]
            for idx, update in enumerate(updates[1:], start=1):
                arr = self._coerce_update_tensor(update)
                if arr.numel() != num_params:
                    raise ValueError(
                        f"Inconsistent update size: client {idx} has {arr.numel()}, expected {num_params}")
                tensors.append(arr)
            if getattr(getattr(self, "aggregator", None), "accepts_unstacked_torch_updates", False):
                stacked = tuple(tensors)
            else:
                stacked = torch.stack(tensors, dim=0)
        else:
            first = self._coerce_update_array(updates[0])
            num_clients = len(updates)
            num_params = first.size
            stacked = np.empty((num_clients, num_params), dtype=first.dtype)
            stacked[0] = first
            for idx, update in enumerate(updates[1:], start=1):
                arr = self._coerce_update_array(update)
                if arr.size != num_params:
                    raise ValueError(
                        f"Inconsistent update size: client {idx} has {arr.size}, expected {num_params}")
                if arr.dtype != first.dtype:
                    arr = arr.astype(first.dtype, copy=False)
                stacked[idx] = arr
        self.client_updates = stacked
        if self.runtime_profiler is not None:
            self.runtime_profiler.add_server_stage(
                "collect_updates", time.perf_counter() - start_time)

    def aggregation(self):
        # aggregate gradient (for fedsgd), model parameters (for fedavg), and return the aggregated model parameters
        start_time = time.perf_counter()
        if self.runtime_profiler is not None:
            self.runtime_profiler.begin_aggregation_breakdown()
        self.aggregated_update = self.aggregator.aggregate(
            self.client_updates, last_global_model=self.global_model, global_weights_vec=self.global_weights_vec, global_epoch=self.global_epoch)
        if self.runtime_profiler is not None:
            self.runtime_profiler.finish_aggregation(
                time.perf_counter() - start_time)

    def update_global(self):
        # update the global model with the aggregated update w.r.t the algorithm
        self.global_weights_vec = self.algorithm.update(
            self.aggregated_update, global_weights_vec=self.global_weights_vec)
