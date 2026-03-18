import sys
from pathlib import Path
from types import SimpleNamespace
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "hdbscan" not in sys.modules:
    hdbscan_stub = types.ModuleType("hdbscan")

    class _DummyHDBSCAN:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit_predict(self, *args, **kwargs):
            raise RuntimeError("Dummy hdbscan stub should not be used in this test")

    hdbscan_stub.HDBSCAN = _DummyHDBSCAN
    sys.modules["hdbscan"] = hdbscan_stub

from flpoison.aggregators.mean import Mean
from flpoison.aggregators.median import Median
from flpoison.aggregators.trimmedmean import TrimmedMean, trimmed_mean
from flpoison.fl.algorithms.fedavg import FedAvg
from flpoison.fl.algorithms.fedsgd import FedSGD
from flpoison.fl.models.model_utils import model2vec
from flpoison.fl.server import Server


def test_mean_aggregate_supports_torch_inputs():
    aggregator = Mean(SimpleNamespace())
    updates = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.float32)

    aggregated = aggregator.aggregate(updates)

    assert torch.is_tensor(aggregated)
    assert torch.allclose(aggregated, torch.tensor([2.0, 4.0], dtype=torch.float32))


def test_median_aggregate_supports_torch_inputs():
    aggregator = Median(SimpleNamespace())
    updates = torch.tensor([[1.0, 9.0], [3.0, 5.0], [2.0, 7.0]], dtype=torch.float32)

    aggregated = aggregator.aggregate(updates)

    assert torch.is_tensor(aggregated)
    assert torch.allclose(aggregated, torch.tensor([2.0, 7.0], dtype=torch.float32))


def test_trimmed_mean_aggregate_supports_torch_inputs():
    aggregator = TrimmedMean(SimpleNamespace(defense_params={"beta": 0.25}))
    updates = torch.tensor(
        [[0.0, 100.0], [1.0, 7.0], [2.0, 8.0], [100.0, 9.0]], dtype=torch.float32
    )

    aggregated = aggregator.aggregate(updates)

    assert torch.is_tensor(aggregated)
    assert torch.allclose(aggregated, torch.tensor([1.5, 8.5], dtype=torch.float32))


def test_trimmed_mean_beta_zero_matches_plain_mean():
    updates = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)

    aggregated = trimmed_mean(updates, filter_frac=0.0)

    assert np.allclose(aggregated, np.array([3.0, 5.0], dtype=np.float32))


def test_fedsgd_get_local_update_preserves_torch_backend():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 4.0]], dtype=torch.float32))

    global_weights_vec = model2vec(model, return_torch=True) + 1.0
    algorithm = FedSGD(SimpleNamespace(), model)

    update = algorithm.get_local_update(global_weights_vec=global_weights_vec)

    assert torch.is_tensor(update)
    assert torch.allclose(update, torch.full_like(global_weights_vec, -1.0))


def test_fedavg_get_local_update_can_stay_on_torch():
    model = torch.nn.Linear(2, 1, bias=False)
    algorithm = FedAvg(SimpleNamespace(local_epochs=1), model)

    update = algorithm.get_local_update(global_weights_vec=torch.zeros(2))

    assert torch.is_tensor(update)


def test_server_collect_updates_stacks_torch_updates_and_converts_numpy():
    server = Server.__new__(Server)
    server.runtime_profiler = None
    server.use_torch_updates = True
    server.global_weights_vec = torch.zeros(3, dtype=torch.float32)
    server.clients = [
        SimpleNamespace(update=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)),
        SimpleNamespace(update=np.array([4.0, 5.0, 6.0], dtype=np.float32)),
    ]

    Server.collect_updates(server, global_epoch=0)

    assert torch.is_tensor(server.client_updates)
    assert torch.allclose(
        server.client_updates,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
    )


def test_server_collect_updates_keeps_mean_updates_unstacked_to_avoid_extra_gpu_peak():
    server = Server.__new__(Server)
    server.runtime_profiler = None
    server.use_torch_updates = True
    server.aggregator = Mean(SimpleNamespace())
    server.global_weights_vec = torch.zeros(3, dtype=torch.float32)
    server.clients = [
        SimpleNamespace(update=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)),
        SimpleNamespace(update=np.array([4.0, 5.0, 6.0], dtype=np.float32)),
    ]

    Server.collect_updates(server, global_epoch=0)

    assert isinstance(server.client_updates, tuple)
    assert len(server.client_updates) == 2
    assert all(torch.is_tensor(update) for update in server.client_updates)
    aggregated = server.aggregator.aggregate(server.client_updates)
    assert torch.allclose(aggregated, torch.tensor([2.5, 3.5, 4.5], dtype=torch.float32))
