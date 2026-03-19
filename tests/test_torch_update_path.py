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
from flpoison.aggregators.bulyan import Bulyan
from flpoison.aggregators.krum import Krum, krum
from flpoison.aggregators.median import Median
from flpoison.aggregators.multikrum import MultiKrum, multi_krum
from flpoison.aggregators.trimmedmean import TrimmedMean, trimmed_mean
from flpoison.attackers.fangattack import craft_fang_attack
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


def _fang_attack_reference_numpy(attacker_updates, perturbation_base, stop_threshold):
    updates = np.array([np.asarray(update).reshape(-1) for update in attacker_updates], dtype=np.float32)
    base = (
        np.zeros_like(updates[0])
        if np.isscalar(perturbation_base)
        else np.asarray(perturbation_base, dtype=np.float32).reshape(-1)
    )
    est_direction = np.sign(np.mean(updates, axis=0))

    simulation_attack_number = 1
    while simulation_attack_number < len(updates):
        lambda_value = 1.0
        simulation_updates = np.empty(
            (len(updates) + simulation_attack_number, updates.shape[1]), dtype=np.float32
        )
        simulation_updates[: len(updates)] = updates
        while True:
            simulation_updates[len(updates): len(updates) + simulation_attack_number] = (
                base - lambda_value * est_direction
            )
            min_idx = krum(
                simulation_updates, simulation_attack_number, return_index=True
            )
            if min_idx >= len(updates) or lambda_value <= stop_threshold:
                break
            lambda_value *= 0.5
        simulation_attack_number += 1
        if min_idx >= len(updates):
            break

    return base - lambda_value * est_direction


def _bulyan_reference_numpy(updates, num_adv, num_clients):
    set_size = num_clients - 2 * num_adv
    active_idx = np.arange(len(updates), dtype=np.int64)
    selected_idx = []

    while len(selected_idx) < set_size and active_idx.size > 0:
        subset = updates[active_idx]
        local_idx = krum(subset, num_adv, return_index=True)
        selected_idx.append(int(active_idx[local_idx]))
        active_idx = np.delete(active_idx, local_idx)

    selected_idx = np.array(selected_idx, dtype=np.int64)
    beta = num_clients - 2 * num_adv
    selected_updates = updates[selected_idx]
    if beta == num_clients or beta == len(selected_idx):
        benign_updates = selected_updates
    else:
        median = np.median(selected_updates, axis=0)
        abs_dist = np.abs(selected_updates - median)
        beta_idx = np.argpartition(abs_dist, beta, axis=0)[:beta]
        benign_updates = np.take_along_axis(selected_updates, beta_idx, axis=0)
    return np.mean(benign_updates, axis=0)


def test_krum_aggregate_supports_torch_inputs():
    numpy_updates = np.array(
        [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [9.0, 9.0]], dtype=np.float32
    )
    torch_updates = torch.tensor(numpy_updates, dtype=torch.float32)
    expected_idx = krum(numpy_updates, num_byzantine=1, return_index=True)
    aggregator = Krum(SimpleNamespace(num_adv=1, defense_params=None))

    aggregated = aggregator.aggregate(torch_updates)

    assert torch.is_tensor(aggregated)
    assert krum(torch_updates, num_byzantine=1, return_index=True) == expected_idx
    assert torch.allclose(aggregated, torch_updates[expected_idx])


def test_multikrum_aggregate_supports_torch_inputs():
    numpy_updates = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [5.0, 5.0], [6.0, 6.0]], dtype=np.float32
    )
    torch_updates = torch.tensor(numpy_updates, dtype=torch.float32)
    expected = multi_krum(numpy_updates, num_byzantine=1, avg_percentage=0.4)
    aggregator = MultiKrum(
        SimpleNamespace(num_adv=1, defense_params={"avg_percentage": 0.4})
    )

    aggregated = aggregator.aggregate(torch_updates)

    assert torch.is_tensor(aggregated)
    assert np.allclose(aggregated.cpu().numpy(), expected)


def test_bulyan_aggregate_supports_torch_inputs_and_keeps_original_indices():
    numpy_updates = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [4.8, 5.0],
            [5.0, 5.2],
            [20.0, 20.0],
        ],
        dtype=np.float32,
    )
    torch_updates = torch.tensor(numpy_updates, dtype=torch.float32)
    expected = _bulyan_reference_numpy(numpy_updates, num_adv=1, num_clients=6)
    aggregator = Bulyan(SimpleNamespace(num_adv=1, num_clients=6, defense_params=None))

    aggregated_numpy = aggregator.aggregate(numpy_updates)
    aggregated_torch = aggregator.aggregate(torch_updates)

    assert np.allclose(aggregated_numpy, expected)
    assert torch.is_tensor(aggregated_torch)
    assert np.allclose(aggregated_torch.cpu().numpy(), expected)


def test_craft_fang_attack_matches_reference_for_numpy_and_torch():
    attacker_updates = np.array(
        [
            [0.2, -0.5, 1.0, 0.3],
            [0.4, -0.1, 0.8, -0.6],
            [0.1, -0.3, 1.2, 0.5],
        ],
        dtype=np.float32,
    )
    perturbation_base = np.array([0.7, -0.2, 0.4, 0.1], dtype=np.float32)
    expected = _fang_attack_reference_numpy(
        attacker_updates, perturbation_base, stop_threshold=1.0e-5
    )

    crafted_numpy = craft_fang_attack(
        attacker_updates, perturbation_base, stop_threshold=1.0e-5
    )
    crafted_torch = craft_fang_attack(
        torch.tensor(attacker_updates, dtype=torch.float32),
        torch.tensor(perturbation_base, dtype=torch.float32),
        stop_threshold=1.0e-5,
    )

    assert np.allclose(crafted_numpy, expected)
    assert torch.is_tensor(crafted_torch)
    assert np.allclose(crafted_torch.cpu().numpy(), expected)
