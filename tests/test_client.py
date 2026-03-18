import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.fl.client import Client


class _DummyModel:
    def train(self):
        return None


class _DummyScheduler:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


def test_local_training_aggregates_metrics_by_sample_count():
    client = Client.__new__(Client)
    client.model = _DummyModel()
    client.train_loader = iter([None, None])
    client.optimizer = object()
    client.criterion_fn = object()
    client.local_epochs = 2
    client.lr_scheduler = _DummyScheduler()
    client.step = lambda optimizer, **kwargs: None

    batches = iter([
        (1, 0.8, 2),
        (3, 1.6, 4),
    ])
    client.train = lambda *args, **kwargs: next(batches)

    train_acc, train_loss, train_samples = Client.local_training(client)

    assert train_acc == pytest.approx(4 / 6)
    assert train_loss == pytest.approx(2.4 / 6)
    assert train_samples == 6
    assert client.last_local_training_stats == {
        "train_acc": pytest.approx(4 / 6),
        "train_loss": pytest.approx(2.4 / 6),
        "num_samples": 6,
    }
    assert client.lr_scheduler.calls == 1
