import warnings
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flpoison.datapreprocessor.data_utils as data_utils
from flpoison.datapreprocessor.data_utils import Partition


class _DummyDataset:
    def __init__(self):
        self.data = torch.arange(2 * 28 * 28, dtype=torch.uint8).reshape(2, 28, 28)
        self.targets = torch.tensor([1, 2], dtype=torch.int64)
        self.classes = list(range(10))


class _ZeroOutSynthesizer:
    def backdoor_batch(self, image, labels, **kwargs):
        image.zero_()
        return image, labels


class _FakeCIFARDataset:
    def __init__(self, root, train, download, transform):
        warnings.warn(
            "dtype(): align should be passed as Python or NumPy boolean but got `align=0`. "
            "Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)",
            getattr(getattr(np, "exceptions", np), "VisibleDeprecationWarning", Warning),
            stacklevel=2,
        )
        self.data = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        self.targets = [0, 1]
        self.transform = transform


class _FakeCIFARDatasetWithOtherWarning:
    def __init__(self, root, train, download, transform):
        warnings.warn("different warning", UserWarning, stacklevel=2)
        self.data = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        self.targets = [0]
        self.transform = transform


def test_partition_caches_deterministic_tensor_transforms():
    dataset = _DummyDataset()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.25,)),
    ])
    partition = Partition(dataset, indices=[0, 1], transform=transform)

    image, target = partition[0]
    expected = transform(Image.fromarray(dataset.data[0].numpy()))

    assert partition.cached_data is not None
    assert image.shape == (1, 32, 32)
    assert torch.allclose(image, expected, atol=2e-2, rtol=0.0)
    assert int(target) == 1


def test_partition_poison_does_not_mutate_cached_tensor():
    dataset = _DummyDataset()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.25,)),
    ])
    partition = Partition(dataset, indices=[0], transform=transform)
    cached_before = partition.cached_data[0].clone()
    partition.poison_setup(_ZeroOutSynthesizer())

    poisoned_image, _ = partition[0]

    assert torch.count_nonzero(poisoned_image) == 0
    assert torch.allclose(partition.cached_data[0], cached_before)


def test_load_data_suppresses_torchvision_cifar_numpy_warning(monkeypatch):
    monkeypatch.setattr(data_utils.datasets, "CIFAR10", _FakeCIFARDataset)
    args = SimpleNamespace(
        dataset="CIFAR10",
        model="resnet18",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2430, 0.2610),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_dataset, test_dataset = data_utils.load_data(args)

    assert not any("align should be passed" in str(w.message) for w in caught)
    assert torch.equal(train_dataset.targets, torch.tensor([0, 1]))
    assert torch.equal(test_dataset.targets, torch.tensor([0, 1]))


def test_load_data_keeps_unrelated_warnings_visible(monkeypatch):
    monkeypatch.setattr(data_utils.datasets, "CIFAR10", _FakeCIFARDatasetWithOtherWarning)
    args = SimpleNamespace(
        dataset="CIFAR10",
        model="resnet18",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2430, 0.2610),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data_utils.load_data(args)

    assert any(str(w.message) == "different warning" for w in caught)
