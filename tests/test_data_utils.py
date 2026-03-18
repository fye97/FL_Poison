import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
