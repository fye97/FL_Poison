import torch.nn as nn
from torchvision import models


def _build_small_head_vgg(model_fn, num_classes):
    """
    Build a VGG model with a lightweight classifier head for small images.
    """
    model = model_fn(num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def _make_small_head_builder(model_fn):
    def _builder(num_classes):
        return _build_small_head_vgg(model_fn, num_classes)
    return _builder


def add_vgg_from_torchvision():
    """
    Add all vgg models from torchvision to model_registry
    """
    reg = {}
    # register vgg models to model_registry
    vgg_models = [i for i in models.vgg.__dict__.keys() if i.startswith('vgg')]
    for model_name in vgg_models:
        # Use BN backbone for plain names (vgg11/vgg13/vgg16/vgg19) to improve
        # optimization stability in FL while keeping original model names.
        preferred_name = f"{model_name}_bn" if not model_name.endswith("_bn") else model_name
        model_fn = getattr(models, preferred_name, getattr(models, model_name))
        reg[model_name.lower()] = _make_small_head_builder(model_fn)
    return reg
