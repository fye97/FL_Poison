import torch.nn as nn

from fl.models import model_registry


@model_registry
class fcn(nn.Module):
    """
    Fully-connected network for vector/tabular inputs (e.g., HAR with 561 features).
    """

    def __init__(
        self,
        input_dim: int = 561,
        num_classes: int = 6,
        hidden_dims=(512, 256),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        dims = [self.input_dim, *list(hidden_dims), self.num_classes]
        layers = []
        for i in range(len(dims) - 2):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, xb):
        xb = xb.reshape(xb.size(0), -1)
        return self.net(xb)

