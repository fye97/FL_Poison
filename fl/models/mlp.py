import torch.nn as nn

from fl.models import model_registry


@model_registry
class mlp(nn.Module):
    """
    A simple MLP for vector/tabular inputs (e.g., HAR with 561 features).
    """

    def __init__(self, input_dim: int = 561, num_classes: int = 6, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, xb):
        xb = xb.reshape(xb.size(0), -1)
        return self.net(xb)

