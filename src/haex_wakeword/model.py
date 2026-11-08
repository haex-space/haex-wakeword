"""The classifier on top of the frozen embeddings: 16 x 96 in, one logit out (openWakeWord's
small fully connected design)."""

import torch
from torch import nn

from . import config


class Classifier(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.WINDOW_FRAMES * config.EMBEDDING_SIZE, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Logits [batch] for windows [batch, 16, 96]."""
        return self.net(windows).squeeze(-1)


class Exported(nn.Module):
    """What haex runs: probabilities instead of logits."""

    def __init__(self, classifier: Classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(embeddings)).unsqueeze(-1)
