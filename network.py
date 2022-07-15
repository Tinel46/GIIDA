from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
import copy

def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x + offset, max=1.))


def set_requires_grad(net, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class ImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True):
        super(ImageClassifier, self).__init__()
        self.backbone = nn.Sequential(backbone, nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten())
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
            )
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor, input_feature=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        if not input_feature:
            f = self.backbone(x)
            f1 = self.bottleneck(f)
            predictions = self.head(f1)
            return predictions, f1
        else:
            predictions = self.head(x)
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

class Classifier(nn.Module):
    def __init__(self, num_classes: int, bottleneck_dim: Optional[int] = 256):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self._features_dim = bottleneck_dim
        self.head = nn.Linear(self._features_dim, num_classes)

    def forward(self, x):

        predictions = self.head(x)
        return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [{"params": self.head.parameters(), "lr": 1.0 * base_lr}]
        return params
