import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5,
                 hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   # logits → we use BCEWithLogitsLoss