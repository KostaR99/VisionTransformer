import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, features: int, hidden_features: int, p: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_features, features)

    def forward(self, x):
        x = self.act(self.fc1(x))  # n_samples, n_patches, hidden_features
        x = self.drop(x)
        x = self.fc2(x)  # n_samples, n_patches, features
        x = self.drop(x)

        return x


if __name__ == "__main__":
    x = torch.randn((1, 16, 768))
    mlp = MLP(features=768, hidden_features=200)
    assert mlp(x).shape == x.shape
