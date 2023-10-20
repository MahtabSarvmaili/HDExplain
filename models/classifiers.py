import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self, feature_dim, latent_dim, n_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
      logit = self.forward(x)
      return F.softmax(logit, dim=1)