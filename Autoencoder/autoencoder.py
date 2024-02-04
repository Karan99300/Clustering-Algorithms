import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, inputs_dim, n_bottleneck):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_bottleneck)  # Bottleneck layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_bottleneck, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, inputs_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
