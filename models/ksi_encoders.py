from pathlib import Path
import torch
from torch import nn
import numpy as np

# Compress or Expand Descriptors
class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim=256, ENCDIM=128):
        super(LinearAutoencoder, self).__init__()
        bottleneck_dim=256-ENCDIM
        self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck_dim))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class LinearEncoder(nn.Module):
    def __init__(self, input_dim=256, ENCDIM=128):
        super(LinearEncoder, self).__init__()
        bottleneck_dim=256-ENCDIM
        self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
 
 # Semantic Encoders
class EncoderDecoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the Encoder-Only Network
class Encoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, encoded_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
