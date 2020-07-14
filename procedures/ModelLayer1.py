import torch
import torch.nn as nn
import numpy as np


class Mammo_FT(nn.Module):
    def __init__(self):
        super(Mammo_FT, self).__init__()
        self.layer_d1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU()
        )
        self.layer_d2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer_d3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_d4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_d5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.layer_u5 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_u4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer_u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.layer_u1 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.layer_d1(x)
        latent = self.layer_d2(latent)
        latent = self.layer_d3(latent)
        latent = self.layer_d4(latent)
        latent = self.layer_d5(latent)
        C1 = latent[:, 0:64, :, :]
        C0 = latent[:, 64:128, :, :]
        S1 = C1.sum(dim=1).sum(dim=1).sum(dim=1)
        S0 = C0.sum(dim=1).sum(dim=1).sum(dim=1)
        a1 = torch.div(S1, S1 + S0 + 0.00000001)
        a1_arr = a1.detach().cpu().numpy()
        for idx, a1_score in enumerate(a1_arr, start=0):
            if a1_score > 0.5:
                latent[idx, 0:64, :, :] = torch.zeros((1, 64, 16, 16))
            else:
                latent[idx, 64:128, :, :] = torch.zeros((1, 64, 16, 16))
        reconstruct = self.layer_u5(latent)
        reconstruct = self.layer_u4(reconstruct)
        reconstruct = self.layer_u3(reconstruct)
        reconstruct = self.layer_u2(reconstruct)
        reconstruct = self.layer_u1(reconstruct)
        return a1, reconstruct

