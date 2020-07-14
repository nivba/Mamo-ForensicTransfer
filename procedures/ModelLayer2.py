import torch
import torch.nn as nn
import numpy as np

class ScoreClassifier(nn.Module):
    def __init__(self):
        super(ScoreClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=5, stride=2, padding=1),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(7),
            #nn.Dropout2d(0.25),
            #nn.BatchNorm2d(8),
            #nn.ReLU()
        )

        self.Linear1 = nn.Sequential(
            nn.Linear(4096, 64),
            nn.Dropout(0.5),
            #nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(64, 8),
            nn.Dropout(0.5),
            #nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.FC =nn.Sequential(
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(self,x):
        c = self.conv1(x)
        mp = self.pool1(c)
        l1 = self.Linear1(torch.flatten(mp,start_dim=1))
        l2 = self.Linear2(l1)
        score = self.FC(l2)
        label = torch.div(score+1, 2)
        return label

