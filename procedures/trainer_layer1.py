import torch
import torch.nn as nn
import DataLoader
import numpy as np

class Mammo_FT(nn.Module):
    def __init__(self):
        super(Mammo_FT, self).__init__()
        self.layer_d1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer_d2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3 ,stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer_d3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_d4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_d5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer_u5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_u4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer_u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.layer_u1 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
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
        a1 = torch.div(S1, S1 + S0)
        a1_arr = a1.detach().numpy()
        for idx, a1_score in enumerate(a1_arr, start=0) :
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

def train(model, data_path, ephocs=1, batch_size=1, learning_rate=0.002):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = DataLoader.data_loader(data_path)
    data_loader.load()
    training_set_size = min(len(data_loader.fake), len(data_loader.real))
    n_batch = (training_set_size//batch_size)*2
    for ephoc in range(ephocs):
        real_ephoc, fake_ephoc = data_loader.load_ephoc()
        real_ephoc = np.expand_dims(real_ephoc, axis=1)
        fake_ephoc = np.expand_dims(fake_ephoc, axis=1)
        for batch in range(n_batch):
            if batch % 2 == 0:
                img_batch = real_ephoc[(batch//2)*batch_size:((batch//2)+1)*batch_size,:,:,:]
                label_batch = torch.from_numpy(np.zeros(batch_size).astype(np.float32)).to(device)
            else:
                img_batch = fake_ephoc[(batch // 2) * batch_size:((batch // 2) + 1) * batch_size, :, :, :]
                label_batch = torch.from_numpy(np.ones(batch_size).astype(np.float32)).to(device)
            img_batch = torch.from_numpy(img_batch).to(device)
            # forward pass
            out_labels, out_images = model(img_batch)
            loss = 0.1*nn.functional.l1_loss(out_images, img_batch) +\
                   nn.functional.binary_cross_entropy(out_labels, label_batch)
            # backward
            optimaizer.zero_grad()
            loss.backward()
            optimaizer.step()
            print("Ephoc: %d/%d, batch: %d/%d, loss = %1.4f, true_label = %d, mean_label = %1.4f" %
                  (ephoc+1,ephocs,batch+1,n_batch,loss.item(),batch % 2,np.mean(out_labels.detach().numpy())))

data_path = "C:\\niv\\Mammo_GAN\\training set"
model = Mammo_FT()
train(model, data_path, ephocs=5, batch_size=2)
