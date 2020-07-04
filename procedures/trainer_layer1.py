import torch
import torch.nn as nn
import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Mammo_FT(nn.Module):
    def __init__(self):
        super(Mammo_FT, self).__init__()
        self.layer_d1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.ReLU()
        )
        self.layer_d2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_d3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_d4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.layer_u4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            #torch.nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.layer_u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.layer_d1(x)
        latent = self.layer_d2(latent)
        latent = self.layer_d3(latent)
        latent = self.layer_d4(latent)
        C1 = latent[:, 0:64, :, :]
        C0 = latent[:, 64:128, :, :]
        S1 = C1.sum(dim=1).sum(dim=1).sum(dim=1)
        S0 = C0.sum(dim=1).sum(dim=1).sum(dim=1)
        a1 = torch.div(S1, S1 + S0)
        a1_arr = a1.detach().cpu().numpy()
        for idx, a1_score in enumerate(a1_arr, start=0):
            if a1_score > 0.5:
                latent[idx, 0:64, :, :] = torch.zeros((1, 64, 16, 16))
            else:
                latent[idx, 64:128, :, :] = torch.zeros((1, 64, 16, 16))
        reconstruct = self.layer_u4(latent)
        reconstruct = self.layer_u3(reconstruct)
        reconstruct = self.layer_u2(reconstruct)
        reconstruct = self.layer_u1(reconstruct)
        return a1, reconstruct

def train(model, data_path, ephocs=1, batch_size=1, learning_rate=0.0002):
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    min_val_ephoc = 0
    train_cross_entropy_loss = []
    val_cross_entropy_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)
    model.to(device)
    optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = DataLoader.data_loader(data_path)
    data_loader.load_train(augmentation=0)
    data_loader.load_val(augmentation=1)
    training_set_size = min(len(data_loader.fake), len(data_loader.real))
    #n_batch = (training_set_size//batch_size)*2
    n_batch = (data_loader.val_size // batch_size) * 2
    for ephoc in range(ephocs):
        data_loader.load_ephoc()
        for batch in range(n_batch):
            if batch % 2 == 0:
                #img_batch = data_loader.real[(batch//2)*batch_size:((batch//2)+1)*batch_size, :, :, :]
                img_batch = data_loader.real_eval[(batch // 2) * batch_size:((batch // 2) + 1) * batch_size, :, :, :]
                label_batch = torch.from_numpy(np.zeros(batch_size).astype(np.float32)).to(device)
            else:
                #img_batch = data_loader.fake[(batch // 2) * batch_size:((batch // 2) + 1) * batch_size, :, :, :]
                img_batch = data_loader.fake_eval[(batch // 2) * batch_size:((batch // 2) + 1) * batch_size, :, :, :]
                label_batch = torch.from_numpy(np.ones(batch_size).astype(np.float32)).to(device)
            img_batch = torch.from_numpy(img_batch).to(device)
            # forward pass
            out_labels, out_images = model(img_batch)
            loss = 1 * nn.functional.l1_loss(out_images, img_batch) +\
                   12 * nn.functional.binary_cross_entropy(out_labels, label_batch)
            # backward
            optimaizer.zero_grad()
            loss.backward()
            optimaizer.step()
            print("Ephoc: %d/%d, batch: %d/%d, loss = %1.4f, true_label = %d, mean_label = %1.4f" %
                  (ephoc+1,ephocs,batch+1,n_batch,loss.item(),batch % 2, np.mean(out_labels.detach().cpu().numpy())))
        if ephoc % 1 == 0:
            val_set = np.concatenate((data_loader.val_real, data_loader.val_fake))
            val_set = torch.from_numpy(val_set).to(device)

            true_label = np.concatenate((np.zeros(data_loader.val_size), np.ones(data_loader.val_size)))
            true_label = true_label.astype(np.float32)
            true_label = torch.from_numpy(true_label).to(device)
            tr_set = np.concatenate((data_loader.real_eval,
                                     data_loader.fake_eval))
            tr_set = torch.from_numpy(tr_set).to(device)
            tr_out_labels, tr_out_images = model(tr_set)

            train_loss.append((1 * nn.functional.l1_loss(tr_out_images, tr_set) +\
                             12 * nn.functional.binary_cross_entropy(tr_out_labels, true_label)).detach().item())
            train_cross_entropy_loss\
                .append(nn.functional.binary_cross_entropy(tr_out_labels, true_label).detach().item())
            val_out_labels, val_out_images = model(val_set)
            V_Loss = (1 * nn.functional.l1_loss(val_out_images, val_set) +\
                        12 * nn.functional.binary_cross_entropy(val_out_labels, true_label)).detach().item()
            val_loss.append(V_Loss)
            val_cross_entropy_loss \
                .append(nn.functional.binary_cross_entropy(val_out_labels, true_label).detach().item())
#           if V_Loss < min_val_loss:
            if (True):
                min_val_loss = V_Loss
                min_val_ephoc = ephoc+1
                torch.save(model.state_dict(), 'mammo-FT4.ckpt')
                print("saving model...")
            if ephoc - min_val_ephoc > 3000:
                break
        print(min_val_ephoc)
    x_axis = np.arange(1, len(train_loss)+1, 1)
    plt.plot(x_axis, np.asarray(train_loss), x_axis, np.asarray(val_loss))
    plt.title("training and validation loss")
    plt.xlabel('ephoc')
    plt.ylabel('loss')
    plt.legend(('training loss', 'validation loss'))
    plt.show()

    plt.plot(x_axis, np.asarray(train_cross_entropy_loss), x_axis, np.asarray(val_cross_entropy_loss))
    plt.title("training and validation cross entropy loss")
    plt.xlabel('ephoc')
    plt.ylabel('loss')
    plt.legend(('training loss', 'validation loss'))
    plt.show()

data_path = "D:\\Breast Cancer\\Databases\\Forensic-Transfer"
model = Mammo_FT()
train(model, data_path, ephocs=800, batch_size=50, learning_rate=0.00002)

