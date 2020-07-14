import torch
import torch.nn as nn
import DataLoaderLayer1
import ModelLayer1
import numpy as np
import matplotlib.pyplot as plt

def train(model, data_path, epochs=1, batch_size=1, learning_rate=0.00002):
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    min_val_epoch = 0
    train_cross_entropy_loss = []
    val_cross_entropy_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)
    model.to(device)
    optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = DataLoaderLayer1.data_loader(data_path)
    data_loader.load_train(augmentation=1)
    data_loader.load_val(augmentation=1)
    training_set_size = min(len(data_loader.fake), len(data_loader.real))*2
    n_batch = training_set_size//batch_size

    val_set = np.concatenate((data_loader.val_real, data_loader.val_fake))
    val_set = torch.from_numpy(val_set).to(device)

    true_label = np.concatenate((np.zeros(data_loader.val_size), np.ones(data_loader.val_size)))
    true_label = true_label.astype(np.float32)
    true_label = torch.from_numpy(true_label).to(device)
    tr_set = np.concatenate((data_loader.real_eval,
                             data_loader.fake_eval))
    tr_set = torch.from_numpy(tr_set).to(device)
    for epoch in range(epochs):
        data_loader.load_epoch()
        for batch in range(n_batch):
            img_batch = np.concatenate((data_loader.real[batch*batch_size//2:(batch+1)*batch_size//2, :, :, :],
                                       data_loader.fake[batch*batch_size//2:(batch+1)*batch_size//2, :, :, :]))
            img_batch = torch.from_numpy(img_batch).to(device)
            label_batch = np.concatenate((np.zeros(batch_size//2).astype(np.float32),
                                         np.ones(batch_size//2).astype(np.float32)))
            label_batch = torch.from_numpy(label_batch).to(device)
            # forward pass
            out_labels, out_images = model(img_batch)
            loss = 8 * nn.functional.l1_loss(out_images, img_batch) +\
                   10 * nn.functional.mse_loss(out_labels, label_batch)
            # backward
            optimaizer.zero_grad()
            loss.backward()
            optimaizer.step()
            print("Epoch: %d/%d, batch: %d/%d, loss = %1.4f" %
                  (epoch+1,epochs,batch+1,n_batch,loss.item()))
        with torch.no_grad():
            tr_out_labels, tr_out_images = model(tr_set)
            train_loss.append((8 * nn.functional.l1_loss(tr_out_images, tr_set) +\
                             10 * nn.functional.mse_loss(tr_out_labels, true_label)).detach().item())
            train_cross_entropy_loss\
                .append(nn.functional.binary_cross_entropy(tr_out_labels, true_label).detach().item())

            val_out_labels, val_out_images = model(val_set)
            V_Loss = (8 * nn.functional.l1_loss(val_out_images, val_set) +\
                        10 * nn.functional.mse_loss(val_out_labels, true_label)).detach().item()
            val_loss.append(V_Loss)
            val_cross_entropy_loss \
                .append(nn.functional.binary_cross_entropy(val_out_labels, true_label).detach().item())
            if V_Loss < min_val_loss:
                min_val_loss = V_Loss
                min_val_epoch = epoch+1
                print("saving model...")
                torch.save(model.state_dict(), 'mammo-FT8.ckpt')
            if epoch - min_val_epoch > 25:
                break
        print(min_val_epoch)
    x_axis = np.arange(1, len(train_loss)+1, 1)
    plt.plot(x_axis, np.asarray(train_loss), x_axis, np.asarray(val_loss))
    plt.title("training and validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(('training loss', 'validation loss'))
    plt.show()

    plt.plot(x_axis, np.asarray(train_cross_entropy_loss), x_axis, np.asarray(val_cross_entropy_loss))
    plt.title("training and validation cross entropy loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(('training loss', 'validation loss'))
    plt.show()


data_path = "D:\\Breast Cancer\\Databases\\Forensic-Transfer\\layer1"
model = ModelLayer1.Mammo_FT()
#model.load_state_dict(torch.load("mammo-FT7-3.ckpt"))
train(model, data_path, epochs=700, batch_size=64, learning_rate=0.0001)