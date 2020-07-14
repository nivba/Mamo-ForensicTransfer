import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import DataLoaderLayer2
import ModelLayer2

def train(model, path, epochs, batch_size, learning_rate = 0.0001):
    dl = DataLoaderLayer2.DataLoader2(path)
    print("loading training set...")
    dl.load_train()
    print("done.")
    print("loading validation set...")
    dl.load_val()
    print("done.")
    train_loss = []
    val_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)
    model.to(device)
    optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # eval parameters
    min_val_loss = 1000
    min_val_epoch = 0
    eval_label = np.concatenate((np.ones((dl.val_size)).astype(np.float32),
                                 np.zeros((dl.val_size)).astype(np.float32)))
    eval_label = np.expand_dims(eval_label, axis=1)
    eval_label = torch.from_numpy(eval_label).to(device)
    eval = np.concatenate((dl.eval_fake, dl.eval_real))
    eval = np.expand_dims(eval, axis=1)
    eval = torch.from_numpy(eval).to(device)
    val = np.concatenate((dl.val_fake, dl.val_real))
    val =np.expand_dims(val, axis=1)
    val = torch.from_numpy(val).to(device)

    #
    training_set_size = min(len(dl.train_real), len(dl.train_fake)) * 2
    n_batch = training_set_size//batch_size
    for epoch in range(epochs):
        dl.shuffle_data()
        for batch in range (n_batch):
            im_batch = np.concatenate((dl.train_fake[batch*(batch_size//2):(batch+1)*(batch_size//2), :, :],
                                      dl.train_real[batch*(batch_size//2):(batch+1)*(batch_size//2), :, :]))
            label_batch = np.concatenate((np.ones((batch_size//2)).astype(np.float32),
                                         np.zeros((batch_size//2)).astype(np.float32)))
            label_batch = np.expand_dims(label_batch, axis=1)
            im_batch = np.expand_dims(im_batch, axis=1)
            im_batch = torch.from_numpy(im_batch).to(device)
            label_batch = torch.from_numpy(label_batch).to(device)
            out_labels = model(im_batch)
            loss = nn.functional.binary_cross_entropy(out_labels, label_batch)
            optimaizer.zero_grad()
            loss.backward()
            optimaizer.step()
            print("Epoch: %d/%d, batch: %d/%d, loss = %1.4f" %
                  (epoch + 1, epochs, batch + 1, n_batch, loss.item()))
        with torch.no_grad():
            out_eval_labels = model(eval)
            eval_loss = nn.functional.binary_cross_entropy(out_eval_labels, eval_label)
            train_loss.append(eval_loss.detach().item())
            out_val_labels = model(val)
            v_loss = nn.functional.binary_cross_entropy(out_val_labels, eval_label)
            val_loss.append(v_loss.detach().item())
            if v_loss < min_val_loss:
                min_val_loss = v_loss
                min_val_epoch = epoch + 1
                print("saving model...")
                torch.save(model.state_dict(), 'Score_Classifier-5.ckpt')
        print(min_val_epoch)
        if epoch-min_val_epoch > 200:
            break
    x_axis = np.arange(1, len(train_loss) + 1, 1)
    plt.plot(x_axis, np.asarray(train_loss), x_axis, np.asarray(val_loss))
    plt.title("training and validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(('training loss', 'validation loss'))
    plt.show()

data_path = "D:\\Breast Cancer\\Databases\\Forensic-Transfer\\layer2"
model = ModelLayer2.ScoreClassifier()

train(model, data_path, epochs=3000, batch_size=32, learning_rate=0.00005)

