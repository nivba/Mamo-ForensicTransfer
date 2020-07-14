import os
import numpy as np
import torch
import ModelLayer1
from matplotlib import pyplot as plt

class DataLoader2:
    def __init__(self, path):
        np.random.seed(4)
        self.img_size = 4096
        self.tile_size = 256
        self.overlap = 0.75
        self.score_arr_size = int(np.floor((self.img_size - self.tile_size)/((1-self.overlap) * self.tile_size) + 1))
        self.step_size = int(np.floor(self.tile_size*(1-self.overlap)))
        self.val_size = 50
        self.train_real = None
        self.train_fake = None
        self.val_real = None
        self.val_fake = None
        self.eval_real = None
        self.eval_fake = None
        self.path = path


    def load_train(self):
        train_path = os.path.join(self.path,"train")
        train_real_path = os.path.join(train_path, "real")
        train_fake_path = os.path.join(train_path, "fake")
        self.train_real = np.load(os.path.join(train_real_path, "score_arr.npy"))
        self.train_fake = np.load(os.path.join(train_fake_path, "score_arr.npy"))
        self.shuffle_data()
        self.eval_fake = self.train_fake[0:self.val_size, :, :]
        self.eval_real = self.train_real[0:self.val_size, :, :]

    def load_val(self):
        val_path = os.path.join(self.path, "val")
        val_real_path = os.path.join(val_path, "real")
        val_fake_path = os.path.join(val_path, "fake")
        self.val_real = np.load(os.path.join(val_real_path, "score_arr.npy"))
        self.val_fake = np.load(os.path.join(val_fake_path, "score_arr.npy"))
        np.random.shuffle(self.val_real)
        np.random.shuffle(self.val_fake)
        self.val_real = self.val_real[0:self.val_size, :, :]
        self.val_fake = self.val_fake[0:self.val_size, :, :]


    def shuffle_data(self):
        np.random.shuffle(self.train_real)
        np.random.shuffle(self.train_fake)


#data_path = "D:\\Breast Cancer\\Databases\\Forensic-Transfer\\layer2"
#dl = DataLoader2(data_path)
#dl.load_val()