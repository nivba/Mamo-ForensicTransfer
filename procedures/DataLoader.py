
import numpy as np
import os
#from cv2 import Laplacian
from matplotlib import pyplot as plt
#from scipy.ndimage.filters import laplace
class data_loader:
    np.random.seed(3)
    def __init__(self, path):
        self.Tile_size = 256
        self.path = path
        self.real = []
        self.fake = []
        self.val_size = 100
        self.val_real = []
        self.val_fake = []



    def load_train(self, augmentation=0):
        train_path = os.path.join(self.path,'train')
        for img in os.listdir(train_path):
            img_name = img[0:-4]
            if img_name[-3:len(img_name)] == "map":
                continue
            im_path = os.path.join(train_path, img)
            img_array = np.load(im_path)
            map_path = os.path.join(train_path,img_name+"_map.npy")
            map_array = np.load(map_path)
            r, f = self.break_to_tiles(img_array, map_array,2)
            self.real.extend(r)
            self.fake.extend(f)
        np.random.shuffle(self.real)
        self.real = self.real[0:len(self.fake)]
        self.real = np.array(self.real).astype(np.float32)/1100
        self.fake = np.array(self.fake).astype(np.float32)/1100
        if (augmentation == 1):
            self.real = self.augmentation(self.real)
            self.fake = self.augmentation(self.fake)
        self.real = np.expand_dims(self.real, axis=1)
        self.fake = np.expand_dims(self.fake, axis=1)
        self.real_eval = np.copy(self.real[0:self.val_size, :, :, :])
        self.fake_eval = np.copy(self.fake[0:self.val_size, :, :, :])
        print("training set:")
        print(str(len(self.real))+" real tiles")
        print(str(len(self.fake))+" fake tiles")

    def load_val(self, augmentation=0):
        val_path = os.path.join(self.path, 'val')
        for img in os.listdir(val_path):
            img_name = img[0:-4]
            if img_name[-3:len(img_name)] == "map":
                continue
            # print(img_name)
            im_path = os.path.join(val_path, img)
            img_array = np.load(im_path)
            map_path = os.path.join(val_path, img_name + "_map.npy")
            map_array = np.load(map_path)
            r, f = self.break_to_tiles(img_array, map_array,4)
            self.val_real.extend(r)
            self.val_fake.extend(f)
        np.random.shuffle(self.val_real)
        np.random.shuffle(self.val_fake)
        self.val_real = np.array(self.val_real).astype(np.float32)/1100
        self.val_fake = np.array(self.val_fake).astype(np.float32)/1100
        if (augmentation == 1):
            self.val_real = self.augmentation(self.val_real)
            self.val_fake = self.augmentation(self.val_fake)
        self.val_real = self.val_real[0:self.val_size, :, :]
        self.val_real = np.expand_dims(self.val_real, axis=1)
        self.val_fake = self.val_fake[0:self.val_size, :, :]
        self.val_fake = np.expand_dims(self.val_fake, axis=1)
        print("validation set:")
        print(str(len(self.val_real)) + " real tiles")
        print(str(len(self.val_fake)) + " fake tiles")

    def break_to_tiles(self, img_array, map_array, part):
        real = []
        fake = []
        rows, cols = img_array.shape
        i = 0
        while i+self.Tile_size <= rows:
            j = 0
            while j+self.Tile_size <= cols:
                im_tile = img_array[i:i+self.Tile_size, j:j+self.Tile_size]
                if not self.is_mammo(im_tile):
                    j = j + int(max(min(self.Tile_size / part, cols - self.Tile_size - j),1))
                    continue
                map_tile = map_array[i:i+self.Tile_size, j:j+self.Tile_size]
                fake_part = self.fake_part(map_tile)
                if fake_part > 0.08:
                    fake.append(im_tile)
                elif fake_part == 0:
                    real.append(im_tile)
                j = j + int(max(min(self.Tile_size / part, cols - self.Tile_size - j),1))
            i = i + int(max(min(self.Tile_size / part, rows - self.Tile_size - i),1))
        return real, fake

    def is_mammo(self,tile):
        return np.mean(tile) > 150

    def fake_part(self, map_tile):
        num_of_pixels = self.Tile_size*self.Tile_size
        num_of_fake_pixels = np.sum(map_tile)
        fake_part = num_of_fake_pixels/num_of_pixels
        return fake_part

    def show_real(self):
        for i in range(len(self.real)):
                plt.figure()
                im = np.squeeze(self.real[i, :, :])
                plt.imshow(im, cmap="gray")
                plt.show()
    def show_fake(self):
        for i in range(len(self.fake)):
                plt.figure()
                im=np.squeeze(self.fake[i,:,:])
                plt.imshow(im, cmap="gray")
                plt.show()

    def load_ephoc(self):
        np.random.shuffle(self.real)
        np.random.shuffle(self.fake)

    def augmentation(self, img_array):
        augmented = []
        num_of_img = np.shape(img_array)[0]
        for i in range(num_of_img):
            img = img_array[i, :, :]
            for j in range(4):
                img = np.rot90(img)
                augmented.append(np.copy(img))
                augmented.append(np.fliplr(img))
        return np.asarray(augmented)

#loader = data_loader("D:\\Breast Cancer\\Databases\\Forensic-Transfer")
#loader.load_train()
#loader.load_ephoc()
#loader.show_real()
