
import numpy as np
import os

from matplotlib import pyplot as plt

class data_loader:
    def __init__(self, path):
        self.path = path
        self.real = []
        self.fake = []
        self.Tile_size = 256
        self. Tile_resize = 64

    def load(self):
        for img in os.listdir(self.path):
            img_name = img[0:-4]
            if img_name[-3:len(img_name)] == "map":
                continue
            #print(img_name)
            im_path = os.path.join(self.path, img)
            img_array = np.load(im_path)
            map_path = os.path.join(self.path,img_name+"_map.npy")
            map_array = np.load(map_path)
            self.break_to_tiles(img_array, map_array)
        self.real = np.array(self.real)
        self.fake = np.array(self.fake)
        print(str(len(self.real))+" real tiles")
        print(str(len(self.fake))+" fake tiles")

    def break_to_tiles(self, img_array, map_array):
        rows, cols = img_array.shape
        i = 0
        while i+self.Tile_size <= rows:
            j = 0
            while j+self.Tile_size <= cols:
                im_tile = img_array[i:i+self.Tile_size, j:j+self.Tile_size]
                if not self.is_mammo(im_tile):
                    j = j + int(max(min(self.Tile_size / 2, cols - self.Tile_size - j),1))
                    continue
                map_tile = map_array[i:i+self.Tile_size, j:j+self.Tile_size]
                if self.label_tile(map_tile) == 0:
                    self.real.append(im_tile)
                else:
                    self.fake.append(im_tile)
                j = j + int(max(min(self.Tile_size / 2, cols - self.Tile_size - j),1))
            i = i + int(max(min(self.Tile_size / 2, rows - self.Tile_size - i),1))

    def is_mammo(self,tile):
        return np.mean(tile) > 100

    def label_tile(self, map_tile):
        num_of_pixels = self.Tile_size*self.Tile_size
        num_of_fake_pixels = np.sum(map_tile)
        fake_part = num_of_fake_pixels/num_of_pixels
        if fake_part > 0.02:
            return 1
        return 0

    def show_real(self):
        for i in range(len(self.real)):
                plt.figure()
                plt.imshow(self.real[i], cmap="gray")
                plt.show()
    def show_fake(self):
        for i in range(len(self.fake)):
                plt.figure()
                plt.imshow(self.fake[i], cmap="gray")
                plt.show()

    def load_ephoc(self):
        real_idx = np.random.permutation(len(self.real))
        fake_idx = np.random.permutation(len(self.fake))
        return (self.real[real_idx, :, :]/1100).astype(np.float32), (self.fake[fake_idx, :, :]/1100).astype(np.float32)

