import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from procedures import ModelLayer1
from procedures import ModelLayer2
from scipy.signal import convolve2d
import cv2
import matplotlib as mpl

class Forgery_detection:
    def __init__(self, path):
        self.path = path
        self.imgs = []
        self.img_names = []
        self.score_arrs = []
        self.scores = []
        self.labels = []
        self.heat_map = []
        self.img_size = 4096
        self.tile_size = 256
        self.overlap = 0.75
        self.score_arr_size = int(np.floor((self.img_size - self.tile_size) / ((1 - self.overlap) * self.tile_size) + 1))
        self.step_size = int(np.floor(self.tile_size * (1 - self.overlap)))
        self.MammoFT = ModelLayer1.Mammo_FT()
        self.MammoFT.load_state_dict(torch.load("procedures\\mammo-FT.ckpt"))
        self.MammoFT.eval()
        self.Classifier = ModelLayer2.ScoreClassifier()
        self.Classifier.load_state_dict(torch.load("procedures\\Score_Classifier.ckpt"))
        self.Classifier.eval()

    def load_imgs(self):
        for name in os.listdir(self.path):
            self.img_names.append(name)
            img = np.load(os.path.join(self.path, name))
            self.imgs.append(img)

    def classification(self):
        for i,img in enumerate(self.imgs, start=0):
            score_arr = self.get_score_array(self.padd(img))
            self.score_arrs.append(score_arr)
            score_arr = torch.from_numpy(np.expand_dims(np.expand_dims(score_arr, axis=0), axis=0).astype(np.float32))
            score = self.Classifier(score_arr)
            self.scores.append(score.item())
            print("name: %s, score: %1.2f, label: %d"%
                  (self.img_names[i], score, 1 if score > 0.5 else 0))

    def padd(self, img):
        rows, cols =img.shape
        padd_img = np.zeros((self.img_size, self.img_size))
        padd_img[self.img_size//2-rows//2:self.img_size//2-rows//2 + rows,
           self.img_size//2-cols//2:self.img_size//2-cols//2 + cols] = img
        return padd_img

    def get_score_array(self, img):
        rows, cols = img.shape
        score_array = np.zeros((self.score_arr_size, self.score_arr_size))
        r = 0
        i = 0
        while r + self.tile_size <= rows:
            j = 0
            c = 0
            while c + self.tile_size <= cols:
                im_tile = img[r:r + self.tile_size, c:c + self.tile_size]
                if not self.is_mammo(im_tile):
                    j += 1
                    c += self.step_size
                    continue
                im_tile = np.expand_dims(np.expand_dims((im_tile/1100), axis=0), axis=0).astype(np.float32)
                im_tile =torch.from_numpy(im_tile)
                tile_score, re = self.MammoFT(im_tile)
                tile_score = tile_score.item()
                score_array[i, j] = tile_score
                j +=1
                c += self.step_size
            i += 1
            r += self.step_size
        return score_array

    def is_mammo(self,tile):
        return np.mean(tile) > 150

    def generate_heat_map(self):
        filter_size = int(1//(1-self.overlap))
        filter = np.ones((filter_size, filter_size))/(filter_size**2)
        for i,score in enumerate(self.score_arrs, start=0):
            pad_size = int(self.img_size//(self.tile_size*(1-self.overlap)))
            pad_score = np.zeros((pad_size,pad_size))
            pad_score[pad_size//2 - self.score_arr_size//2: pad_size//2 - self.score_arr_size//2 + self.score_arr_size,
                pad_size//2 - self.score_arr_size//2: pad_size//2 - self.score_arr_size//2 + self.score_arr_size]\
                = score
            base = convolve2d(pad_score, filter)
            heatmap = cv2.resize(base,(self.img_size, self.img_size))
            heatmap = heatmap/np.max(heatmap)
            img = self.imgs[i]
            rows, cols =img.shape
            cut_heatmap = np.copy(heatmap[self.img_size//2-rows//2:self.img_size//2-rows//2 + rows,
                self.img_size//2-cols//2:self.img_size//2-cols//2 + cols])
            #heatmap_img = cv2.applyColorMap(cut_heatmap, cv2.COLORMAP_JET)
            #print(heatmap_img.shape)
            img = self.display(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = np.expand_dims(img, axis=2)
            #fin = cv2.addWeighted(heatmap_img, 0.3, img, 0.7, 0)
            print("name: %s, score: %1.2f, label: %d" %
                  (self.img_names[i], self.scores[i], 1 if self.scores[i] > 0.5 else 0))
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.imshow(cut_heatmap, cmap='jet', alpha=0.2)
            #plt.colorbar()
            plt.show()

    def display(self,img_array):
        # displaying mammogram image to the canvas with color normalization
        mammo = np.copy(img_array)
        mammo[np.where(mammo > 800)] = 800
        mammo[np.where(mammo < 300)] = 300
        mammo = ((mammo - 300) / (800 - 300)) * 255
        return mammo

path ="D:\\Breast Cancer\\Databases\\Forensic-Transfer\\test set\\NVIDIA"
Fd = Forgery_detection(path)
Fd.load_imgs()
Fd.classification()
print("generate heat map")
Fd.generate_heat_map()
