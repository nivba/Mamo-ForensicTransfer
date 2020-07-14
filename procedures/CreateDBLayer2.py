import os
import numpy as np
import torch
import ModelLayer1

class DBCreator:
    def __init__(self, path):
        np.random.seed(4)
        self.img_size = 4096
        self.tile_size = 256
        self.overlap = 0.75
        self.score_arr_size = int(np.floor((self.img_size - self.tile_size)/((1-self.overlap) * self.tile_size) + 1))
        self.step_size = int(np.floor(self.tile_size*(1-self.overlap)))
        self.val_size = 50
        self.train_real = []
        self.train_fake = []
        self.val_real = []
        self.val_fake = []
        self.eval_real = None
        self.eval_fake = None
        self.path = path
        self.model_1 = ModelLayer1.Mammo_FT()
        self.model_1.load_state_dict(torch.load("mammo-FT8.ckpt"))
        #self.model_1.load_state_dict(torch.load("mammo-FT7-3.ckpt", map_location=lambda storage, loc: storage))
        self.model_1.eval()

    def load_any(self, path, lst):
        i=0
        for img_name in os.listdir(path):
            i+=1
            print(i)
            img_path = os.path.join(path, img_name)
            img = np.load(img_path)
            rows, cols = img.shape
            LeftTop = np.zeros((self.img_size,self.img_size))
            LeftTop[0:rows,0:cols] = img
            LeftBottom = np.zeros((self.img_size,self.img_size))
            LeftBottom[self.img_size-rows:self.img_size,0:cols] = img
            RightTop = np.zeros((self.img_size,self.img_size))
            RightTop[0:rows, self.img_size-cols:self.img_size] = img
            RightBottom = np.zeros((self.img_size, self.img_size))
            RightBottom[self.img_size-rows:self.img_size, self.img_size - cols:self.img_size] = img
            Middle = np.zeros((self.img_size,self.img_size))
            Middle[self.img_size//2 - rows//2:self.img_size//2 - rows//2 + rows,
                    self.img_size//2 -cols//2:self.img_size//2 -cols//2 + cols] = img
            lst.append(self.get_score_array(LeftTop))
            lst.append(self.get_score_array(LeftBottom))
            lst.append(self.get_score_array(RightTop))
            lst.append(self.get_score_array(RightBottom))
            lst.append(self.get_score_array(Middle))



    def load_train(self):
        train_path = os.path.join(self.path,"train")
        train_real_path = os.path.join(train_path, "real")
        self.load_any(train_real_path, self.train_real)
        self.train_real = np.asarray(self.train_real).astype(np.float32)
        train_fake_path = os.path.join(train_path, "fake")
        self.load_any(train_fake_path, self.train_fake)
        self.train_fake = np.asarray(self.train_fake).astype(np.float32)
        np.save(os.path.join(train_real_path, "score_arr"), self.train_real)
        np.save(os.path.join(train_fake_path, "score_arr"), self.train_fake)

    def load_val(self):
        val_path = os.path.join(self.path, "val")
        val_real_path = os.path.join(val_path, "real")
        self.load_any(val_real_path, self.val_real)
        self.val_real = np.asarray(self.val_real).astype(np.float32)
        val_fake_path = os.path.join(val_path, "fake")
        self.load_any(val_fake_path, self.val_fake)
        self.val_fake = np.asarray(self.val_fake).astype(np.float32)
        np.save(os.path.join(val_real_path, "score_arr"), self.val_real)
        np.save(os.path.join(val_fake_path, "score_arr"), self.val_fake)


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
                tile_score, re = self.model_1(im_tile)
                tile_score = tile_score.item()
                score_array[i, j] = tile_score
                j +=1
                c+= self.step_size
            i += 1
            r += self.step_size
        return score_array

    def is_mammo(self,tile):
        return np.mean(tile) > 150




data_path = "D:\\Breast Cancer\\Databases\\Forensic-Transfer\\layer2"
db = DBCreator(data_path)
print("loading validation set...")
#db.load_val()
print("done.")
print("loading training set...")
db.load_train()
print("done.")