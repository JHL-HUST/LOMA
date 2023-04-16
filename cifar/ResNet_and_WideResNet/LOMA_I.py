import numpy
import torch
import numpy as np
import random
import torch.nn as nn
import math
from PIL import Image
import torchvision.transforms as transforms


def int_round(x):
    return np.int(np.round(x))

class LOMA(object):
    def __init__(self, probability=0.5, a_max=3, r_max=0.7):
        assert isinstance(probability, float)
        self.probability = probability
        self.a_max = a_max
        self.r_max = r_max

        
    def __call__(self, img):
        
        if random.uniform(0, 1) < self.probability:
            # print('input')
            # print(self.r_max,self.a_max,self.probability)
            cols=img.size[0]
            rows=img.size[1]
            center_rows = int_round(random.uniform(1, rows - 2))
            center_cols = int_round(random.uniform(1, cols - 2))
            radius = random.uniform(0.03*max(rows,cols), self.r_max*max(rows,cols))
            choice=random.randint(0,1)
            spect_ratio1 = 1
            spect_ratio2 = 1
            if choice==1:
                spect_ratio1 = random.uniform(1, self.a_max)
            else:
                spect_ratio2 = random.uniform(1, self.a_max)
            img_np = np.array(img)
            new_img_np = img_np.copy()
            cols_np = np.arange(cols)
            rows_np = np.arange(rows)
            cols_np_t = np.tile(cols_np, (rows, 1))
            cols_pow = pow(cols_np_t - center_cols,2)
            rows_np_t = np.tile(rows_np, (cols, 1))
            rows_pow = pow(rows_np_t - center_rows, 2)
            dis = np.sqrt(cols_pow+rows_pow.transpose())
            judge=spect_ratio1*abs(rows_np_t-center_rows).transpose()+spect_ratio2*abs((cols_np_t-center_cols))#rhombus
            # judge=np.sqrt(spect_ratio1*cols_pow+spect_ratio2*rows_pow.transpose())#ellipse
            
            index=np.where(judge <= radius)
            index_rows = np.rint(index[0]).astype(int)
            index_cols = np.rint(index[1]).astype(int)
            
            dis_val=dis[np.where(judge <= radius)]
            
            old_i=np.floor(dis_val/radius*(index_rows-center_rows)+center_rows)
            old_j= np.floor(dis_val / radius * (index_cols - center_cols) + center_cols)
            new_img_np[index_rows.astype('int64'),index_cols.astype('int64'),:]=img_np[old_i.astype('int64'),old_j.astype('int64'),:]
            new_img = Image.fromarray(new_img_np)
            return new_img
        else:
            return img


