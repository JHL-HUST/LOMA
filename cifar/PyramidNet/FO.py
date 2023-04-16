from turtle import forward
import numpy
import torch
import numpy as np
import random
import torch.nn as nn
import math
from PIL import Image
from torch.nn import functional as F
import LOMA_F as fs
import sys


class off_layer(nn.Module):
    def __init__(self,off_pro):
        super(off_layer,self).__init__()
        self.off_pro = off_pro
       
        
    def forward(self,feature):
        # print('off')
        # print(self.off_pro)
        x_offset = random.uniform(-self.off_pro,self.off_pro)
        y_offset = random.uniform(-self.off_pro,self.off_pro)
        theta = torch.tensor([
            [1, 0, x_offset],
            [0, 1, y_offset]
        ], dtype=torch.float)
        N, C, W, H = feature.size()
        size_a = torch.Size((N, C, W , H))
        theta = theta.repeat(N,1,1)
        grid = F.affine_grid(theta.cuda(), size_a,align_corners=True)
        output = F.grid_sample(feature, grid,align_corners=True)
        #归一化
        nozero_num = torch.count_nonzero(output.data)
        allnum = output.numel()
        output = output*(allnum/nozero_num)
        return output

def compose(feature,feature_p,scale_turn,offset_turn,sc,of):
    # assert scale_turn!=False or offset_turn!=False
   
    if scale_turn==True and offset_turn == True :
        if random.uniform(0,1)<feature_p:
            feature = sc(feature)
            feature = of(feature)      
        
    elif scale_turn==True and offset_turn == False:
        if random.uniform(0,1)<feature_p:
            feature = sc(feature)
          
    elif scale_turn==False and offset_turn == True:
        if random.uniform(0,1)<feature_p:
            feature = of(feature)
          
    else:
        pass
    return feature 

