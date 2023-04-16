import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import FO as offset
import LOMA_F as fs

__all__ = ['wrn_feature']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        #print(nb_layers)
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet_F(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,off_pro=0.5, a_max=3, r_max=0.7):
        super(WideResNet_F, self).__init__()
       
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = int((depth - 4) / 6)
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        
        self.off = offset.off_layer(off_pro)
        self.scale = fs.scale_layer(a_max, r_max)
        
                
    def forward(self, x, flag=-2, feature_p=0.5,scale_turn=False,offset_turn=False):
        if self.training:
            assert flag in [-1,0,1,2,3]

        if self.training and flag == -1:
            x=offset.compose(x,feature_p,scale_turn,offset_turn,self.scale,self.off)

        out = self.conv1(x)

        if self.training and flag == 0:
            out=offset.compose(out,feature_p,scale_turn,offset_turn,self.scale,self.off)

        out = self.block1(out)

        if self.training and flag == 1:
            out=offset.compose(out,feature_p,scale_turn,offset_turn,self.scale,self.off)

        out = self.block2(out)

        if self.training and flag == 2:
            out=offset.compose(out,feature_p,scale_turn,offset_turn,self.scale,self.off)#128, 320, 16, 16

        out = self.block3(out)

        if self.training and flag == 3:
            out=offset.compose(out,feature_p,scale_turn,offset_turn,self.scale,self.off)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn_feature(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet_F(**kwargs)
    return model