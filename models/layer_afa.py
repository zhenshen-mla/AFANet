import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    def __init__(self, channels, r=16, L=32):
        '''
        channels: Bottleneck's planes
        r: reduction ration
        L: Minimum dimension threshold
        '''
        super(CAM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cat_channels = channels * 8
        d = max(int(self.cat_channels/r), L)
        self.fc = nn.Linear(in_features=self.cat_channels, out_features=d, bias=False)
        # fc layer is designed as a dimensionality reduction layer
        self.trans1 = nn.Linear(in_features=d, out_features=round(self.cat_channels/2), bias=False)
        self.trans2 = nn.Linear(in_features=d, out_features=round(self.cat_channels/2), bias=False)
        # the layer's weights are c*(2c/r).
        # parameter matrices
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        d_A = self.gap(x1)
        d_B = self.gap(x2)
        d_concate = torch.cat((d_A, d_B), dim=1)
        d_concate = d_concate.view(d_concate.size(0), -1)

        g = self.relu(self.fc(d_concate))
        g1 = torch.unsqueeze(self.trans1(g), dim=1)
        g2 = torch.unsqueeze(self.trans2(g), dim=1)
        w = torch.cat((g1, g2), dim=1)
        w = F.softmax(w, dim=1).permute(1, 0, 2)
        w_self = w[0].view(w[0].size(0), w[0].size(1), 1, 1)
        w_other = w[1].view(w[1].size(0), w[1].size(1), 1, 1)

        out1 = w_self * x1 + w_other * x2
        out2 = w_self * x2 + w_other * x1
        return out1, out2


class SAM(nn.Module):
    def __init__(self, h=19, w=19, kernel_size=7, padding=3):
        '''
        h: height
        w: width
        kernel_size, padding: conv layer's params
        '''
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.channels = h * w
        self.trans1 = nn.Linear(in_features=self.channels, out_features=self.channels, bias=False)
        self.trans2 = nn.Linear(in_features=self.channels, out_features=self.channels, bias=False)
        # the layer's weights are hw*hw.
        # parameter matrices
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        b, c, height, width = x1.size()
        d_A = torch.mean(x1, dim=1, keepdim=True)
        d_B = torch.mean(x2, dim=1, keepdim=True)
        d_concat = torch.cat((d_A, d_B), dim=1)

        g = self.relu(self.conv(d_concat))
        g_vector = g.view(b, -1)
        g1 = torch.unsqueeze(self.trans1(g_vector), dim=1)
        g2 = torch.unsqueeze(self.trans2(g_vector), dim=1)
        w = torch.cat((g1, g2), dim=1)
        w = F.softmax(w, dim=1).permute(1, 0, 2)
        w_self = w[0].view(w[0].size(0), -1, height, width)
        w_other = w[1].view(w[1].size(0), -1, height, width)

        out1 = w_self * x1 + w_other * x2
        out2 = w_self * x2 + w_other * x1
        return out1, out2


class AFA_layer_cam(nn.Module):
    def __init__(self, channels=512):
        super(AFA_layer_cam, self).__init__()
        self.cam = CAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1, x2 = self.cam(x1, x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        return x1, x2


class AFA_layer_sam(nn.Module):
    def __init__(self, h=7, w=7):
        super(AFA_layer_sam, self).__init__()
        self.sam = SAM(h, w)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1, x2 = self.sam(x1, x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        return x1, x2


'''
Sometimes the difference between the two tasks is so great that it is easy to be dominated by one task when updating parameters. 
When dealing with these tasks, we can return the calibrated gradient.
'''
class AFA_layer_cam_data(nn.Module):
    def __init__(self, channels=512):
        super(AFA_layer_cam_data, self).__init__()
        self.cam = CAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        data1 = x1.data
        data2 = x2.data
        o1, o2 = self.cam(data1, data2)
        out1 = o1 + x1 - data1
        out2 = o2 + x2 - data2
        out1 = self.relu(out1)
        out2 = self.relu(out2)
        return out1, out2


class AFA_layer_sam_data(nn.Module):
    def __init__(self, h=7, w=7):
        super(AFA_layer_sam_data, self).__init__()
        self.sam = SAM(h, w)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        data1 = x1.data
        data2 = x2.data
        o1, o2 = self.sam(data1, data2)
        out1 = o1 + x1 - data1
        out2 = o2 + x2 - data2
        out1 = self.relu(out1)
        out2 = self.relu(out2)
        return out1, out2


