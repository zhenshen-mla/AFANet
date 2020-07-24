import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    def __init__(self, channels):
        super(CAM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=channels * 8, out_features=round(channels / 2), bias=False)
        self.fc1 = nn.Linear(in_features=round(channels / 2), out_features=channels * 4, bias=False)
        self.fc2 = nn.Linear(in_features=round(channels / 2), out_features=channels * 4, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        d_A = self.gap(x1)
        d_B = self.gap(x2)
        d_concate = torch.cat((d_A, d_B), dim=1)
        d_concate = d_concate.view(d_concate.size(0), -1)

        g = self.relu(self.fc(d_concate))
        g1 = torch.unsqueeze(self.fc1(g), dim=1)
        g2 = torch.unsqueeze(self.fc2(g), dim=1)
        w = torch.cat((g1, g2), dim=1)
        w = F.softmax(w, dim=1).permute(1, 0, 2)
        w_self = w[0].view(w[0].size(0), w[0].size(1), 1, 1)
        w_other = w[1].view(w[1].size(0), w[1].size(1), 1, 1)

        out1 = w_self * x1 + w_other * x2
        out2 = w_self * x2 + w_other * x1
        return out1, out2


class SAM(nn.Module):
    def __init__(self, channels):
        super(SAM, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.fc1 = nn.Linear(in_features=channels, out_features=channels, bias=False)
        self.fc2 = nn.Linear(in_features=channels, out_features=channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        b, c, height, width = x1.size()
        d_A = torch.mean(x1, dim=1, keepdim=True)
        d_B = torch.mean(x2, dim=1, keepdim=True)
        d_concat = torch.cat((d_A, d_B), dim=1)

        g = self.relu(self.conv(d_concat))
        g_vector = g.view(b, -1)
        g1 = torch.unsqueeze(self.fc1(g_vector), dim=1)
        g2 = torch.unsqueeze(self.fc2(g_vector), dim=1)
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
    def __init__(self, channels=49):
        super(AFA_layer_sam, self).__init__()
        self.sam = SAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1, x2 = self.sam(x1, x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        return x1, x2

'''
Sometimes the difference between the multiple tasks is so great that it is easy to be dominated by one task when updating parameters. 
When processing such tasks, we calibrate the return gradient.
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
    def __init__(self, channels=49):
        super(AFA_layer_sam_data, self).__init__()
        self.sam = SAM(channels)
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


