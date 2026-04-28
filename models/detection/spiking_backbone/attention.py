import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torchvision


# Deformable Convolution Module
class DeformConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        # Initialize learnable weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels//groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Initialize offset convolution layer
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding, dilation, groups)

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in')
        nn.init.constant_(self.bias, 0)

    def forward(self, input, mask=None):
        # Calculate offsets using the offset convolution layer
        offset = self.offset_conv(input)

        # Keep API compatibility; current implementation does not use external mask.
        output = torchvision.ops.deform_conv2d(
            input=input,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            mask=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        return output

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()

        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        batch_size, channels, height, width = x.size()

        # Compute key, query, and value
        keys = self.key(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        queries = self.query(x).view(batch_size, -1, height * width)
        values = self.value(x).view(batch_size, -1, height * width).permute(0, 2, 1)

        # Compute attention scores
        scores = torch.bmm(queries, keys)
        attention_weights = self.softmax(scores)

        # Compute weighted sum using attention weights
        attended_values = torch.bmm(values, attention_weights).permute(0, 2, 1)

        attended_values = attended_values.view(batch_size, channels, height, width)

        return attended_values

class StoA(nn.Module):
    def __init__(self, T=10, in_channels=256):
        super(StoA, self).__init__()
        self.readout_layer = nn.Sequential(
            nn.Conv2d(in_channels=T,out_channels=T,kernel_size=3, stride=1, padding=1),
            )
        self.t_SelfAtten = SelfAttention2D(in_channels=T)
        self.groupwiseDformConv = nn.Sequential(
            DeformConv2D(in_channels=T,out_channels=T,kernel_size=5, stride=1, padding=2, dilation=1, groups=T),
            nn.ReLU()
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=T, out_channels=1,kernel_size=1),
            nn.InstanceNorm2d(T),
            nn.ReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self,x):
        # print(x.shape)
        b, t, c, h, w = x.shape
        x_tmp = x. permute(0,2,1,3,4).reshape(b*c,t,h,w) # b,t,c,h,w -> b,c,t,h,w

        out = self.readout_layer(x_tmp)
        out = self.groupwiseDformConv(out)
        out = self.t_SelfAtten(out)
        out = self.conv1x1_1(out).squeeze(1).reshape(b,c,h,w)
        out = out * torch.sigmoid(x.sum(dim=1))
        out = self.conv1x1_2(out)
        return out

if __name__ == '__main__':
    x= torch.rand(8,5,256,32,32)
    op = StoA(T=5)
    print(op(x).shape)


