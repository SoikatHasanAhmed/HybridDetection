# https://github.com/sunny2109/MobileSR-NTIRE2022/blob/main/models/mobilesr.py

from torch import nn
from .ConvDownscale import get_downsample_layer, DWConvDownsampling

from spikingjelly.clock_driven import layer, neuron, surrogate
###########################################################################
   
class SpikeBlock(nn.Module):
    def __init__(self, c_in, c_out, down_scale=2):
        super(SpikeBlock, self).__init__()
        self.down_scale = down_scale
        if self.down_scale==0:
            self.convdown = layer.SeqToANNContainer(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.convdown = layer.SeqToANNContainer(get_downsample_layer(c_in,c_out,down_scale))

        self.neuron = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=1.,
            surrogate_function=surrogate.ATan(),
            detach_reset=True, backend='torch',
        )
    
    def forward(self, x):   
        x = self.convdown(x)
        x = self.neuron(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1,
                  activation='relu',  down_scale=2):
        
        super(Conv2dBlock, self).__init__()
        self.down_scale = down_scale
        if self.down_scale==0:
           self.conv  = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
                nn.BatchNorm2d(output_dim)
                # LayerNorm2d(output_dim)
            )
        else:
            self.conv  = get_downsample_layer(input_dim,output_dim,down_scale)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):

        x = self.conv(x)
       
        if self.activation:
            x = self.activation(x)

        return x
class DwConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1,
                  activation='relu',  down_scale=2):
        
        super(DwConv2dBlock, self).__init__()
        self.down_scale = down_scale
        if self.down_scale==0:
           self.conv  = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding,groups=input_dim),
                nn.Conv2d(input_dim, output_dim, 1),
                nn.BatchNorm2d(output_dim)
                # LayerNorm2d(output_dim)
            )
        else:
            self.conv  = DWConvDownsampling(input_dim,output_dim,down_scale)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):

        x = self.conv(x)
       
        if self.activation:
            x = self.activation(x)

        return x


# if __name__== '__main__':
#     a = torch.randn(1, 10, 5, 256, 256)
#     model = SpikeBlock(5,10,10,0)
#     print(model)
#     output = model(a)
#     print(output.shape)