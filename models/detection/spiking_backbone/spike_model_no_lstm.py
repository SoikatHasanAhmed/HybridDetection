import torch
from torch import nn
import spikingjelly
from .attention import StoA
from .network_blocks import SpikeBlock, Conv2dBlock


class Backbone(nn.Module):
    """Backbone without LSTM layers - for sequence_length=1 experiments."""
    
    def __init__(self):
        super(Backbone, self).__init__()
        ch = 256
        
        self.features_01 = nn.Sequential(
            SpikeBlock(2, 64),  #1/2
            SpikeBlock(64, 128), #1/4,
        )

        self.features_23 = nn.Sequential(
            SpikeBlock(128, 256),#1/8    
            SpikeBlock(256, ch,down_scale=0)      
           
        )

        self.ann_features_1 = nn.Sequential(
            Conv2dBlock(ch, ch, 3, padding=1,down_scale=0), # 1/16 out
            Conv2dBlock(ch, ch, 3, padding=1),
        )

        self.ann_features_2 = nn.Sequential(
            Conv2dBlock(ch, ch, 3, padding=1,down_scale=0),
            Conv2dBlock(ch, ch, 3, padding=1), # 1/32
        )
   
        self.accumulate_1 = StoA(T=10, in_channels=ch)
      
   
    def forward(self, x, h_c):
        """
        Forward pass without LSTM.
        h_c is ignored but kept for interface compatibility.
        """
        # x shape: (B, seq_len*events_per_frame, H, W)
        x = torch.cat((x[:,0:10,:,:].unsqueeze(2), x[:,10:,:,:].unsqueeze(2)), dim=2)
        x = self.features_01(x)
        x_1 = self.features_23(x)
        x_1 = self.accumulate_1(x_1)
        x_2 = self.ann_features_1(x_1)
        
        # No LSTM - pass through directly
        x_3 = self.ann_features_2(x_2)
  
        output = {2: x_1, 3: x_2, 4: x_3}

        return output, h_c

    def reset(self):
        for f in [self.features_01, self.features_23]:
             spikingjelly.clock_driven.functional.reset_net(f)
