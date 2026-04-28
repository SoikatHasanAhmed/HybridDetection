from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

from ...spiking_backbone import Backbone
from ...spiking_backbone.spike_model_no_lstm import Backbone as BackboneNoLSTM
from .build import build_yolox_fpn, build_yolox_head
from utils.timers import TimerDummy as CudaTimer

from data.utils.types import BackboneFeatures, LstmStates


class YoloXDetectorHybrid(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        # Load backbone based on config
        backbone_name = backbone_cfg.name
        if backbone_name == 'v1_attention':
            self.backbone = Backbone()
        elif backbone_name == 'v1_attention_no_lstm':
            self.backbone = BackboneNoLSTM()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        in_channels = (256, 256, 256)
        print(fpn_cfg)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor, h_c) -> \
            Tuple[BackboneFeatures]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x,h_c)
        return backbone_features, states

    def forward_detect(self,
                       backbone_features: BackboneFeatures,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.yolox_head(fpn_features, targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        assert losses is None
        return outputs, losses

    def forward(self,
                x: th.Tensor,
                retrieve_detections: bool = True,
                targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None]]:
        backbone_features = self.forward_backbone(x)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses
        outputs, losses = self.forward_detect(backbone_features=backbone_features, targets=targets)
        return outputs, losses

