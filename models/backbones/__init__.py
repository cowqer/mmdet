# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .arconvresnet import ARResNet, ARResNetV1d
from .pconvresnet import PResNet , PResNetV1d 
from .apconvresnet import APResNet, APResNetV1d
from .arpconvresnet import ARPResNet, ARPResNetV1d
from .ppconvresnet import PPResNet, PPResNetV1d
from .rpconvresnet import RPResNet, RPResNetV1d
from .pppconvresnet import PPPResNet, PPPResNetV1d
from .Gatedpconvresnet import GatedPResNet, GatedPResNetV1d
from .GatedSpconvresnet import GatedSPResNet, GatedSPResNetV1d
from .Gatedpconv1resnet import GatedP1ResNet
from .Gatedhwconvresnet import GatedHWResNet, GatedHWResNetV1d
from .Gatedhwconv1resnet import GatedHW1ResNet, GatedHW1ResNetV1d
from .sgpconvresnet import SGPCResNet
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet','ARResNet','ARResNetV1d',
    'PResNet','PResNetV1d','APResNet','APResNetV1d',
    'ARPResNet','ARPResNetV1d','PPResNet','PPResNetV1d','PPPResNet','PPPResNetV1d',
    'RPResNet','RPResNetV1d','GatedPResNet','GatedPResNetV1d','GatedSPResNet','GatedSPResNetV1d',
    'GatedP1ResNet','GatedHWResNet','GatedHWResNetV1d','GatedHW1ResNet','GatedHW1ResNetV1d',
    'SGPCResNet'
]
