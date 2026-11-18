# CSPNeXt 백본과 FPN 디코더로 세그멘테이션 로짓을 계산한다.
from torch import nn
import torch
from torch.nn import functional as F
from .rtmdet_official import CSPNeXt, ConvBNAct

# 백본 특징을 융합해 픽셀별 클래스 로짓을 계산한다.
class CSPNeXtSeg(nn.Module):
    """RTMDet-L CSPNeXt backbone with a light stride-4/8/16/32 FPN decoder."""
    normalization = {'source_tensor': 'RGB float in [0, 1]', 'mmdet_bgr_to_rgb': False, 'mean_bgr_0_255': [103.53, 116.28, 123.675], 'std_bgr_0_255': [57.375, 57.12, 58.395]}

    # CSPNeXtSeg의 구성 요소와 설정을 초기화한다.
    def __init__(self, num_classes=3, class_weights=None, decoder_channels=128):
        super().__init__()
        self.num_classes = num_classes
        # 백본과 해상도별 디코더 경로를 구성한다.
        self.backbone = CSPNeXt(width=1.0, depth=1.0)
        channels = (128, 256, 512, 1024)
        self.lateral = nn.ModuleList((ConvBNAct(cin, decoder_channels, 1) for cin in channels))
        self.smooth = nn.ModuleList((ConvBNAct(decoder_channels, decoder_channels, 3) for _ in range(3)))
        self.classifier = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels, 3), nn.Conv2d(decoder_channels, num_classes, 1))
        self.register_buffer('class_weights', None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32))
        self.register_buffer('_mean_bgr', torch.tensor(self.normalization['mean_bgr_0_255']).view(1, 3, 1, 1))
        self.register_buffer('_std_bgr', torch.tensor(self.normalization['std_bgr_0_255']).view(1, 3, 1, 1))
        self.pretrained_parameters_loaded = 0
        self.architecture_signature = {'family': 'CSPNeXtSeg', 'backbone': 'RTMDet-L CSPNeXt', 'backbone_width': 1.0, 'backbone_depth': 1.0, 'decoder': 'light_fpn_stride4_8_16_32', 'decoder_channels': decoder_channels, 'normalization': self.normalization}

    # 입력을 정규화하고 네 단계의 백본 특징을 추출한다.
    def _features(self, x):
        x = x[:, (2, 1, 0)] * 255.0
        x = (x - self._mean_bgr) / self._std_bgr
        x = self.backbone.stem(x)
        c2 = self.backbone.stages[0](x)
        c3 = self.backbone.stages[1](c2)
        c4 = self.backbone.stages[2](c3)
        c5 = self.backbone.stages[3](c4)
        return (c2, c3, c4, c5)

    # 백본 특징을 융합해 픽셀별 클래스 로짓을 계산한다.
    def forward(self, x, target_mask=None):
        input_size = x.shape[-2:]
        # 1) 백본에서 여러 해상도의 특징 추출하기
        features = self._features(x)
        p2, p3, p4, p5 = (layer(feature) for layer, feature in zip(self.lateral, features))
        # 2) 상위 특징을 합쳐 입력 해상도의 로짓 복원하기
        p4 = self.smooth[2](p4 + F.interpolate(p5, size=p4.shape[-2:], mode='nearest'))
        p3 = self.smooth[1](p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest'))
        p2 = self.smooth[0](p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest'))
        logits = F.interpolate(self.classifier(p2), size=input_size, mode='bilinear', align_corners=False)
        return logits

# 지정 클래스 수와 가중치로 세그멘테이션 모델을 만든다.
def build_cspnext_seg(num_classes=3, class_weights=None):
    return CSPNeXtSeg(num_classes, class_weights)
