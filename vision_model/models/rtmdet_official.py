# 체크포인트와 호환되는 CSPNeXt 백본과 RTMDet 검출기를 구성한다.
# Copyright (c) OpenMMLab. All rights reserved. Apache-2.0.
# Adapted from MMDetection v3.3.0 and local checkpoint-compatible implementation.
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import batched_nms

# 합성곱·배치 정규화·활성화를 묶는다.
class ConvBNAct(nn.Module):

    # ConvBNAct의 구성 요소와 설정을 초기화한다.
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1)
        self.act = nn.SiLU(inplace=True)

    # 합성곱·배치 정규화·활성화를 묶는다.
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 깊이별 합성곱과 채널 혼합 합성곱을 연결한다.
class DepthwiseSeparableConv(nn.Sequential):

    # DepthwiseSeparableConv의 구성 요소와 설정을 초기화한다.
    def __init__(self, cin, cout, kernel=5):
        super().__init__(ConvBNAct(cin, cin, kernel, groups=cin), ConvBNAct(cin, cout, 1))

# 분리 합성곱과 선택적 잔차 연결을 구성한다.
class CSPNeXtBlock(nn.Module):

    # CSPNeXtBlock의 구성 요소와 설정을 초기화한다.
    def __init__(self, channels, identity=True):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 5)
        self.add_identity = identity

    # 분리 합성곱과 선택적 잔차 연결을 계산한다.
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add_identity else y

# 전역 특징에서 채널별 가중치를 계산한다.
class ChannelAttention(nn.Module):

    # ChannelAttention의 구성 요소와 설정을 초기화한다.
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid(inplace=True)

    # 전역 특징에서 채널별 가중치를 계산한다.
    def forward(self, x):
        return x * self.act(self.fc(self.pool(x)))

# 특징을 두 경로로 처리하고 결합한다.
class CSPLayer(nn.Module):

    # CSPLayer의 구성 요소와 설정을 초기화한다.
    def __init__(self, cin, cout, blocks=3, identity=True, attention=False):
        super().__init__()
        mid = int(cout * 0.5)
        self.main_conv = ConvBNAct(cin, mid, 1)
        self.short_conv = ConvBNAct(cin, mid, 1)
        self.blocks = nn.Sequential(*(CSPNeXtBlock(mid, identity) for _ in range(blocks)))
        self.attention = ChannelAttention(2 * mid) if attention else nn.Identity()
        self.final_conv = ConvBNAct(2 * mid, cout, 1)

    # 특징을 두 경로로 처리하고 결합한다.
    def forward(self, x):
        main = self.blocks(self.main_conv(x))
        short = self.short_conv(x)
        return self.final_conv(self.attention(torch.cat((main, short), dim=1)))

# 서로 다른 풀링 범위의 특징을 결합한다.
class SPPBottleneck(nn.Module):

    # SPPBottleneck의 구성 요소와 설정을 초기화한다.
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels // 2, 1)
        self.poolings = nn.ModuleList((nn.MaxPool2d(k, 1, k // 2) for k in (5, 9, 13)))
        self.conv2 = ConvBNAct(2 * channels, channels, 1)

    # 서로 다른 풀링 범위의 특징을 결합한다.
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.poolings], 1))

# 여러 해상도의 CSPNeXt 백본 특징을 추출한다.
class CSPNeXt(nn.Module):

    # CSPNeXt의 구성 요소와 설정을 초기화한다.
    def __init__(self, width=1.0, depth=1.0):
        super().__init__()
        channels = [int(c * width) for c in (64, 128, 256, 512, 1024)]
        self.out_channels = channels[2:]
        self.block_counts = [max(round(n * depth), 1) for n in (3, 6, 6, 3)]
        self.stem = nn.Sequential(ConvBNAct(3, channels[0] // 2, 3, 2), ConvBNAct(channels[0] // 2, channels[0] // 2), ConvBNAct(channels[0] // 2, channels[0]))
        # 백본 해상도별 단계를 구성한다.
        self.stages = nn.ModuleList()
        for i, blocks in enumerate(self.block_counts):
            layers = [ConvBNAct(channels[i], channels[i + 1], 3, 2)]
            if i == 3:
                layers.append(SPPBottleneck(channels[i + 1]))
            layers.append(CSPLayer(channels[i + 1], channels[i + 1], blocks, identity=i != 3, attention=True))
            self.stages.append(nn.Sequential(*layers))

    # 여러 해상도의 CSPNeXt 백본 특징을 추출한다.
    def forward(self, x):
        x = self.stem(x)
        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:
                out.append(x)
        return tuple(out)

# 위아래 경로를 통해 여러 해상도의 특징을 융합한다.
class CSPNeXtPAFPN(nn.Module):

    # CSPNeXtPAFPN의 구성 요소와 설정을 초기화한다.
    def __init__(self, channels, out_channels, num_csp_blocks=3):
        super().__init__()
        self.num_csp_blocks = num_csp_blocks
        self.reduce_layers = nn.ModuleList((ConvBNAct(channels[i], channels[i - 1], 1) for i in (2, 1)))
        self.top_down_blocks = nn.ModuleList((CSPLayer(2 * channels[i - 1], channels[i - 1], num_csp_blocks, False) for i in (2, 1)))
        self.downsamples = nn.ModuleList((ConvBNAct(channels[i], channels[i], 3, 2) for i in (0, 1)))
        self.bottom_up_blocks = nn.ModuleList((CSPLayer(2 * channels[i], channels[i + 1], num_csp_blocks, False) for i in (0, 1)))
        self.out_convs = nn.ModuleList((ConvBNAct(c, out_channels) for c in channels))

    # 위아래 경로를 통해 여러 해상도의 특징을 융합한다.
    def forward(self, inputs):
        # 1) 높은 단계 특징을 낮은 단계로 전달하기
        inner = [inputs[-1]]
        for j, i in enumerate((2, 1)):
            high = self.reduce_layers[j](inner[0])
            inner[0] = high
            inner.insert(0, self.top_down_blocks[j](torch.cat((F.interpolate(high, size=inputs[i - 1].shape[-2:], mode='nearest'), inputs[i - 1]), 1)))
        # 2) 낮은 단계 특징을 다시 모아 출력하기
        outs = [inner[0]]
        for i in (0, 1):
            outs.append(self.bottom_up_blocks[i](torch.cat((self.downsamples[i](outs[-1]), inner[i + 1]), 1)))
        return tuple((conv(x) for conv, x in zip(self.out_convs, outs)))

# 검출 클래스 로짓·거리·기준점을 계산한다.
class RTMDetSepBNHead(nn.Module):
    strides = (8, 16, 32)

    # RTMDetSepBNHead의 구성 요소와 설정을 초기화한다.
    def __init__(self, channels=256, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList((nn.ModuleList((ConvBNAct(channels, channels) for _ in range(2))) for _ in self.strides))
        self.reg_convs = nn.ModuleList((nn.ModuleList((ConvBNAct(channels, channels) for _ in range(2))) for _ in self.strides))
        self.rtm_cls = nn.ModuleList((nn.Conv2d(channels, num_classes, 1) for _ in self.strides))
        self.rtm_reg = nn.ModuleList((nn.Conv2d(channels, 4, 1) for _ in self.strides))
        for level in (1, 2):
            for i in (0, 1):
                self.cls_convs[level][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[level][i].conv = self.reg_convs[0][i].conv
        # 공유 합성곱 연결 후 검출 헤드 가중치를 초기화한다.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for cls in self.rtm_cls:
            nn.init.constant_(cls.bias, -math.log(99.0))

    # 검출 클래스 로짓·거리·기준점을 계산한다.
    def forward(self, feats):
        # 1) 해상도별 클래스 점수와 거리 수집하기
        scores, distances, priors = ([], [], [])
        for i, (x, stride) in enumerate(zip(feats, self.strides)):
            cls, reg = (x, x)
            for c in self.cls_convs[i]:
                cls = c(cls)
            for c in self.reg_convs[i]:
                reg = c(reg)
            score = self.rtm_cls[i](cls)
            dist = self.rtm_reg[i](reg).float().clamp(-16.0, 16.0).exp() * stride
            batch, _, h, w = score.shape
            scores.append(score.permute(0, 2, 3, 1).reshape(batch, -1, self.num_classes).float())
            distances.append(dist.permute(0, 2, 3, 1).reshape(batch, -1, 4))
            # 2) 각 특징 격자의 기준점 좌표 만들기
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
            priors.append(torch.stack((xx * stride, yy * stride, torch.full_like(xx, stride)), -1).reshape(-1, 3).float())
        return (torch.cat(scores, 1), torch.cat(distances, 1), torch.cat(priors))

# 기준점과 네 방향 거리에서 박스 좌표를 복원한다.
def distance_to_bbox(priors, distances):
    return torch.cat((priors[None, :, :2] - distances[..., :2], priors[None, :, :2] + distances[..., 2:]), -1)

# RTMDet 검출과 임계값·NMS 처리를 수행한다.
class RTMDet(nn.Module):

    # RTMDet의 구성 요소와 설정을 초기화한다.
    def __init__(self, num_classes=4, width=1.0, depth=1.0, conf_thresh=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = False
        self.init_cfg = None
        self.pretrained_parameters_loaded = 0
        self.conf_threshold = conf_thresh
        self.backbone = CSPNeXt(width, depth)
        channels = int(256 * width)
        self.neck = CSPNeXtPAFPN(self.backbone.out_channels, channels, max(round(3 * depth), 1))
        self.head = RTMDetSepBNHead(channels, num_classes)
        self.architecture_signature = {'family': 'RTMDet', 'width': width, 'depth': depth, 'backbone_blocks': self.backbone.block_counts, 'neck_blocks': self.neck.num_csp_blocks, 'channel_attention': True, 'head': 'RTMDetSepBNHead', 'shared_conv': True, 'point_offset': 0, 'assignment': 'DynamicSoftLabelAssigner', 'loss': 'QFL+GIoU'}

    # RTMDet 검출과 임계값·NMS 처리를 수행한다.
    def forward(self, x, targets=None):
        # 1) 백본·넥·헤드에서 박스 후보 만들기
        logits, distances, priors = self.head(self.neck(self.backbone(x)))
        boxes = distance_to_bbox(priors, distances)
        if self.training:
            raise RuntimeError('Inference-only model: call eval().')
        results = []
        for b in range(len(x)):
            scores = logits[b].sigmoid()
            anchor_idx, labels = torch.where(scores >= self.conf_threshold)
            selected_scores = scores[anchor_idx, labels]
            if len(selected_scores) > 30000:
                top = selected_scores.topk(30000).indices
                anchor_idx, labels, selected_scores = (anchor_idx[top], labels[top], selected_scores[top])
            # 2) 좌표 범위를 제한하고 유효 박스 거르기
            selected_boxes = boxes[b, anchor_idx].clone()
            selected_boxes[:, 0::2].clamp_(0, x.shape[-1])
            selected_boxes[:, 1::2].clamp_(0, x.shape[-2])
            valid = (selected_boxes[:, 2:] > selected_boxes[:, :2]).all(1)
            selected_boxes, labels, selected_scores = (selected_boxes[valid], labels[valid], selected_scores[valid])
            # 3) 클래스별 NMS 후 최종 검출 선택하기
            keep = batched_nms(selected_boxes, selected_scores, labels, 0.65)[:300]
            results.append({'boxes': selected_boxes[keep], 'scores': selected_scores[keep], 'labels': labels[keep]})
        return results
