# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import torch.nn.functional as F
import math
from ..utils import SELayer


@DETECTORS.register_module()
class GFLHCF(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 backbone_t=None,
                 neck_t=None,
                 fs_type='cat',
                 pre_fs_type='cat',
                 pre_layer=False,
                 pre_mode='backbone',
                 pre_stage='v+t',
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFLHCF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        if backbone_t is not None:
            self.backbone_t = build_backbone(backbone_t)
        else:
            self.backbone_t = build_backbone(backbone)
        if neck_t is not None:
            self.neck_t = build_backbone(neck_t)
        else:
            self.neck_t = build_backbone(neck)
        
        self.fuse_layer = FusionLayer(neck['out_channels'], fs_type=fs_type)

        self.pre_fuse_layer = FusionLayer(neck['out_channels'], fs_type=pre_fs_type)

        self.pre_layer = pre_layer
        self.pre_mode = pre_mode
        self.pre_stage = pre_stage

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # self.iter = self.iter + 1
        v_img, t_img = img
        v_feats_b = self.backbone(v_img)
        t_feats_b = self.backbone_t(t_img)
        if self.with_neck:
            v_feats = self.neck(v_feats_b)
            t_feats = self.neck_t(t_feats_b)
        fused_feats = self.fuse_layer(v_feats, t_feats)
        if self.pre_layer is True:
            if self.pre_mode == 'backbone':
                v_feats = self.backbone(v_feats, fused_feats)
                t_feats = self.backbone_t(t_feats, fused_feats)
                pre_fused_feats = self.fuse_layer(v_feats, t_feats)
                fused_feats = [fused_feats[0]]
                for feat in pre_fused_feats:
                    fused_feats.append(feat)
            if self.pre_mode == 'neck':
                if self.pre_stage == 'v+t':
                    v_feats = self.neck(v_feats_b, fused_feats)
                    t_feats = self.neck_t(t_feats_b, fused_feats)
                    fused_feats = self.fuse_layer(v_feats, t_feats)
                if self.pre_stage == 'v':
                    v_feats = self.neck(v_feats_b, fused_feats)
                    fused_feats = self.fuse_layer(v_feats, t_feats)
                if self.pre_stage == 't':
                    t_feats = self.neck_t(t_feats_b, fused_feats)
                    fused_feats = self.fuse_layer(v_feats, t_feats)
        return fused_feats


class FusionLayer(torch.nn.Module):
    def __init__(self, in_channels, fs_type='cat', scale_factor=2, num_layuers=5):
        super(FusionLayer, self).__init__()
        self.fs_type = fs_type
        self.in_channels = in_channels
        self.poolupsample = PoolingUpsample(in_channels)
        if fs_type == 'cat' or fs_type == 'upcat':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        elif fs_type == 'add':
            assert in_channels % 2 == 0, "in_channels should be divisible by 2 for add fusion type"
        elif fs_type == 'guide_v_fusion' or fs_type == 'guide_t_fusion':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        elif fs_type == 'tour_fusion' or fs_type == 'tour_v_fusion' or fs_type == 'tour_t_fusion':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif fs_type == 'se_fusion' or fs_type == 'se_v_fusion' or fs_type == 'se_t_fusion':
            self.se_layer = SELayer(in_channels)
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        elif fs_type == 'half_v_fusion' or fs_type == 'half_t_fusion':
            self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.fuse_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        elif fs_type == 'tour_se_fusion' or fs_type == 'tour_se_v_fusion' or fs_type == 'tour_se_t_fusion':
            self.se_layer = SELayer(in_channels)
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif fs_type == 'weight_fusion' or fs_type == 'weight_v_fusion' or fs_type == 'weight_t_fusion':
            self.conv_v_layers = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(num_layuers)])
            self.conv_t_layers = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(num_layuers)])
        elif fs_type == 'self_attention':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            self.attention = SelfAttention(in_channels)
        elif fs_type == 'cross_attention_v' or fs_type == 'cross_attention_t':
            self.cross_attention = CrossAttention(in_channels)
        elif fs_type == 'channel_attention':
            self.channel_attention = ChannelAttention(in_channels=in_channels * 2, reduction=16)
            self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        else:
            raise ValueError(f"Unsupported fusion type: {fs_type}")

    def forward(self, v_feats, t_feats):
        fused_feats = []
        if self.fs_type == 'cat':
            for i in range(len(v_feats)):
                fused_feat = torch.cat([v_feats[i], t_feats[i]], dim=1)
                fused_feats.append(self.conv(fused_feat))
        if self.fs_type == 'upcat':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.poolupsample(v_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'add':
            for i in range(len(v_feats)):
                fused_feat = v_feats[i] + t_feats[i]
                fused_feats.append(fused_feat)
        elif self.fs_type == 'guide_v_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.max_pool(v_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'guide_t_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                t_feat = self.max_pool(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'tour_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.up_sample(v_feat)
                t_feat = self.up_sample(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'tour_v_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.up_sample(v_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'tour_t_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                t_feat = self.up_sample(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'se_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.se_layer(v_feat)
                t_feat = self.se_layer(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'se_v_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.se_layer(v_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'se_t_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                t_feat = self.se_layer(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'half_v_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v1, v2 = torch.split(v_feat, self.in_channels // 2, dim=1)
                t_feat = self.conv(torch.cat([v2, t_feat], dim=1))
                fused_feats.append(self.fuse_conv(torch.cat([v1, t_feat], dim=1)))
        elif self.fs_type == 'half_t_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                t1, t2 = torch.split(t_feat, self.in_channels // 2, dim=1)
                v_feat = self.conv(torch.cat([t2, v_feat], dim=1))
                fused_feats.append(self.fuse_conv(torch.cat([t1, v_feat], dim=1)))
        elif self.fs_type == 'tour_se_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.up_sample(v_feat)
                t_feat = self.up_sample(t_feat)
                v_feat = self.se_layer(v_feat)
                t_feat = self.se_layer(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'tour_se_v_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                v_feat = self.up_sample(v_feat)
                v_feat = self.se_layer(v_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))   
        elif self.fs_type == 'tour_se_t_fusion':
            for v_feat, t_feat in zip(v_feats, t_feats):
                t_feat = self.up_sample(t_feat)
                t_feat = self.se_layer(t_feat)
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))
        elif self.fs_type == 'weight_fusion':
            for v_feat, t_feat, conv_v, conv_t in zip(v_feats, t_feats, self.conv_v_layers, self.conv_t_layers):
                weight = F.sigmoid(conv_v(v_feat) + conv_t(t_feat))
                fused_feat = v_feat * weight + t_feat * (1 - weight)
                fused_feats.append(fused_feat)
        elif self.fs_type == 'weight_v_fusion':
            for v_feat, t_feat, conv_v, conv_t in zip(v_feats, t_feats, self.conv_v_layers, self.conv_t_layers):
                weight = F.sigmoid(conv_t(t_feat))
                fused_feat = v_feat * weight + t_feat * (1 - weight)
                fused_feats.append(fused_feat)
        elif self.fs_type == 'weight_t_fusion':
            for v_feat, t_feat, conv_v, conv_t in zip(v_feats, t_feats, self.conv_v_layers, self.conv_t_layers):
                weight = F.sigmoid(conv_t(t_feat))
                fused_feat = v_feat * weight + t_feat * (1 - weight)
                fused_feats.append(fused_feat)
        elif self.fs_type == 'self_attention':
            for v_feat, t_feat in zip(v_feats, t_feats):
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.attention(self.conv(fused_feat)))
        elif self.fs_type == 'cross_attention_v': # v作为query 
            for v_feat, t_feat in zip(v_feats, t_feats):
                fused_feats.append(self.cross_attention(v_feat, t_feat))
        elif self.fs_type == 'cross_attention_t': # t作为query
            for v_feat, t_feat in zip(v_feats, t_feats):
                fused_feats.append(self.cross_attention(t_feat, v_feat))
        elif self.fs_type == 'channel_attention':
            for v_feat, t_feat in zip(v_feats, t_feats):
                combined_feat = torch.cat([v_feat, t_feat], dim=1)
                attn_weights = self.channel_attention(combined_feat)
                fused_feats.append(self.conv(combined_feat * attn_weights))
        # TODO 写一个objectness的，
        return fused_feats


class CrossSELayer(BaseModule):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(CrossSELayer, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x, y):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return y * out


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class CrossAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm_x = nn.GroupNorm(norm_groups, in_channel)
        self.norm_y = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x, y):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm_x = self.norm_x(x)
        norm_y = self.norm_y(y)

        q = self.q(norm_x).view(batch, n_head, head_dim, height, width)
        kv = self.kv(norm_y).view(batch, n_head, head_dim * 2, height, width)
        key, value = kv.chunk(2, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", q, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + y


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    

class PoolingUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.maxpooling = nn.MaxPool2d(2, 2, dilation=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1)
    
    def forward(self, x):
        x_ = self.maxpooling(x)
        # x_ = self.upsample(x_)
        x_ = F.interpolate(x_, mode='bilinear', size=x.shape[-2:], align_corners=True)
        # import pdb; pdb.set_trace()
        x = self.conv1x1(torch.cat((x, x_), 1))
        return x