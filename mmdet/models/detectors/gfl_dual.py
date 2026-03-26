# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_neck
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import torch.nn.functional as F
import math
from ..utils import SELayer
from mmdet.models.backbones.attention_modules import MultiScaleAttention
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET


@DETECTORS.register_module()
class GFLDual(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 neck_t=None,
                 fs_type='cat',
                 usepoolup=False,
                 scale_mode='scale2',
                 om_kernels=[9, 7, 5, 3, 1],
                 msf_kernels=[5, 3, 1],
                 scale_weight=False,
                 backbone_fuse=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFLDual, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        self.backbone_fuse = backbone_fuse
        self.iter = -1
        if backbone_fuse is not True:
            if neck_t is not None:
                self.neck_t = build_neck(neck_t)
            else:
                self.neck_t = build_neck(neck)
            
            self.fuse_layer = FusionLayer(
                neck['out_channels'], 
                fs_type=fs_type, 
                usepoolup=usepoolup, 
                scale_mode=scale_mode, 
                scale_weight=scale_weight,
                om_kernels=om_kernels,
                msf_kernels=msf_kernels)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        self.iter = self.iter + 1
        v_img, t_img = img
        if self.backbone_fuse is True:
            feats = self.backbone(v_img, t_img)
            if self.with_neck:
                feats = self.neck(feats)
            return feats
        else:
            v_feats_b, t_feats_b = self.backbone(v_img, t_img)
            if self.with_neck:
                v_feats = self.neck(v_feats_b)
                t_feats = self.neck_t(t_feats_b)
            # save_feature_to_img(v_feats, 'v_feat_before_pu', self.iter)
            # draw_feature_map(v_feats, name='v_feat_before_pu', iter=self.iter)
            fused_feats = self.fuse_layer(v_feats, t_feats, iter=self.iter)
            # save_feature_to_img(fused_feats, 'fuse_feat', self.iter)
            # draw_feature_map(fused_feats, name='fuse_feat', iter=self.iter)
            return fused_feats


class FusionLayer(torch.nn.Module):
    def __init__(
            self, 
            in_channels, 
            fs_type='cat', 
            scale_factor=2, 
            num_layers=5, 
            usepoolup=False, 
            scale_mode='scale2', 
            scale_weight=False,
            om_kernels=[9, 7, 5, 3, 1],
            msf_kernels=[7, 5, 3]):
        super(FusionLayer, self).__init__()
        self.fs_type = fs_type
        self.in_channels = in_channels
        self.usepoolup = usepoolup
        if self.usepoolup:
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
            self.conv_v_layers = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(num_layers)])
            self.conv_t_layers = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(num_layers)])
        elif fs_type == 'self_attention':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            self.attention = SelfAttention(in_channels)
        elif fs_type == 'cross_attention_v' or fs_type == 'cross_attention_t':
            self.cross_attention = CrossAttention(in_channels)
        elif fs_type == 'channel_attention':
            self.channel_attention = ChannelAttention(in_channels=in_channels * 2, reduction=16)
            self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        elif fs_type == 'scale_attention':
            self.scale_attention = MultiScaleAttention(in_channels=in_channels, reduction=16, mode=scale_mode, weight=scale_weight)
        elif fs_type == 'tour_v_scale_attention' or fs_type == 'tour_t_scale_attention':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.scale_attention = MultiScaleAttention(in_channels=in_channels, reduction=16, mode=scale_mode, weight=scale_weight)
        elif fs_type == 'offset_scale_attention':
            self.scale_attention = nn.ModuleList()
            for i in range(num_layers):
                self.scale_attention.append(
                    MultiScaleAttention(
                        in_channels=in_channels, 
                        reduction=16, 
                        mode=scale_mode, 
                        offset=True, 
                        weight=scale_weight, 
                        stage_id=i,
                        om_kernels=om_kernels,
                        msf_kernels=msf_kernels))
        elif fs_type == 'tour_v_offset_scale_attention' or fs_type == 'tour_t_offset_scale_attention':
            self.scale_attention = nn.ModuleList()
            for i in range(num_layers):
                self.scale_attention.append(
                    MultiScaleAttention(
                        in_channels=in_channels, 
                        reduction=16, 
                        mode=scale_mode, 
                        offset=True, 
                        weight=scale_weight, 
                        stage_id=i,
                        om_kernels=om_kernels,
                        msf_kernels=msf_kernels))
            # self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif fs_type == 'tour_v_pixel_offset_scale_attention':
            self.scale_attention = nn.ModuleList()
            for i in range(num_layers):
                self.scale_attention.append(MultiScaleAttention(in_channels=in_channels, reduction=16, mode=scale_mode, offset=True, weight=scale_weight, stage_id=i))
            self.pixel_upsample = PixelShuffleNet(in_channels=in_channels, upscale_factor=2)
        else:
            raise ValueError(f"Unsupported fusion type: {fs_type}")

    def forward(self, v_feats, t_feats, iter=0):
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
        elif self.fs_type == 'scale_attention':
            for v_feat, t_feat in zip(v_feats, t_feats):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                fused_feats.append(self.scale_attention(v_feat, t_feat))
        elif self.fs_type == 'tour_t_scale_attention':
            for v_feat, t_feat in zip(v_feats, t_feats):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                t_feat = self.up_sample(t_feat)
                fused_feats.append(self.scale_attention(v_feat, t_feat))
        elif self.fs_type == 'tour_v_scale_attention':
            for v_feat, t_feat in zip(v_feats, t_feats):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                v_feat = self.up_sample(v_feat)
                fused_feats.append(self.scale_attention(v_feat, t_feat))
        elif self.fs_type == 'offset_scale_attention':
            for v_feat, t_feat, scale_attention in zip(v_feats, t_feats, self.scale_attention):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                fused_feats.append(scale_attention(v_feat, t_feat))
        elif self.fs_type == 'tour_v_offset_scale_attention':
            for v_feat, t_feat, scale_attention in zip(v_feats, t_feats, self.scale_attention):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                # v_feat = self.up_sample(v_feat)
                v_feat = F.interpolate(v_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=True)
                fused_feats.append(scale_attention(v_feat, t_feat, iter=iter))
        elif self.fs_type == 'tour_t_offset_scale_attention':
            for v_feat, t_feat, scale_attention in zip(v_feats, t_feats, self.scale_attention):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                t_feat = self.up_sample(t_feat)
                fused_feats.append(scale_attention(v_feat, t_feat))
        elif self.fs_type == 'tour_v_pixel_offset_scale_attention':
            for v_feat, t_feat, scale_attention in zip(v_feats, t_feats, self.scale_attention):
                if self.usepoolup:
                    v_feat = self.poolupsample(v_feat)
                v_feat = self.pixel_upsample(v_feat)
                fused_feats.append(scale_attention(v_feat, t_feat))
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


class PixelShuffleNet(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super(PixelShuffleNet, self).__init__()
        # 四倍于目标通道数，因为上采样系数为2（2*2）
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        _x = F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x + _x


def parse_gt_from_xml(xml_path):
    """
    从 XML 文件中解析出 gt 边界框，返回归一化的 [xmin, ymin, xmax, ymax]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_bboxes = []
    
    # 遍历所有边界框对象
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 添加边界框 (xmin, ymin, xmax, ymax) 已经归一化到 0-1 范围
        gt_bboxes.append([xmin, ymin, xmax, ymax])
    
    return gt_bboxes

def calculate_similarity_with_gt(feat_map, gt_bboxes, height, width):
    """
    计算特征图与所有 GT 目标的 mask 之间的相关性。
    feat_map: 特征图
    gt_bboxes: GT 边界框列表，每个边界框为 [xmin, ymin, xmax, ymax] 格式
    height: 特征图高度
    width: 特征图宽度
    """
    # 将特征图归一化到 [0, 1] 之间
    feat_map_normalized = (feat_map - np.min(feat_map)) / (np.max(feat_map) - np.min(feat_map) + 1e-6)

    # 创建一个与特征图相同尺寸的 mask，初始化为 0
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历所有 GT 边界框，将包含目标的区域设置为 1
    for gt_bbox in gt_bboxes:
        # 将 GT 边界框转换到特征图的尺寸
        gt_bbox_rescaled = [
            int(gt_bbox[0] * width), 
            int(gt_bbox[1] * height),
            int(gt_bbox[2] * width), 
            int(gt_bbox[3] * height)
        ]
        # 设置边界框区域为 1
        gt_mask[gt_bbox_rescaled[1]:gt_bbox_rescaled[3], gt_bbox_rescaled[0]:gt_bbox_rescaled[2]] = 1

    # 计算特征图与 mask 的相似度
    similarity = np.sum(feat_map_normalized * gt_mask) / (np.sum(gt_mask) + 1e-6)  # 防止除以 0

    return similarity


def apply_gaussian_blur(colored_feat_map, ksize=(15, 15), sigma=0):
    """
    对特征图应用高斯模糊，使其更加平滑
    """
    return cv2.GaussianBlur(colored_feat_map, ksize, sigma)

def enhance_target_color(colored_feat_map, alpha=1.2, beta=25):
    """
    增强目标位置的颜色对比度，使其更加突出，尤其是红色。
    alpha: 对比度控制（>1 增强对比度）
    beta: 亮度控制（>0 增加亮度）
    """
    # 将颜色映射到红色为主
    colored_feat_map[:, :, 2] = np.clip(colored_feat_map[:, :, 2] * 1.5, 0, 255)  # 增加红色通道的强度
    enhanced_map = cv2.convertScaleAbs(colored_feat_map, alpha=alpha, beta=beta)
    return enhanced_map

def generate_cam_like_heatmap(feat_map, gt_mask):
    """
    生成类似 CAM 的热图，增强目标位置。
    """
    heatmap = feat_map * gt_mask
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)  # 归一化到0-1
    heatmap = np.uint8(heatmap * 255)  # 转换为0-255的范围
    return heatmap

def save_feature_to_img(feats, feat_name, iter_num):
    """
    Save the feature map that best matches the GT and overlay it onto the original image.
    """
    if isinstance(feats, (tuple, list)):
        feats = feats[0]  # 取第一个特征图进行处理

    file_name_list = [
        {"id": 0, "file_name": "05120.jpg", "height": 512, "width": 640}, 
        {"id": 1, "file_name": "05563.jpg", "height": 512, "width": 640}, 
        {"id": 2, "file_name": "05569.jpg", "height": 512, "width": 640}, 
        {"id": 3, "file_name": "05650.jpg", "height": 512, "width": 640}, 
        {"id": 4, "file_name": "05661.jpg", "height": 512, "width": 640}, 
        {"id": 5, "file_name": "06006.jpg", "height": 512, "width": 640}
    ]
    gt_name = file_name_list[iter_num]['file_name']
    gt_xml_path = f'/data3/pengpeiran/datasets/RGBTDronePerson/val/annotation/{gt_name[:-4]}.xml'
    gt_bboxes = parse_gt_from_xml(gt_xml_path)

    batch_size, channels, height, width = feats.shape

    # 假设只处理 batch 的第一个图像
    for i in range(batch_size):
        best_similarity = -float('inf')  # 用于记录最佳相似度
        best_feat_map = None  # 记录最优的特征图
        best_channel = None

        # 读取原图并获取原图的宽和高
        v_img_path = f'/data3/pengpeiran/datasets/RGBTDronePerson/val/visible/{gt_name[:-4]}.jpg'
        # original_img = cv2.imread(v_img_path, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(v_img_path, cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"Warning: Image at {v_img_path} could not be loaded.")
            continue
        
        orig_height, orig_width = original_img.shape[:2]  # 获取原图的宽和高

        # 对 GT 边界框进行归一化
        gt_bboxes_normalized = []
        for gt_bbox in gt_bboxes:
            gt_bbox_normalized = [
                gt_bbox[0] / orig_width,  # xmin 归一化
                gt_bbox[1] / orig_height, # ymin 归一化
                gt_bbox[2] / orig_width,  # xmax 归一化
                gt_bbox[3] / orig_height  # ymax 归一化
            ]
            gt_bboxes_normalized.append(gt_bbox_normalized)

        for j in range(channels):
            feat_map = feats[i, j, :, :].cpu().detach().numpy()

            # 将特征图归一化到 0-255 之间
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min()) * 255
            feat_map = np.uint8(feat_map)

            # 计算特征图与归一化后的 GT 的相似性
            similarity = calculate_similarity_with_gt(feat_map, gt_bboxes_normalized, height, width)

            # 更新最佳特征图
            if similarity > best_similarity:
            # if similarity >= 0.3:
                best_similarity = similarity
                best_feat_map = feat_map
                best_channel = j

        # 只保存与 GT 最符合的特征图
        if best_feat_map is not None:
            # 生成类似 CAM 的热图，增强目标区域
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            for gt_bbox in gt_bboxes_normalized:
                # xmin, ymin, xmax, ymax = int(gt_bbox[0] * width), int(gt_bbox[1] * height), int(gt_bbox[2] * width), int(gt_bbox[3] * height)
                xmin, ymin, xmax, ymax = int(gt_bbox[0] * width * 1.05), int(gt_bbox[1] * height * 1.05), int(gt_bbox[2] * width * 1.05), int(gt_bbox[3] * height * 1.05)
                gt_mask[ymin:ymax, xmin:xmax] = 1

            cam_heatmap = generate_cam_like_heatmap(best_feat_map, gt_mask)
            colored_feat_map = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_HOT)

            # 对彩色特征图应用高斯模糊，使其更加平滑
            colored_feat_map = apply_gaussian_blur(colored_feat_map)

            # 增强目标位置的颜色，使用红色为主
            colored_feat_map = enhance_target_color(colored_feat_map)

            # 保存彩色特征图
            # save_path = f'work_dir/rgbtdroneperson/vis_results/features/ours_v/{feat_name}_{iter_num}_batch_{i}_best_channel_{best_channel}.png'
            save_path = f'work_dir/rgbtdroneperson/vis_results/feat_vis/{feat_name}_{iter_num}_batch_{i}_best_channel_{best_channel}.jpg'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, colored_feat_map)

            # 将原图调整为特征图大小
            original_img_resized = cv2.resize(original_img, (width, height))

            # 将灰度图转换为三通道以匹配彩色特征图的通道数
            # original_img_resized_color = cv2.cvtColor(original_img_resized, cv2.COLOR_GRAY2BGR)

            # 将彩色特征图叠加到三通道的原始图像上
            # overlay = cv2.addWeighted(original_img_resized_color, 0.6, colored_feat_map, 0.4, 0)
            overlay = cv2.addWeighted(original_img_resized, 0.6, colored_feat_map, 0.4, 0)

            # 保存叠加后的图像
            # overlay_save_path = f'work_dir/rgbtdroneperson/vis_results/features/ours_v/overlay_{feat_name}_{iter_num}_batch_{i}_best_channel_{best_channel}.png'
            overlay_save_path = f'work_dir/rgbtdroneperson/vis_results/feat_vis/overlay_{feat_name}_{iter_num}_batch_{i}_best_channel_{best_channel}.jpg'
            cv2.imwrite(overlay_save_path, overlay)

import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = 'work_dir/rgbtdroneperson/vis_results/feature_map',name = None, iter=0):
    i=0
    file_name_list = [
        {"id": 0, "file_name": "05120.jpg", "height": 512, "width": 640}, 
        {"id": 1, "file_name": "05563.jpg", "height": 512, "width": 640}, 
        {"id": 2, "file_name": "05569.jpg", "height": 512, "width": 640}, 
        {"id": 3, "file_name": "05650.jpg", "height": 512, "width": 640}, 
        {"id": 4, "file_name": "05661.jpg", "height": 512, "width": 640}, 
        {"id": 5, "file_name": "06006.jpg", "height": 512, "width": 640}
    ]
    gt_name = file_name_list[iter]['file_name']
    img = cv2.imread(f'/data3/pengpeiran/datasets/RGBTDronePerson/val/visible/{gt_name[:-4]}.jpg', cv2.IMREAD_COLOR)
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))  
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'_{iter}.png'), superimposed_img)
                i=i+1

    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'_{gt_name[:-4]}.png'), superimposed_img)

                superimposed_img = heatmap * 0.5 + img*0.3
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'overlay_{gt_name[:-4]}.png'), superimposed_img)
                i=i+1