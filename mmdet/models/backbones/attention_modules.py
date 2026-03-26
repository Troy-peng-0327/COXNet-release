import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
import einops
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET


class ChannelFusion(BaseModule):
    def __init__(self, in_channels, reduction=1, poolupsample=False, weight=False):
        super(ChannelFusion, self).__init__()
        self.poolupsample = poolupsample
        self.weight = weight
        if poolupsample is True:
            self.poolupsample = PoolingUpsample(in_channels // 2)
        if weight is True:
            self.weighting_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
            self.weighting_t = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.qkv = nn.Conv2d(in_channels * 2, in_channels * 6, kernel_size=1)
        self.proj = nn.Conv2d(in_channels * 2 // reduction, in_channels, kernel_size=1)
    
    def forward(self, feat_v, feat_t):
        if self.poolupsample is True:
            v_feat = self.poolupsample(v_feat)
        if self.weight is True:
            v_feat = v_feat * self.weighting_v + v_feat
            t_feat = t_feat * self.weighting_t + t_feat
        feat_c = torch.cat([feat_v, feat_t], dim=1)
        # qkv = self.qkv(feat_c)
        # q, k, v = torch.chunk(qkv, 3, dim=1)
        query = self.query(feat_c)
        key = self.key(feat_c)
        value = self.value(feat_c)
        q_gap = self.gap(query)
        k_gmp = self.gmp(key)
        attn = torch.cat([q_gap, k_gmp], dim=1)
        attn = self.proj(attn)
        out_c = value * torch.sigmoid(attn)
        # out_c = value * F.softmax(attn, dim=-1)
        # out_c = feat_c + out_c
        out_v, out_t = torch.chunk(out_c, 2, dim=1)
        return out_v + feat_v, out_t + feat_t


class ChannelFusionV2(BaseModule):
    def __init__(self, in_channels, reduction=1, offset=False):
        super(ChannelFusion, self).__init__()
        self.offset = offset
        if offset:
            self.offset_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels * 2 // reduction, in_channels, kernel_size=1)
    
    def forward(self, feat_v, feat_t):
        feat_c = torch.cat([feat_v, feat_t], dim=1)

        if self.offset:
            offset = self.offset_conv(feat_c)
            offset = torch.tanh(offset)

            grid = F.affine_grid(offset.permute(0, 2, 3, 1), feat_v.size(), align_corners=False)
            feat_v = F.grid_sample(feat_v, grid, mode='bilinear', align_corners=False)
            feat_t = F.grid_sample(feat_t, grid, mode='bilinear', align_corners=False)
        
        feat_c = torch.cat([feat_v, feat_t], dim=1)
        query = self.query(feat_c)
        key = self.key(feat_c)
        value = self.value(feat_c)

        q_gap = self.gap(query)
        k_gmp = self.gmp(key)
        attn = torch.cat([q_gap, k_gmp], dim=1)
        attn = self.proj(attn)
        out = value * torch.sigmoid(attn) + feat_c
        out_v, out_t = torch.chunk(out, 2, dim=1)
        return out_v, out_t


class ChannelFusionV3(BaseModule):
    def __init__(self, in_channels, reduction=1, offset=False):
        super(ChannelFusion, self).__init__()
        self.offset = offset
        kernel_size = [9, 7, 5, 3]
        if offset:
            self.conv_offset = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
                LayerNormProxy(self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels * 2 // reduction, in_channels, kernel_size=1)
    
    def forward(self, feat_v, feat_t):
        feat_c = torch.cat([feat_v, feat_t], dim=1)

        if self.offset:
            offset = self.offset_conv(feat_c)
            offset = torch.tanh(offset)

            grid = F.affine_grid(offset.permute(0, 2, 3, 1), feat_v.size(), align_corners=False)
            feat_v = F.grid_sample(feat_v, grid, mode='bilinear', align_corners=False)
            feat_t = F.grid_sample(feat_t, grid, mode='bilinear', align_corners=False)
        
        feat_c = torch.cat([feat_v, feat_t], dim=1)
        query = self.query(feat_c)
        key = self.key(feat_c)
        value = self.value(feat_c)

        q_gap = self.gap(query)
        k_gmp = self.gmp(key)
        attn = torch.cat([q_gap, k_gmp], dim=1)
        attn = self.proj(attn)
        out = value * torch.sigmoid(attn) + feat_c
        out_v, out_t = torch.chunk(out, 2, dim=1)
        return out_v, out_t


class ChannelAttentionFusion(BaseModule):
    def __init__(self, in_channels, reduction=1, poolupsample=False, weight=False, num_heads=2):
        super(ChannelAttentionFusion, self).__init__()
        self.poolupsample = poolupsample
        self.weight = weight
        self.reduction = reduction
        if poolupsample:
            self.poolupsample = PoolingUpsample(in_channels // 2)
        if weight:
            self.weighting_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
            self.weighting_t = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)

        # MultiheadAttention expects inputs in the format (L, N, E)
        self.num_heads = num_heads
        self.embedding_dim = in_channels // reduction  # Assuming embedding dimension matches channel dimension
        self.reduce_dim = nn.Conv2d(in_channels // 2, self.embedding_dim // 2, 1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads)
        self.fc = nn.Linear(in_channels // reduction, in_channels)  # Fully connected layer to transform the output back to channel space

    def forward(self, feat_v, feat_t):
        batch_size, channels, height, width = feat_v.size()

        if self.poolupsample:
            feat_v = self.poolupsample(feat_v)
            feat_t = self.poolupsample(feat_t)

        if self.weight:
            feat_v = feat_v * self.weighting_v(feat_v)
            feat_t = feat_t * self.weighting_t(feat_t)
        
        if self.reduction != 1:
            feat_v = self.reduce_dim(feat_v)
            feat_t = self.reduce_dim(feat_t)

        # Reshape and transpose from (B, C, H, W) to (L, N, E) for MultiheadAttention
        feat_v = feat_v.view(batch_size, channels // self.reduction, -1).permute(2, 0, 1)  # (H*W, B, C)
        feat_t = feat_t.view(batch_size, channels // self.reduction, -1).permute(2, 0, 1)

        # Concatenate along the sequence dimension
        feat_cat = torch.cat([feat_v, feat_t], dim=2)  # (2*H*W, B, C)

        # Self-attention
        attn_output, _ = self.multihead_attn(feat_cat, feat_cat, feat_cat)
        
        # Project back to the channel space and reshape
        out = self.fc(attn_output)
        out = out.permute(1, 2, 0).view(batch_size, channels, 2, height, width)  # Reshape back to (B, C, 2, H, W)
        out = out.sum(dim=2)  # Combine the output from self-attention
        
        return out[:, :, :height, :width], out[:, :, height:, :width]


class SpatialFusion(BaseModule):
    def __init__(self, in_channels, reduction=16, poolupsample=False, weight=False):
        super(SpatialFusion, self).__init__()
        self.poolupsample = poolupsample
        self.weight = weight
        if poolupsample is True:
            self.poolupsample = PoolingUpsample(in_channels // 2)
        if weight is True:
            self.weighting_v = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, padding=0, groups=in_channels)
            self.weighting_t = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, padding=0, groups=in_channels)
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value =  nn.Conv2d(in_channels, in_channels, kernel_size=1) # 或者维持in_channels，视情况而定

    def forward(self, feat_v, feat_t):
        if self.poolupsample is True:
            v_feat = self.poolupsample(v_feat)
        if self.weight is True:
            v_feat = v_feat * self.weighting_v + v_feat
            t_feat = t_feat * self.weighting_t + t_feat
        b, _, h, w = feat_v.shape
        feat_c = torch.cat([feat_v, feat_t], dim=1)
        query = self.query(feat_c).view(b, -1, h*w).permute(0, 2, 1)    # B x (H*W) x C
        key = self.key(feat_c).view(b, -1, h*w)                         # B x C x (H*W)
        value = self.value(feat_c).view(b, -1, h*w)                     # B x C x (H*W)
        attn = torch.matmul(query, key)                                 # 执行批量矩阵乘法 B x (H*W) x (H*W)
        attn = F.softmax(attn, dim=-1)                                  # 应用 softmax 于最后一个维度上
        # attn = torch.sigmoid(attn)
        out = torch.matmul(attn, value.permute(0, 2, 1))                # B x (H*W) x C
        out_c = out.permute(0, 2, 1).view(b, -1, h, w)
        # out_c = feat_c + out
        out_v, out_t = torch.chunk(out_c, 2, dim=1)
        return out_v + feat_v, out_t + feat_t


class TransformerFusionModule(nn.Module):
    def __init__(self, in_channels, weight=False):
        super(TransformerFusionModule, self).__init__()
        self.weight = weight
        if weight is True:
            self.weighting_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
            self.weighting_t = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
        self.position_embedding = nn.Parameter(torch.randn(1, 256))  # 假设 FC' 的维度是 256
        self.transformer_block = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, activation='gelu')

    def forward(self, feat_v, feat_t):
        if self.weight is True:
            v_feat = v_feat * self.weighting_v + v_feat
            t_feat = t_feat * self.weighting_t + t_feat

        return v_feat, t_feat


class MultiScaleAttention(BaseModule):
    def __init__(
            self, 
            in_channels, 
            reduction=16, 
            mode='scale2', 
            weight=False, 
            offset=False, 
            offset_kernels=[9, 7, 5, 3, 1], 
            offset_range_factor =[1, 2, 3, 4], 
            stage_id=0,
            om_kernels=[9, 7, 5, 3, 1],
            msf_kernels=[7, 5, 3]):
        super(MultiScaleAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.in_channels = in_channels
        self.reduction = reduction
        self.weight = weight
        self.mode = mode
        self.offset = offset
        if stage_id == 4:
            self.offset = False

        if weight is True:
            self.interact_conv = nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=1, 
                groups=1)
        if self.offset is True:
            self.offset_range_factor = offset_range_factor[stage_id]
            self.conv_offset = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    in_channels, 
                    om_kernels[stage_id], 
                    1, 
                    om_kernels[stage_id] // 2, 
                    groups=in_channels),
                LayerNormProxy(in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, 2, 1, 1, 0, bias=False)
            )
            self.conv_off = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        if mode == 'scale2':
            self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=7, padding=3)
        elif mode == 'scale3':
            self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(in_channels // reduction, in_channels // reduction * 3, kernel_size=1)  # 使用1x1卷积代替全连接层
        self.proj_conv = nn.Conv2d(in_channels // reduction * 3, in_channels, kernel_size=1)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        # ref = ref[None, ...].expand(B * self.reduction, -1, -1, -1)  # B * g H W 2
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, feat_v, feat_t, iter=0):
        B, C, H, W = feat_v.size()

        if H >= 128:
            # save_feature_to_img(feat_v, 'v_feat_after_pu', iter)
            draw_feature_map(feat_v, name='v_feat_after_pu', iter=iter)
        
        feat_c = self.conv(torch.cat([feat_v, feat_t], dim=1))

        if H >= 128:
            # save_feature_to_img(feat_c, 'feat_c', iter)
            draw_feature_map(feat_c, name='feat_c', iter=iter)

        if self.weight is True:
            feat_c = self.interact_conv(feat_c) + feat_c
        if self.offset is True:
            # proj_offset = einops.rearrange(feat_c, 'b (g c) h w -> (b g) c h w', g=self.reduction, c=self.in_channels // self.reduction)
            proj_offset = feat_c
            offset = self.conv_offset(proj_offset)
            Hk, Wk = offset.size(2), offset.size(3)
            n_sample = Hk * Wk

            if self.offset_range_factor > 0:
                offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=offset.device).reshape(1, 2, 1, 1)
                offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
            if H >= 128:
                # save_feature_to_img(offset, 'offset', iter)
                draw_feature_map(offset, name='offset', iter=iter)

            offset = einops.rearrange(offset, 'b c h w -> b h w c')

            vis_reference = self._get_ref_points(Hk, Wk, B, feat_v.dtype, feat_v.device)
            lwir_reference = self._get_ref_points(Hk, Wk, B, feat_v.dtype, feat_v.device)

            if self.offset_range_factor >= 0:
                vis_pos = vis_reference + offset
                lwir_pos = lwir_reference
            else:
                vis_pos = (vis_reference + offset).tanh()
                lwir_pos = lwir_reference.tanh()

            feat_v = F.grid_sample(
                input=feat_v,
                grid=vis_pos[..., (1, 0)],  
                mode='bilinear', align_corners=True)  

            feat_t = F.grid_sample(
                input=feat_t,
                grid=lwir_pos[..., (1, 0)],  
                mode='bilinear', align_corners=True)
            
            feat_c = self.conv_off(torch.cat([feat_v, feat_t], dim=1))

            if H >= 128:
                # save_feature_to_img(feat_c, 'after_offset', iter)
                draw_feature_map(feat_c, name='after_offset', iter=iter)

        if self.mode == 'scale2' or 'scale3':
            u1 = self.conv1(feat_c)
            u2 = self.conv2(feat_c)
            u3 = self.conv3(feat_c)
            u = u1 + u2 + u3
            wc = self.relu(self.gap(u))
            ws = self.fc(wc)
            z1, z2, z3 = torch.chunk(ws, 3, dim=1)
            a1 = u1 * F.softmax(z1, dim=1)
            a2 = u2 * F.softmax(z2, dim=1)
            a3 = u3 * F.softmax(z3, dim=1)
        a = self.proj_conv(torch.cat([a1, a2, a3], dim=1))

        if H >= 128:
            # save_feature_to_img(a, 'scale', iter)
            draw_feature_map(a, name='scale', iter=iter)

        out_c = feat_c + a

        # if H >= 128:
        #     save_feature_to_img(out_c, 'out', iter)

        return out_c

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


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')



def my_norm(x1, x2, type='minmax'):
    assert type in ['standard', 'minmax']
    bs, _ , H, W = x1.size()
    _, _, h, w = x2.size()
    x1 = x1.view(bs, -1, H*W)
    x2 = x2.view(bs, -1, h*w)
    concat = torch.cat((x1, x2), dim=2)
    if type == 'standard':
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
    elif type == 'minmax':
        x_min = torch.min(concat, dim=2, keepdim=True)[0]
        x_max = torch.max(concat, dim=2, keepdim=True)[0]
        x1 = (x1 - x_min) / x_max
        x2 = (x2 - x_min) / x_max
    x1 = x1.view(bs, -1, H, W)
    x2 = x2.view(bs, -1, h, w)
    return [x1, x2]


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
            save_path = f'work_dir/rgbtdroneperson/vis_results/feat_vis/gfl/{feat_name}_{iter_num}_batch_{i}_best_channel_{best_channel}.jpg'
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
            overlay_save_path = f'work_dir/rgbtdroneperson/vis_results/feat_vis/gfl/overlay_{feat_name}_{iter_num}_batch_{i}_best_channel_{best_channel}.jpg'
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
            
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'_{gt_name[:-4]}.png'), superimposed_img)

                superimposed_img = heatmap * 0.5 + img*0.3
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'overlay_{gt_name[:-4]}.png'), superimposed_img)
                i=i+1

    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_im