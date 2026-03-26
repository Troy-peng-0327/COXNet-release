# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector, build_backbone
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn


@DETECTORS.register_module()
class KnowledgeDistillationHFSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 backbone_t=None,
                 teacher_is_qf=True,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        if backbone_t is not None:
            self.backbone_t = build_backbone(backbone_t)
        else:
            self.backbone_t = build_backbone(backbone)
        self.fuse = Fusion_strategy(neck['out_channels'])
        self.teacher_is_qf = teacher_is_qf
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.quality_attention = True
        self.poolupsample = None
        self.base_fusion = 'cat'

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # self.iter = self.iter + 1
        v_img, t_img = img
        v_feat = self.backbone(v_img)
        t_feat = self.backbone_t(t_img)
        if self.with_neck:
            v_feat = self.neck(v_feat)
            t_feat = self.neck(t_feat)
        features = []
        for i in range(len(v_feat)):
            feat = self.fuse(v_feat[i], t_feat[i], 'cat')
            features.append(feat)
        return features

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        with torch.no_grad():
            if self.teacher_is_qf:
                teacher_xs = self.teacher_model.extract_feat(img)
                quality_preds_t, quality_preds_v = self.teacher_model.bbox_prehead.forward_test(teacher_xs)
                teacher_x = self.qce_fusion(teacher_xs, quality_preds_t, quality_preds_v)
                out_teacher = self.teacher_model.bbox_head(teacher_x)
            else:
                teacher_x = self.teacher_model.extract_feat(img)
                out_teacher = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        return losses

    def qce_fusion(self, x, quality_t, quality_v):
        x_vs, x_ts = x

        num_level = len(x_vs)

        fused_x = []
        for i in range(num_level):
            x_t = x_ts[i]
            x_v = x_vs[i]
            if self.quality_attention:
                quality_pred_t = torch.max(quality_t[i], dim=1, keepdim=True)[0]
                quality_pred_v = torch.max(quality_v[i], dim=1, keepdim=True)[0]

                quality_pred_t, quality_pred_v = my_norm(quality_pred_t, quality_pred_v, type='minmax')
                
                x_t = (1 + quality_pred_t) * x_t
                x_v = (1 + quality_pred_v) * x_v            
            
            if self.poolupsample is not None and i < num_level-1:
                x_v = self.poolupsample(x_v)
                # x_t = self.poolupsample(x_t)

            fused_x_ = self.fuse(x_t, x_v, self.base_fusion)

            fused_x.append(fused_x_)

        return fused_x

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp

class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp

class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp

class Fusion_CAT(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2*in_channels, in_channels, 1)
    
    def forward(self, en_ir, en_vi):
        temp = torch.cat((en_ir, en_vi), 1)
        temp = self.conv1x1(temp)
        return temp

class Fusion_GATED(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.t_conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.v_conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, t_feat, v_feat):
        t_gate = self.sigmoid(self.t_conv1x1(t_feat))
        v_gate = self.sigmoid(self.v_conv1x1(v_feat))
        return t_gate*t_feat + v_gate*v_feat

EPSILON = 1e-10

class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f

# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_strategy, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_cat = Fusion_CAT(in_channels=in_channels)
        self.fusion_cat2 = Fusion_CAT(in_channels=in_channels)
        self.fusion_spa = Fusion_SPA()
        self.fusion_gated = Fusion_GATED(in_channels=in_channels)

    def forward(self, v_feat, t_feat, fs_type):
        self.fs_type = fs_type
        if self.fs_type == 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type == 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type == 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type == 'cat':
            fusion_operation = self.fusion_cat
        elif self.fs_type == 'cat2':
            fusion_operation = self.fusion_cat2
        elif self.fs_type == 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type == 'gated':
            fusion_operation = self.fusion_gated
        if isinstance(v_feat, tuple) or isinstance(v_feat, list):
            fused_feat = []
            for i in range(len(v_feat)):
                fused_feat.append(fusion_operation(v_feat[i], t_feat[i]))
        else:
            fused_feat = fusion_operation(v_feat, t_feat)
        
        return fused_feat
    
def my_norm(x1, x2, type='standard'):
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