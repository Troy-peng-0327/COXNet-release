# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_neck
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RetinaNetRGBT(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNetRGBT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
        self.backbone_t = build_backbone(backbone)
        self.neck_t = build_neck(neck)
        self.fuse = nn.Conv2d(neck['out_channels'] * 2, neck['out_channels'], kernel_size=3, stride=1, padding=1)

    def extract_feat(self, img):
        v_img, t_img = img
        v_feats = self.backbone(v_img)
        t_feats = self.backbone_t(t_img)
        if self.with_neck:
            v_feats = self.neck(v_feats)
            t_feats = self.neck_t(t_feats)
        features = []
        for i in range(len(v_feats)):
            feat = self.fuse(torch.cat([v_feats[i], t_feats[i]], dim=1))
            features.append(feat)
        return features