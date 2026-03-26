# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch


@DETECTORS.register_module()
class ATSSRGBT(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSSRGBT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        self.t_channels = backbone['in_channels'] - 3
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # self.iter = self.iter + 1
        v_img, t_img = img
        if self.t_channels == 1:
            t_img = t_img.mean(dim=1, keepdim=True)
        img = torch.cat([v_img, t_img], dim=1)
        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        return features
