# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_neck
from .single_stage import SingleStageDetector
from ..utils import COXFusionLayer


@DETECTORS.register_module()
class COXNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 neck_t=None,
                 reduction=16,
                 num_layers=5,
                 fs_type='cat',
                 use_clfm=[],
                 clfm_loss=False,
                 clfm_initialize=False,
                 clfm_use_ca=False,
                 clfm_use_sa=False,
                 use_aam=False,
                 use_grid=False,
                 aam_kernels=[9, 7, 5, 3, 1],
                 aam_dc=[False, False, False, False, False],
                 aam_range_factor=[1, 1, 1, 1, 1],
                 use_wf=[False, False, False, False, False],
                 wf_loss=['t'],
                 wf_initialize=[False, False, False, False, False],
                 wf_use_sa=[False, False, False, False, False],
                 wf_use_ca=[False, False, False, False, False],
                 use_msf=[False, False, False, False, False],
                 use_msf_dc=[False, False, False, False, False],
                 msf_kernels=[3, 3, 3],
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(COXNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        if neck_t is not None:
            self.neck_t = build_neck(neck_t)
        else:
            self.neck_t = build_neck(neck)
        
        self.num_layers = num_layers
        self.clfm_loss = clfm_loss
        self.wf_loss = wf_loss
        self.fuse_layer = COXFusionLayer(
            neck['out_channels'],
            reduction=reduction,
            num_layers=num_layers,
            fs_type=fs_type,
            use_clfm=use_clfm,
            clfm_initialize=clfm_initialize,
            clfm_use_ca=clfm_use_ca,
            clfm_use_sa=clfm_use_sa,
            use_aam=use_aam,
            use_grid=use_grid,
            aam_kernels=aam_kernels,
            aam_range_factor=aam_range_factor,
            aam_dc=aam_dc,
            use_wf=use_wf,
            wf_loss=wf_loss,
            wf_initialize=wf_initialize,
            wf_use_sa=wf_use_sa,
            wf_use_ca=wf_use_ca,
            use_msf=use_msf,
            use_msf_dc=use_msf_dc,
            msf_kernels=msf_kernels,
            )
                
    def extract_feat(self, img, gt_bboxes=None, img_metas=None):
        """Directly extract features from the backbone+neck."""
        v_img, t_img = img
        v_feats, t_feats = self.backbone(v_img, t_img)
        if self.with_neck:
            v_feats = self.neck(v_feats)
            t_feats = self.neck_t(t_feats)
        return self.fuse_layer(v_feats, t_feats, gt_bboxes, img_metas)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if self.wf_loss:
            x, wf_loss = self.extract_feat(img, gt_bboxes, img_metas)
        else:
            x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.clfm_loss:
            losses.update(dict(clfm_loss=self.fuse_layer.get_wavelet_loss(mode='clfm')))
        # if self.wf_loss:
        #     losses.update(dict(dasr_loss=self.fuse_layer.get_wavelet_loss(mode='wf3')))
        if self.wf_loss:
            losses.update(dict(wf_loss=wf_loss))

        return losses