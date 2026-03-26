# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_neck
from .single_stage import SingleStageDetector
from ..utils import FusionLayer
import matplotlib.pyplot as plt
import torch
import os
import mmcv
import numpy as np
import cv2
# from mmdet.core.visualization import imshow_det_bboxes


@DETECTORS.register_module()
class FusionNetXO(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 neck_t=None,
                 reduction=16,
                 num_layers=5,
                 fs_type='cat',
                 use_om=True,
                 use_grid=True,
                 use_fixed_ga=False,
                 use_msf=True,
                 om_kernels=[9, 7, 5, 3, 1],
                 msf_kernels=[5, 3, 1],
                 wf_loss=False,
                 wf_loss_mode='kl_v1',
                 wf_loss_weight=0.1,
                 use_clfm=[],
                 clfm_mode='up_new',
                #  usepoolup=['v'],
                 usepoolup=[],
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FusionNetXO, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        self.wf_loss = wf_loss
        self.wf_loss_mode = wf_loss_mode
        if neck_t is not None:
            self.neck_t = build_neck(neck_t)
        else:
            self.neck_t = build_neck(neck)
        
        self.fuse_layer = FusionLayer(
            neck['out_channels'],
            reduction=reduction,
            num_layers=num_layers,
            fs_type=fs_type,
            use_om=use_om,
            use_grid=use_grid,
            use_fixed_ga=use_fixed_ga,
            use_msf=use_msf,
            om_kernels=om_kernels,
            msf_kernels=msf_kernels,
            wf_loss=wf_loss,
            wf_loss_mode=wf_loss_mode,
            wf_loss_weight=wf_loss_weight,
            use_clfm=use_clfm,
            clfm_mode=clfm_mode,
            usepoolup=usepoolup)
        
        # self.iter = -1
            
    def extract_feat(self, img, gt_bboxes=None, img_metas=None):
        """Directly extract features from the backbone+neck."""
        v_img, t_img = img

        # # 保存原始图像
        # with torch.no_grad():
        #     for idx, img_tensor in enumerate(v_img):
        #         save_path = f"work_dir/vis/v_image_{idx}.png"
        #         save_image(img_tensor, save_path)

        # 保存原始图像
        # with torch.no_grad():
        #     for idx, img_tensor in enumerate(t_img):
        #         save_path = f"work_dir/vis/t_image_{idx}.png"
        #         save_image(img_tensor, save_path)
        # self.iter += 1
        v_feats, t_feats = self.backbone(v_img, t_img)
        if self.with_neck:
            v_feats = self.neck(v_feats)
            t_feats = self.neck_t(t_feats)
        # draw_feature_map(v_feats[1], name='feats', iter=self.iter)
        # feats = self.fuse_layer(v_feats, t_feats, gt_bboxes, img_metas)
        
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

        if self.wf_loss:
            losses.update(dict(wf_loss=wf_loss))

        return losses

    # def show_result(self,
    #                 img,
    #                 result,
    #                 gt_bboxes=None,
    #                 gt_labels=None,
    #                 score_thr=0.3,
    #                 bbox_color={'correct': (0, 255, 0),  # 绿色
    #                             'missed': (0, 0, 255),  # 红色
    #                             'false_positive': (0, 255, 0)},  # 橙色
    #                 text_color=(72, 101, 241),
    #                 mask_color=None,
    #                 thickness=2,
    #                 font_size=13,
    #                 win_name='',
    #                 show=False,
    #                 wait_time=0,
    #                 out_file=None):
    #     """Draw `result` over `img`.

    #     Args:
    #         img (str or Tensor): The image to be displayed.
    #         result (Tensor or tuple): The results to draw over `img`
    #             bbox_result or (bbox_result, segm_result).
    #         score_thr (float, optional): Minimum score of bboxes to be shown.
    #             Default: 0.3.
    #         bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
    #            The tuple of color should be in BGR order. Default: 'green'
    #         text_color (str or tuple(int) or :obj:`Color`):Color of texts.
    #            The tuple of color should be in BGR order. Default: 'green'
    #         mask_color (None or str or tuple(int) or :obj:`Color`):
    #            Color of masks. The tuple of color should be in BGR order.
    #            Default: None
    #         thickness (int): Thickness of lines. Default: 2
    #         font_size (int): Font size of texts. Default: 13
    #         win_name (str): The window name. Default: ''
    #         wait_time (float): Value of waitKey param.
    #             Default: 0.
    #         show (bool): Whether to show the image.
    #             Default: False.
    #         out_file (str or None): The filename to write the image.
    #             Default: None.

    #     Returns:
    #         img (Tensor): Only if not `show` or `out_file`
    #     """
    #     img = mmcv.imread(img)
    #     img = img.copy()
    #     if isinstance(result, tuple):
    #         bbox_result, segm_result = result
    #         if isinstance(segm_result, tuple):
    #             segm_result = segm_result[0]  # ms rcnn
    #     else:
    #         bbox_result, segm_result = result, None
    #     bboxes = np.vstack(bbox_result)
    #     labels = [
    #         np.full(bbox.shape[0], i, dtype=np.int32)
    #         for i, bbox in enumerate(bbox_result)
    #     ]
    #     labels = np.concatenate(labels)
    #     # draw segmentation masks
    #     segms = None
    #     if segm_result is not None and len(labels) > 0:  # non empty
    #         segms = mmcv.concat_list(segm_result)
    #         if isinstance(segms[0], torch.Tensor):
    #             segms = torch.stack(segms, dim=0).detach().cpu().numpy()
    #         else:
    #             segms = np.stack(segms, axis=0)
    #     # if out_file specified, do not show image in window
    #     if out_file is not None:
    #         show = False

    #     # 创建标记正确、漏检和误检的逻辑
    #     bbox_colors_list = []
    #     if gt_bboxes is not None and gt_labels is not None:
    #         from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

    #         # 计算 IoU
    #         ious = bbox_overlaps(bboxes, gt_bboxes)
    #         matched_gt = ious.argmax(axis=1)  # 每个预测框匹配的真值框索引
    #         max_ious = ious.max(axis=1)  # 每个预测框的最大 IoU
            
    #         # 遍历预测框，分配颜色
    #         for idx, (iou, pred_label) in enumerate(zip(max_ious, labels)):
    #             if iou >= 0.5 and pred_label == gt_labels[matched_gt[idx]]:
    #                 bbox_colors_list.append(bbox_color['correct'])  # 正确检测
    #             elif iou < 0.5:
    #                 bbox_colors_list.append(bbox_color['false_positive'])  # 误检
    #         for gt_idx, gt_bbox in enumerate(gt_bboxes):
    #             if not any(matched_gt == gt_idx):  # 没有任何预测框匹配到该真值框
    #                 bbox_colors_list.append(bbox_color['missed'])  # 漏检
    #                 # bboxes = np.vstack([bboxes, gt_bbox])  # 添加漏检的框
    #                 # labels = np.append(labels, gt_labels[gt_idx])
    #     else:
    #         bbox_colors_list = [bbox_color] * len(bboxes)  # 默认颜色
        
    #     # draw bounding boxes
    #     img = imshow_det_bboxes(
    #         img,
    #         bboxes,
    #         labels,
    #         class_names=self.CLASSES,
    #         score_thr=score_thr,
    #         bbox_color=bbox_colors_list,
    #         thickness=3,
    #         font_size=2,
    #         out_file=out_file)

    #     if not (show or out_file):
    #         return img


def save_image(img_tensor, save_path):
    """
    保存原始图像
    :param img_tensor: 输入图像 (C, H, W)，torch.Tensor 格式
    :param save_path: 保存路径
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        raise TypeError("Input must be a torch.Tensor")
    
    # 转换为 H, W, C 格式并归一化
    img = img.transpose(1, 2, 0)  # 转为 H, W, C 格式
    img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]

    # 保存为图像
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def imshow_det_bboxes(img,
                     bboxes,
                     labels,
                     class_names,
                     score_thr=0.0,
                     bbox_color=None,
                     thickness=2,
                     font_size=10,
                     show_text=False,  # 新增参数，用于控制是否显示文字
                     out_file=None):
    """
    Draw bounding boxes on the input image with optional labels.

    Args:
        img (str | ndarray): Image file path or loaded image as a NumPy array.
        bboxes (ndarray): Array of shape (n, 5), where each row is 
                          (x1, y1, x2, y2, score).
        labels (ndarray): Array of shape (n,), containing class indices for each bbox.
        class_names (list[str]): List of class names.
        score_thr (float): Minimum score threshold to display bboxes. Default: 0.0.
        bbox_color (list[tuple[int]]): List of colors (R, G, B) for each class or bbox.
                                       Default: None (use random colors).
        thickness (int): Thickness of the bounding box lines. Default: 2.
        font_size (int): Font size for text. Default: 10.
        show_text (bool): Whether to display text on the bounding boxes. Default: True.
        out_file (str | None): Path to save the output image. If None, displays the image.
                               Default: None.

    Returns:
        ndarray: Image with bounding boxes drawn on it.
    """
    # Load the image if it's a file path
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()

    # Filter bboxes by score threshold
    valid_inds = bboxes[:, -1] >= score_thr
    bboxes = bboxes[valid_inds]
    labels = labels[valid_inds]

    # Assign random colors if bbox_color is not provided
    if bbox_color is None:
        rng = np.random.default_rng()
        bbox_color = [tuple(rng.integers(0, 256, size=3).tolist()) for _ in range(len(class_names))]

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2, score = bbox

        # Get the color for the current bbox
        color = bbox_color[i] if isinstance(bbox_color, list) else bbox_color

        # Draw the bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=thickness)

        if show_text:  # Only draw text if show_text is True
            label_text = f'{class_names[label]}: {score:.2f}'

            # Compute text size and position
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_size / 10, thickness)
            text_width, text_height = text_size[0]
            text_origin = (int(x1), int(y1) - text_height if y1 > text_height + 2 else int(y1) + text_height + 2)

            # Draw background rectangle for text
            cv2.rectangle(img,
                          (text_origin[0], text_origin[1] - text_height - 2),
                          (text_origin[0] + text_width, text_origin[1] + 2),
                          color=color, thickness=-1)

            # Draw text
            cv2.putText(img, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                        font_size / 10, (255, 255, 255), thickness)

    # Save or display the image
    if out_file:
        cv2.imwrite(out_file, img)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return img


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



def draw_feature_map(features,save_dir = 'work_dir/rgbtdroneperson/vis_results/wavelet',name = None, iter=0):
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