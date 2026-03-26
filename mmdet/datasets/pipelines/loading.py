# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImagePairFromFile(LoadImageFromFile):
    """Load RGB-Thermal image pair from two files.

    Args:
        spectrals (tuple/list): Names of folders denoting different spectrals.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self,
                 spectrals=('visible', 'thermal'),
                 to_float32=False,
                 color_type='color',
                 daynight=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.spectrals = spectrals
        self.daynight = daynight

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename1 = osp.join(results['img_prefix'], self.spectrals[0],
                                results['img_info']['filename'])
            filename2 = osp.join(results['img_prefix'], self.spectrals[1],
                                results['img_info']['filename'])
        else:
            filename1 = osp.join(self.spectrals[0], results['img_info']['filename'])
            filename2 = osp.join(self.spectrals[1], results['img_info']['filename'])

        img1_bytes = self.file_client.get(filename1)
        img2_bytes = self.file_client.get(filename2)
        img1 = mmcv.imfrombytes(img1_bytes, flag=self.color_type)
        img2 = mmcv.imfrombytes(img2_bytes, flag=self.color_type)
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
        if self.daynight:
            dn_label_ = int(results['img_info']['filename'][4])
            dn_label = (1, 0) if dn_label_ < 3 else (0, 1)
            results['dn_labels'] = dn_label

        results['filename'] = (filename1, filename2)
        results['ori_filename'] = results['img_info']['filename']
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        results['img_fields'] = ['img1', 'img2']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'spectrals={self.spectrals},'
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadYOLOImagePairFromFile(LoadImageFromFile):
    """Load RGB-Thermal image pair from YOLO-style directory structure.

    Used for VTUAV and RGBTDronePerson datasets where visible and infrared
    images are stored in separate subdirectories.

    Args:
        spectrals (tuple/list): Names of folders denoting different spectrals.
        to_float32 (bool): Whether to convert the loaded image to float32.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
        file_client_args (dict): Arguments to instantiate a FileClient.
    """
    def __init__(self,
                 spectrals=('visible', 'infrared'),
                 to_float32=False,
                 color_type='color',
                 daynight=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.spectrals = spectrals
        self.daynight = daynight

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            dir_path, file_name = osp.split(results['img_prefix'])
            filename1 = osp.join(dir_path, self.spectrals[0], 'images', file_name,
                                 results['img_info']['filename'])
            filename2 = osp.join(dir_path, self.spectrals[1], 'images', file_name,
                                 results['img_info']['filename'])
        else:
            filename1 = osp.join(self.spectrals[0], results['img_info']['filename'])
            filename2 = osp.join(self.spectrals[1], results['img_info']['filename'])

        img1_bytes = self.file_client.get(filename1)
        img2_bytes = self.file_client.get(filename2)
        img1 = mmcv.imfrombytes(img1_bytes, flag=self.color_type)
        img2 = mmcv.imfrombytes(img2_bytes, flag=self.color_type)
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
        if self.daynight:
            dn_label_ = int(results['img_info']['filename'][4])
            dn_label = (1, 0) if dn_label_ < 3 else (0, 1)
            results['dn_labels'] = dn_label

        results['filename'] = (filename1, filename2)
        results['ori_filename'] = results['img_info']['filename']
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        results['img_fields'] = ['img1', 'img2']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'spectrals={self.spectrals},'
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load bounding box and label annotations.

    Args:
        with_bbox (bool): Whether to parse and load bbox annotations. Default: True.
        with_label (bool): Whether to parse and load label annotations. Default: True.
        with_mask (bool): Whether to parse and load mask annotations. Default: False.
        with_seg (bool): Whether to parse and load semantic segmentation. Default: False.
        poly2mask (bool): Whether to convert polygon masks to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative to absolute. Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            h, w = results['img_shape'][:2]
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h
            results['gt_bboxes'] = results['gt_bboxes'].astype(np.float32)

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth boxes.
        keep_empty (bool): Whether to return None when it becomes empty. Default: True.
    """

    def __init__(self, min_gt_bbox_wh, keep_empty=True):
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            if self.keep_empty:
                return None
            else:
                return results
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh},' \
               f'keep_empty={self.keep_empty})'
