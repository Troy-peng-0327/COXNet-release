# Copyright (c) OpenMMLab. All rights reserved.
from .res_layer import ResLayer, SimplifiedBasicBlock
from .fusion_strategy import FusionLayer
from .fusion_layer_coxnet import COXFusionLayer

__all__ = [
    'ResLayer', 'SimplifiedBasicBlock', 'FusionLayer', 'COXFusionLayer'
]
