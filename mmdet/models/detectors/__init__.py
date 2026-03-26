# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDetector
from .single_stage import SingleStageDetector
from .coxnet import COXNet
from .fusionnet_xo import FusionNetXO

__all__ = [
    'BaseDetector', 'SingleStageDetector', 'COXNet', 'FusionNetXO'
]
