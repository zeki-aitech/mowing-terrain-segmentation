# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom datasets package for visual segmentation benchmark.
"""

# Import all custom datasets to register them with MMSegmentation
from . import datasets

# Re-export all datasets for convenience
from .datasets import *

__all__ = [
    'datasets',
    # All dataset names are automatically included via "from .datasets import *"
]
