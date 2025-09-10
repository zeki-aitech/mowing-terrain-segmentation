import os
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class YCORDataset(BaseSegDataset):
    """YCOR dataset for off-road navigation segmentation.
    
    This dataset contains 1,076 images (931 train + 145 valid) with 9 semantic classes
    for off-road navigation. Each sample contains:
    - rgb.jpg: RGB image (1024x544)
    - labels.png: Segmentation mask with indexed color palette (1024x544)
    
    Classes:
        0: Background
        1: Smooth trail  
        2: Traversable grass
        3: Rough trail
        4: Puddle
        5: Obstacle
        6: Non-traversable vegetation
        7: High vegetation
        8: Sky
    """
    
    METAINFO = dict(
        classes=('background', 'smooth_trail', 'traversable_grass', 'rough_trail',
                'puddle', 'obstacle', 'non_traversable_vegetation', 'high_vegetation', 'sky'),
        palette=[[255, 255, 255], [178, 176, 153], [128, 255, 0], [156, 76, 30],
                [255, 0, 128], [255, 0, 0], [0, 160, 0], [40, 80, 0], [1, 88, 255]]
    )
    
    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    
    def load_data_list(self):
        """Load YCOR data structure: data/ycor/{train,valid}/iid*/{rgb.jpg, labels.png}
        
        Returns:
            list: List of data samples, each containing:
                - img_path: Path to RGB image
                - seg_map_path: Path to segmentation mask
                - split: 'train' or 'valid'
                - sample_id: Sample identifier (e.g., 'iid000000')
        """
        data_list = []
        data_root = self.data_root
        
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root directory not found: {data_root}")
        
        # Determine which split to load based on data_prefix
        if hasattr(self, 'data_prefix') and 'img_path' in self.data_prefix:
            splits_to_load = [self.data_prefix['img_path']]
        else:
            # Default: load both splits (for backward compatibility)
            splits_to_load = ['train', 'valid']
        
        # Load from specified split(s)
        for split in splits_to_load:
            split_path = os.path.join(data_root, split)
            if os.path.exists(split_path):
                # Get all sample directories that start with 'iid'
                sample_dirs = [d for d in os.listdir(split_path) 
                             if d.startswith('iid') and os.path.isdir(os.path.join(split_path, d))]
                sample_dirs.sort()  # Sort for consistent ordering
                
                for sample_dir in sample_dirs:
                    sample_path = os.path.join(split_path, sample_dir)
                    rgb_file = os.path.join(sample_path, 'rgb.jpg')
                    labels_file = os.path.join(sample_path, 'labels.png')
                    
                    # Check if both required files exist
                    if os.path.exists(rgb_file) and os.path.exists(labels_file):
                        data_list.append({
                            'img_path': rgb_file,
                            'seg_map_path': labels_file,
                            'seg_fields': [],  # Required by MMSegmentation
                            'sample_idx': len(data_list),  # Required by MMSegmentation
                            'split': split,  # Custom field for your dataset
                            'sample_id': sample_dir  # Custom field for your dataset
                        })
                    else:
                        # Log missing files for debugging
                        missing = []
                        if not os.path.exists(rgb_file):
                            missing.append('rgb.jpg')
                        if not os.path.exists(labels_file):
                            missing.append('labels.png')
                        print(f"Warning: Missing files in {sample_path}: {missing}")
        
        if not data_list:
            raise ValueError(f"No valid data samples found in {data_root}")
        
        print(f"Loaded {len(data_list)} samples from YCOR dataset")
        return data_list
    
    def _get_default_metainfo(self):
        """Get default metainfo for YCOR dataset"""
        return self.METAINFO
