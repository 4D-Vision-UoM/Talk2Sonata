import os
import glob
import numpy as np
from torch.utils.data import Dataset

class ScanNetDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (str): Path to the folder containing scene folders 
                             (e.g., 'data/scannet_processed/val').
            transform (callable, optional): Sonata transform pipeline.
        """
        self.data_root = data_root
        self.transform = transform
        
        # specific to your directory structure: data_root/sceneXXXX_XX/*.npy
        # We search for all folders inside data_root
        self.scene_paths = sorted(glob.glob(os.path.join(data_root, "scene*")))
        
        if len(self.scene_paths) == 0:
            raise ValueError(f"No scene folders found in {data_root}. Check your path.")

        print(f"Loaded {len(self.scene_paths)} scenes from {data_root}")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        scene_name = os.path.basename(scene_path)

        # Load the specific .npy files you identified
        try:
            coord = np.load(os.path.join(scene_path, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(scene_path, "color.npy")).astype(np.float32)
            normal = np.load(os.path.join(scene_path, "normal.npy")).astype(np.float32)
            
            # Load labels if they exist (usually for train/val)
            segment_path = os.path.join(scene_path, "segment20.npy")
            instance_path = os.path.join(scene_path, "instance.npy")
            
            if os.path.exists(segment_path):
                segment = np.load(segment_path).astype(np.int64)
            else:
                segment = np.zeros(coord.shape[0], dtype=np.int64) - 1 # Ignore index

            if os.path.exists(instance_path):
                instance = np.load(instance_path).astype(np.int64)
            else:
                instance = np.zeros(coord.shape[0], dtype=np.int64) - 1

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy file in {scene_name}: {e}")

        # Construct the dictionary expected by Sonata/Pointcept
        data_dict = {
            "coord": coord,
            "color": color,
            "normal": normal,
            "segment20": segment,  
            "instance": instance,
            "name": scene_name,
            "id": idx
        }

        # Apply Sonata transforms (grid sampling, normalization, etc.)
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict