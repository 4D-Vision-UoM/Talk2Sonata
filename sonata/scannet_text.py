import os
import glob
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import sonata  # Assuming your custom sonata package is available
from sonata.scannet_labels import CLASS_LABELS_200, IGNORED_CLASS_IDS_200

# --- CONSTANTS ---
# Standard ScanNet v2 20-class mapping
CLASS_NAMES = {
    0: "wall", 1: "floor", 2: "cabinet", 3: "bed", 4: "chair",
    5: "sofa", 6: "table", 7: "door", 8: "window", 9: "bookshelf",
    10: "picture", 11: "counter", 12: "desk", 13: "curtain",
    14: "refrigerator", 15: "shower curtain", 16: "toilet",
    17: "sink", 18: "bathtub", 19: "otherfurniture"
}

# Classes to exclude from being Ground Truth (ScanNet20)
IGNORED_CLASS_IDS_20 = {0, 1, 19}  # wall, floor, otherfurniture

# Valid candidates for fallback (if a scene is empty of objects)
VALID_CLASS_IDS = [i for i in CLASS_NAMES.keys() if i not in IGNORED_CLASS_IDS_20]


class ScanNetTextDataset(Dataset):
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

        print(f"Loaded {len(self.scene_paths)} scenes for Text-Guided Training.")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        scene_name = os.path.basename(scene_path)

        # 1. Load Data strictly (No try/except, no exist checks)
        # If a file is missing, this will raise FileNotFoundError immediately.
        coord = np.load(os.path.join(scene_path, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(scene_path, "color.npy")).astype(np.float32)
        normal = np.load(os.path.join(scene_path, "normal.npy")).astype(np.float32)
        segment20 = np.load(os.path.join(scene_path, "segment20.npy")).astype(np.int64)
        instance = np.load(os.path.join(scene_path, "instance.npy")).astype(np.int64)
        
        # Load segment200 if available (already in 0-199 contiguous format)
        segment200_path = os.path.join(scene_path, "segment200.npy")
        segment200 = None
        
        if os.path.exists(segment200_path):
            segment200 = np.load(segment200_path).astype(np.int64)
            # segment200.npy already contains 0-199 contiguous IDs, no remapping needed!

        # --- Dynamic Ground Truth Selection (Text Logic) ---
        # Generate targets for BOTH segment20 and segment200 regardless of use_scannet200 flag
        
        # Segment20 targets
        unique_classes_20 = np.unique(segment20)
        candidates_20 = [c for c in unique_classes_20 if c not in IGNORED_CLASS_IDS_20 and c != -1]
        
        if len(candidates_20) > 0:
            target_cid_20 = [int(c) for c in candidates_20]
            target_text_20 = [CLASS_NAMES[c] for c in target_cid_20]
        else:
            target_cid_20 = []
            target_text_20 = []
        
        # Segment200 targets (if available)
        target_cid_200 = []
        target_text_200 = []
        if segment200 is not None:
            unique_classes_200 = np.unique(segment200)
            # segment200 already has 0-199 IDs, use them directly
            # Exclude: wall(0), floor(2), ceiling(35) based on CLASS_LABELS_200 positions
            candidates_200 = [c for c in unique_classes_200 if c not in IGNORED_CLASS_IDS_200 and c != -1 and c < len(CLASS_LABELS_200)]
            
            if len(candidates_200) > 0:
                target_cid_200 = [int(c) for c in candidates_200]
                target_text_200 = [CLASS_LABELS_200[c] for c in target_cid_200]
        else:
            raise ValueError(f"Scene {scene_name} is missing segment200.npy file. Both segment20 and segment200 are required.")
        
        # Ensure both have valid candidates
        if len(target_cid_20) == 0:
            raise ValueError(f"Scene {scene_name} has no valid object candidates in segment20 (all excluded or empty).")
        if len(target_cid_200) == 0:
            raise ValueError(f"Scene {scene_name} has no valid object candidates in segment200 (all excluded or empty).")
        
        # Use segment20 as primary
        target_cid = target_cid_20
        target_text = target_text_20

        # 2. Construct Data Dictionary (Geometric Data Only)
        # This dict goes into the transform pipeline. Keys here might be dropped/renamed.
        data_dict = {
            "coord": coord,
            "color": color,
            "normal": normal,
            # "segment20": segment, 
            "instance": instance,
        }

        # 3. Apply Sonata Transform (Voxelization, Augmentation)
        if self.transform:
            data_dict = self.transform(data_dict)

        # 4. Construct Metadata Dictionary (Protected Data)
        # These keys will bypass the transform logic entirely to prevent loss.
        meta_dict = {
            "name": scene_name,
            "id": idx,             
            "target_cid": target_cid,  # Primary target (based on use_scannet200 flag)
            "text": target_text,        # Primary text (based on use_scannet200 flag)
            "segment20": segment20,
            "segment200": segment200,  # Already 0-199 contiguous labels, None if not available
            # Store both target sets for cache
            "target_cid_segment20": target_cid_20,
            "text_segment20": target_text_20,
            "target_cid_segment200": target_cid_200 if len(target_cid_200) > 0 else None,
            "text_segment200": target_text_200 if len(target_text_200) > 0 else None,
        }

        return data_dict, meta_dict

def training_collate_fn(batch):
    """
    Collate function that handles the separated data and metadata.
    Args:
        batch: List of tuples [(data_dict, meta_dict), ...]
    Returns:
        collated_data: Sonata Point object (batched coordinates/features)
        collated_meta: List of metadata dictionaries
    """
    # Filter out invalid samples (though strictly there shouldn't be any now)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Unzip the batch into two lists
    data_batch = [item[0] for item in batch]
    meta_batch = [item[1] for item in batch]

    # Use Sonata's native collate function ONLY on the data part
    collated_data = sonata.data.collate_fn(data_batch)
    
    # Return separated tuple
    return collated_data, meta_batch

# # --- Usage Example (Verification) ---
# if __name__ == "__main__":
#     try:
#         data_root = "data/scannet_data/val" # Adjust as needed
#         print(f"Testing Dataset with root: {data_root}")
        
#         # Initialize
#         dataset = ScanNetTextDataset(data_root, transform=sonata.transform.default())
        
#         # 1. Test Single Sample
#         print(f"\n--- Sample 0 Verification ---")
#         # Returns tuple now
#         data_sample, meta_sample = dataset[0]
        
#         print(f"Data Keys (from Transform): {list(data_sample.keys())}")
#         print(f"Meta Keys (Preserved): {list(meta_sample.keys())}")
        
#         # Verify Metadata
#         print(f"Scene: {meta_sample['name']}")
#         print(f"Selected GT Text: '{meta_sample['text']}' (ID: {meta_sample['target_cid']})")
#         print(f"Coord Shape: {data_sample['coord'].shape}")
        
#         # 2. Test Collate Function
#         print(f"\n--- Collate Function Verification ---")
#         # Grab a small batch (Sample 0 and Sample 1)
#         batch_list = [dataset[0], dataset[1]]
#         print(f"Collating batch of size {len(batch_list)}...")
        
#         collated_data, collated_meta = training_collate_fn(batch_list)
        
#         print(f"Collated Data Keys: {list(collated_data.keys())}")
#         print(f"Collated Meta Size: {len(collated_meta)}")
        
#         # Check for offset (Crucial for Sparse Conv)
#         if 'offset' in collated_data:
#             print(f"Offset Tensor: {collated_data['offset']} (Shape: {collated_data['offset'].shape})")
#         else:
#             print("WARNING: 'offset' key missing! Sparse Conv requires this.")
            
#         print("\nVerification Passed!")

#     except ImportError:
#         print("Sonata package not found, skipping verification.")
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         print(f"\nVerification FAILED with error: {e}")
        