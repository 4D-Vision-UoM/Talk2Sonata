import os
import glob
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import sonata  # Assuming your custom sonata package is available

# --- CONSTANTS ---
# Standard ScanNet v2 20-class mapping
CLASS_NAMES = {
    0: "wall", 1: "floor", 2: "cabinet", 3: "bed", 4: "chair",
    5: "sofa", 6: "table", 7: "door", 8: "window", 9: "bookshelf",
    10: "picture", 11: "counter", 12: "desk", 13: "curtain",
    14: "refrigerator", 15: "shower curtain", 16: "toilet",
    17: "sink", 18: "bathtub", 19: "otherfurniture"
}

# Classes to exclude from being Ground Truth
IGNORED_CLASS_IDS = {0, 1, 19}

# Valid candidates for fallback (if a scene is empty of objects)
VALID_CLASS_IDS = [i for i in CLASS_NAMES.keys() if i not in IGNORED_CLASS_IDS]


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
        segment = np.load(os.path.join(scene_path, "segment20.npy")).astype(np.int64)
        instance = np.load(os.path.join(scene_path, "instance.npy")).astype(np.int64)

        # --- Dynamic Ground Truth Selection (Text Logic) ---
        
        # Find all unique classes present in this scene based on the loaded segment
        unique_classes = np.unique(segment)
        
        # Filter out ignored classes (Wall, Floor, Other) and -1 (ignore index)
        candidates = [c for c in unique_classes if c not in IGNORED_CLASS_IDS and c != -1]

        if len(candidates) > 0:
           #get all candicates for later use
            target_cid = [int(c) for c in candidates]
        else:
            # STRICT MODE: Raise error if no valid candidates found.
            raise ValueError(f"Scene {scene_name} has no valid object candidates (candidates={candidates}).")

        # Get the text label
        target_text = [CLASS_NAMES[c] for c in target_cid]

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
            "target_cid": target_cid, # For mask generation later
            "text": target_text,       # For CLIP encoding
            "segment20": segment,
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
        