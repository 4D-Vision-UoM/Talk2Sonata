import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
DATA_ROOT = 'data/scannet_data/train'  # Adjust if needed
BATCH_SIZE = 4
NUM_WORKERS = 4

# Standard ScanNet v2 20-class mapping
# These map the 'segment20' integers to text descriptions
CLASS_NAMES = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "door",
    8: "window",
    9: "bookshelf",
    10: "picture",
    11: "counter",
    12: "desk",
    13: "curtain",
    14: "refrigerator",
    15: "shower curtain",
    16: "toilet",
    17: "sink",
    18: "bathtub",
    19: "otherfurniture"
}

# --- DATASET CLASS (Your Code) ---
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

        print(f"Dataset loaded: {len(self.scene_paths)} scenes from {data_root}")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        scene_name = os.path.basename(scene_path)

        # Load the specific .npy files you identified
        try:
            # Loading coord confirms valid scene
            coord = np.load(os.path.join(scene_path, "coord.npy")).astype(np.float32)
            
            # Load Color data
            color_path = os.path.join(scene_path, "color.npy")
            if os.path.exists(color_path):
                color = np.load(color_path).astype(np.float32)
            else:
                # Default to black if missing, though unlikely in ScanNet
                color = np.zeros_like(coord)

            # Load labels if they exist (usually for train/val)
            segment_path = os.path.join(scene_path, "segment20.npy")
            
            if os.path.exists(segment_path):
                segment = np.load(segment_path).astype(np.int64)
            else:
                segment = np.zeros(coord.shape[0], dtype=np.int64) - 1 # Ignore index

        except FileNotFoundError as e:
            # Skip broken scenes gracefully in analysis loop
            print(f"Warning: {e}")
            return None

        return {
            "segment20": segment,
            "color": color,
            "name": scene_name
        }

def collate_fn(batch):
    # Custom collate to filter out None returns from broken files
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch

def main():
    print(f"--- Analyzing ScanNet Labels & Colors in {DATA_ROOT} ---")
    
    dataset = ScanNetDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Statistics containers
    class_point_counts = defaultdict(int)
    class_scene_counts = defaultdict(int)
    class_color_sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))  # Store RGB sums
    
    total_points = 0
    total_scenes_processed = 0

    print("Iterating through scenes...")
    for batch in tqdm(dataloader):
        if batch is None: continue
        
        for item in batch:
            total_scenes_processed += 1
            labels = item['segment20']
            colors = item['color']  # (N, 3) - usually 0-255 or 0-1
            
            # Filter out ignore labels (-1) if any
            valid_mask = labels != -1
            valid_labels = labels[valid_mask]
            valid_colors = colors[valid_mask]
            
            if len(valid_labels) == 0:
                continue

            # 1. Count points per class
            unique_classes, counts = np.unique(valid_labels, return_counts=True)
            for cls_id, count in zip(unique_classes, counts):
                class_point_counts[cls_id] += count
                
                # Accumulate colors for average calc later
                # Create mask for this specific class
                cls_mask = (valid_labels == cls_id)
                # Sum RGB values for all points of this class in this scene
                class_color_sums[cls_id] += valid_colors[cls_mask].sum(axis=0)
                
            # 2. Count scenes per class (presence/absence)
            for cls_id in unique_classes:
                class_scene_counts[cls_id] += 1
                
            total_points += len(valid_labels)

    # --- PRINT REPORT ---
    print("\n" + "="*85)
    print(f"{'ID':<4} | {'Name':<16} | {'Scene Freq':<12} | {'Pts (M)':<8} | {'%':<6} | {'Avg RGB (Approx)':<20}")
    print("-" * 85)

    # Sort by ID for clean output
    sorted_ids = sorted(CLASS_NAMES.keys())

    for cls_id in sorted_ids:
        name = CLASS_NAMES[cls_id]
        scene_freq = class_scene_counts[cls_id]
        point_count = class_point_counts[cls_id]
        
        # Calculate percentages
        scene_pct = (scene_freq / total_scenes_processed) * 100 if total_scenes_processed > 0 else 0
        point_pct = (point_count / total_points) * 100 if total_points > 0 else 0
        
        # Calculate Avg Color
        if point_count > 0:
            avg_rgb = class_color_sums[cls_id] / point_count
            # Format as integer (0-255) if the range looks like 0-255, or float if 0-1
            # Assuming standard ScanNet is usually 0-255 for raw .npy, but let's check range implicitly
            # Simple format: R: G: B:
            color_str = f"[{avg_rgb[0]:3.0f}, {avg_rgb[1]:3.0f}, {avg_rgb[2]:3.0f}]"
        else:
            color_str = "[N/A]"

        print(f"{cls_id:<4} | {name:<16} | {scene_freq:<4} ({scene_pct:4.1f}%) | {point_count/1e6:<8.2f} | {point_pct:5.2f}% | {color_str:<20}")
    
    print("="*85)
    print(f"Total Scenes Processed: {total_scenes_processed}")
    print(f"Recommendation: Pick classes with >10% Scene Freq for robust testing.")
    print("Suggested GT Texts:")
    
    # Suggest texts based on top 5 most frequent objects (excluding wall/floor)
    valid_objs = [(id, count) for id, count in class_scene_counts.items() if id not in [0, 1]]
    valid_objs.sort(key=lambda x: x[1], reverse=True)
    
    for cls_id, count in valid_objs[:5]:
        print(f" - 'a {CLASS_NAMES[cls_id]}'")

if __name__ == "__main__":
    main()