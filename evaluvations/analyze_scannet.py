import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import sys

# Import ScanNet label mappings
sys.path.append(os.path.dirname(__file__))
from sonata.scannet_labels import (
    VALID_CLASS_IDS_20, CLASS_LABELS_20,
    VALID_CLASS_IDS_200, CLASS_LABELS_200
)

# --- CONFIGURATION ---
DATA_ROOT = 'data/scannet_data/val'  # Adjust if needed
BATCH_SIZE = 4
NUM_WORKERS = 4
USE_SCANNET200 = True  # Set to True to analyze segment200 instead of segment20

# Build label mappings based on configuration
if USE_SCANNET200:
    # Map from NYU40 IDs to 0-199 contiguous labels
    VALID_IDS = VALID_CLASS_IDS_200
    LABELS = CLASS_LABELS_200
    SEGMENT_FILE = 'segment200.npy'
    
    # Create mapping: NYU40_ID -> Contiguous_ID (0-199)
    ID_TO_LABEL = {nyu_id: i for i, nyu_id in enumerate(VALID_IDS)}
    # Inverse mapping: Contiguous_ID -> Name
    LABEL_TO_NAME = {i: name for i, name in enumerate(LABELS)}
else:
    # Map from segment20 to 0-19 contiguous labels
    VALID_IDS = VALID_CLASS_IDS_20
    LABELS = CLASS_LABELS_20
    SEGMENT_FILE = 'segment20.npy'
    
    # Create mapping: NYU40_ID -> Contiguous_ID (0-19)
    ID_TO_LABEL = {nyu_id: i for i, nyu_id in enumerate(VALID_IDS)}
    # Inverse mapping: Contiguous_ID -> Name
    LABEL_TO_NAME = {i: name for i, name in enumerate(LABELS)}

# --- DATASET CLASS (Your Code) ---
class ScanNetDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (str): Path to the folder containing scene folders.
            transform (callable, optional): Transform pipeline.
        
        Returns segments remapped to contiguous 0-N labels based on USE_SCANNET200 flag.
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

            # Load labels based on configuration
            segment_path = os.path.join(scene_path, SEGMENT_FILE)
            
            if os.path.exists(segment_path):
                segment_raw = np.load(segment_path).astype(np.int64)
                
                # Remap from NYU40 IDs to contiguous 0-N labels
                segment = np.full(segment_raw.shape, -1, dtype=np.int64)  # Default to ignore
                for nyu_id, label_id in ID_TO_LABEL.items():
                    mask = (segment_raw == nyu_id)
                    segment[mask] = label_id
            else:
                segment = np.full(coord.shape[0], -1, dtype=np.int64)  # All ignore

        except FileNotFoundError as e:
            # Skip broken scenes gracefully in analysis loop
            print(f"Warning: {e}")
            return None

        return {
            "segment": segment,  # Remapped to 0-N contiguous labels
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
    print(f"Mode: {'ScanNet200 (200 classes)' if USE_SCANNET200 else 'ScanNet20 (20 classes)'}")
    print(f"Loading from: {SEGMENT_FILE}")
    print(f"Remapping to contiguous labels: 0-{len(LABEL_TO_NAME)-1}\n")
    
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
            labels = item['segment']  # Already remapped to 0-N
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
    print(f"{'ID':<4} | {'Name':<25} | {'Scene Freq':<12} | {'Pts (M)':<8} | {'%':<6} | {'Avg RGB (Approx)':<20}")
    print("-" * 85)

    # Sort by ID for clean output
    sorted_ids = sorted(LABEL_TO_NAME.keys())

    for cls_id in sorted_ids:
        name = LABEL_TO_NAME[cls_id]
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

        print(f"{cls_id:<4} | {name:<25} | {scene_freq:<4} ({scene_pct:4.1f}%) | {point_count/1e6:<8.2f} | {point_pct:5.2f}% | {color_str:<20}")
    
    print("="*85)
    print(f"Total Scenes Processed: {total_scenes_processed}")
    print(f"Recommendation: Pick classes with >10% Scene Freq for robust testing.")
    print("Suggested GT Texts (Top 5 most frequent):")
    
    # Suggest texts based on top 5 most frequent objects
    valid_objs = [(id, count) for id, count in class_scene_counts.items()]
    valid_objs.sort(key=lambda x: x[1], reverse=True)
    
    for cls_id, count in valid_objs[:10]:  # Show top 10
        print(f" - '{LABEL_TO_NAME[cls_id]}' (ID: {cls_id}, appears in {count} scenes)")

if __name__ == "__main__":
    main()