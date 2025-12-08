import os
import torch
import numpy as np
import tqdm
import sonata
from sonata.scannet_text import ScanNetTextDataset

# --- CONFIGURATION ---
CONFIG = {
    'data_root': 'data/scannet_data/val',  # Change to 'train' as needed
    'output_dir': 'data/scannet_cache_val',     # Change output folder accordingly
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_scannet200': True,  # Set to True to cache with segment200 (200 classes)
    'sonata_config': dict(
        enc_patch_size=[1024 for _ in range(5)],
        enable_flash=False,
        enc_mode=True,
        freeze_encoder=True, 
    )
}

def main():
    print(f"--- Starting Caching Process ---")
    print(f"Source: {CONFIG['data_root']}")
    print(f"Target: {CONFIG['output_dir']}")
    
    # 1. Setup
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    sonata.utils.set_seed(42)

    # 2. Load Dataset
    # We use the dataset directly (no DataLoader) to keep logic simple as requested
    print("Loading Dataset...")
    seg_mode = "ScanNet200 (200 classes)" if CONFIG['use_scannet200'] else "ScanNet20 (20 classes)"
    print(f"Segmentation Mode: {seg_mode}")
    transform = sonata.transform.default()
    dataset = ScanNetTextDataset(data_root=CONFIG['data_root'], transform=transform, use_scannet200=CONFIG['use_scannet200'])

    # 3. Load Model
    print("Loading Sonata Backbone...")
    model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=CONFIG['sonata_config'])
    model = model.to(CONFIG['device'])
    model.eval()

    print(f"Processing {len(dataset)} scenes...")

    # 4. Cache Loop
    # We iterate using range(len) to handle the dataset directly
    for idx in tqdm.tqdm(range(len(dataset))):
        
        # A. Get Data
        try:
            sample = dataset[idx]
        except ValueError as e:
            # Handle scenes with no valid objects
            tqdm.tqdm.write(f"Sample {idx}: Skipped ({str(e)})")
            continue
        
        # Dataset returns (point_data, meta_data) tuple
        point_data, meta_data = sample
        
        # B. Move to GPU (convert dict to Point object for model)
        # Point object needs to be created from dict
        for k, v in point_data.items():
            if isinstance(v, torch.Tensor):
                point_data[k] = v.to(CONFIG['device'], non_blocking=True)
        
        # C. Forward Pass
        with torch.no_grad():
            # The model expects Point-like object and returns dict with stage info
            out = model(point_data)
        
        # D. Prepare Cache Payload
        # Match the exact structure that training script expects
        cache_data = {
            # Meta information (must include text, target_cid, name for training)
            "meta_data": meta_data,
            
            # Stage embeddings and coordinates from Sonata
            "stage_embeddings": [x.detach().cpu() for x in out['stage_embeddings']],
            "stage_coords": [x.detach().cpu() for x in out['stage_coords']],
            
            # Inverse mappings (stage -> dense point mapping)
            "stage_inverse": [x.detach().cpu() if x is not None else None for x in out.get('stage_pooling_inverses', [])],
            
            # Dense-level segmentation data (convert numpy to tensor if needed)
            # Always cache both segment20 and segment200 regardless of use_scannet200 flag
            "dense_segments": torch.from_numpy(meta_data['segment20']) if isinstance(meta_data['segment20'], np.ndarray) else meta_data['segment20'],
            "dense_segments200": torch.from_numpy(meta_data['segment200']) if (meta_data['segment200'] is not None and isinstance(meta_data['segment200'], np.ndarray)) else None,
            "dense_inverse": point_data['inverse'].detach().cpu() if 'inverse' in point_data else None,
            
            # Store both sets of target info for flexibility during training
            "meta_data_segment20_targets": {
                "target_cid": meta_data.get('target_cid_segment20', meta_data['target_cid']),
                "text": meta_data.get('text_segment20', meta_data['text'])
            } if 'target_cid_segment20' in meta_data else None,
            "meta_data_segment200_targets": {
                "target_cid": meta_data.get('target_cid_segment200'),
                "text": meta_data.get('text_segment200')
            } if 'target_cid_segment200' in meta_data else None,
        }
        
        # E. Save to Disk
        scene_name = meta_data['name']
        save_path = os.path.join(CONFIG['output_dir'], f"{scene_name}.pth")
        torch.save(cache_data, save_path)
        
        # Optional: Clear GPU cache every few steps if VRAM is tight
        # if idx % 50 == 0: torch.cuda.empty_cache()

    print(f"\nCaching complete! Saved {len(os.listdir(CONFIG['output_dir']))} files to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()