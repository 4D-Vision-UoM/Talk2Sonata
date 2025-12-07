import os
import torch
import tqdm
import sonata
from sonata.scannet_text import ScanNetTextDataset

# --- CONFIGURATION ---
CONFIG = {
    'data_root': 'data/scannet_data/train',  # Change to 'train' as needed
    'output_dir': 'outputs/scannet_cache_train',     # Change output folder accordingly
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
    transform = sonata.transform.default()
    dataset = ScanNetTextDataset(data_root=CONFIG['data_root'], transform=transform)

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
        sample = dataset[idx]
        
        # Handle skipped scenes (e.g. no valid objects found)
        if sample is None:
            # tqdm.write allows printing without breaking the progress bar
            tqdm.tqdm.write(f"Sample {idx}: Skipped (no valid objects)")
            continue
        
        point_data, meta_data = sample
        
        # B. Move to GPU
        # Only move tensors that exist in the point_data dictionary
        for k, v in point_data.items():
            if isinstance(v, torch.Tensor):
                point_data[k] = v.to(CONFIG['device'], non_blocking=True)
        
        # C. Forward Pass
        with torch.no_grad():
            # The model modifies point_data in place or returns a dict-like object
            out = model(point_data)
        
        # D. Prepare Cache Payload
        # IMPORTANT: Move everything to CPU before saving to avoid CUDA requirement on load
        cache_data = {
            "dense_segments": meta_data.pop('segment20'),
            "meta_data": meta_data,
            # Sparse Features (List of Tensors)
            "stage_embeddings": [x.detach().cpu() for x in out['stage_embeddings']],
            "stage_coords": [x.detach().cpu() for x in out['stage_coords']],
            # Dense Data (Useful for mapping back to original points)
            "dense_coord": point_data['coord'].detach().cpu(),
            "dense_inverse": point_data['inverse'].detach().cpu(), # Ensure this key matches your dataset return
            # Save pooling inverses if available (helps mapping sparse->dense)
            "stage_inverse": [x.detach().cpu() if x is not None else None for x in out.get('stage_pooling_inverses', [])]
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