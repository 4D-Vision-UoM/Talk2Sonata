import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import clip  # Using OpenAI CLIP
from scipy.spatial import cKDTree

from sonata.scannet import ScanNetDataset
import sonata as sonata

# --- CONFIGURATION ---
SEED = 42
DATA_PATH = 'data/scannet_data/val'
OUTPUT_DIR = "outputs/search_vis"
TARGET_TEXT = "a red chair" 

# Ratios for search space pruning. 
# Index 0 is the Top-most (Coarsest) stage.
# We are generous at the top and strict at the bottom.
TOP_K_RATIO = [0.4, 0.3, 0.2, 0.1, 0.05] 

CUSTOM_CONFIG = dict(
    enc_patch_size=[1024 for _ in range(5)],
    enable_flash=False,
    enc_mode=True,
    freeze_encoder=True,
)

# --- 1. THE PROJECTION LAYER ---
class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        # Maps CLIP (512) -> Sonata Stage Dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# --- 2. THE MAIN SEARCH CLASS ---
class SparseHierarchicalSearcher(nn.Module):
    def __init__(self, sonata_channel_dims): 
        """
        sonata_channel_dims: list of ints, dims for stages [0, 1, ..., N]
        """
        super().__init__()
        
        print(f"Initializing Searcher with target dims: {sonata_channel_dims}")
        print("Loading OpenAI CLIP...")
        
        self.clip_model, _ = clip.load("ViT-B/16", device='cuda')
        self.clip_model.eval()
        
        # Create Projectors dynamically based on input list
        self.projectors = nn.ModuleList([
            ProjectionLayer(512, dim) for dim in sonata_channel_dims
        ])
        
    def encode_text(self, text):
        with torch.no_grad():
            tokenized = clip.tokenize([text]).cuda()
            text_emb = self.clip_model.encode_text(tokenized)
            # FIX: CLIP returns float16 on GPU, but Projector is float32.
            # Explicitly cast to float32.
            return F.normalize(text_emb, dim=-1).float()

    def compute_similarity(self, text_emb, stage_feats, projector_idx):
        # 1. Project Text to Stage Space
        projector = self.projectors[projector_idx]
        text_query = projector(text_emb) 
        text_query = F.normalize(text_query, dim=-1)
        
        # 2. Normalize Stage Features
        # FIX: Ensure stage features are also float32 to match text_query
        stage_feats = F.normalize(stage_feats, dim=-1).to(dtype=text_query.dtype)
        
        # 3. Cosine Similarity
        sim_scores = torch.mm(stage_feats, text_query.T).squeeze()
        return sim_scores

# --- 3. UTILS FOR MAPPING ---
def get_children_indices(parent_coords, child_coords, parent_mask):
    """
    Given selected parent coordinates, find spatial children.
    """
    p_coords = parent_coords.detach().cpu().numpy()
    c_coords = child_coords.detach().cpu().numpy()
    p_mask = parent_mask.detach().cpu().numpy()
    
    selected_p_coords = p_coords[p_mask]
    
    if len(selected_p_coords) == 0:
        return torch.zeros(len(child_coords), dtype=torch.bool, device=parent_mask.device)

    # Use KDTree to map children to selected parents
    tree = cKDTree(selected_p_coords)
    dists, _ = tree.query(c_coords, k=1)
    
    # Distance threshold relative to coordinate scale (heuristic)
    keep_mask_np = dists < 0.15 
    
    return torch.tensor(keep_mask_np, device=parent_mask.device)

def save_visualization(coords, mask, stage_name, output_dir):
    coords = coords.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    
    # Color Scheme: Grey for background, Red for selected regions
    colors = np.ones_like(coords) * 0.7 
    colors[mask] = [1.0, 0.0, 0.0]     
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    fname = os.path.join(output_dir, f"{stage_name}.pcd")
    o3d.io.write_point_cloud(fname, pcd)
    print(f"  > Saved visualization: {fname}")


# --- 4. MAIN PIPELINE ---
def main():
    sonata.utils.set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Initializing Pipeline ---")
    dataset = ScanNetDataset(data_root=DATA_PATH)
    
    # Load Sonata Backbone
    model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=CUSTOM_CONFIG).cuda()
    transform = sonata.transform.default()

    print(f"--- Processing Query: '{TARGET_TEXT}' ---")
    sample = dataset[2]
    point = sample.copy()
    point = transform(point)
    
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        
        # 1. RUN BACKBONE FIRST TO DETECT DIMENSIONS
        print("Running Backbone...")
        out = model(point)
        
        # Robustly find embeddings
        if 'stage_embeddings' in out:
            stage_embs = out['stage_embeddings']
            stage_coords = out['stage_coords']
        elif 'embeddings' in out:
            stage_embs = out['embeddings']
            stage_coords = out['coords'] 
        elif 'embeddings' in point: 
             stage_embs = point['embeddings']
             stage_coords = point.get('stage_coords', point.get('coords'))
        else:
            raise ValueError("Could not find embeddings. Check model output keys.")

        # 2. DYNAMICALLY INITIALIZE SEARCHER
        # This fixes the mismatch error. We read the actual dimensions from the model output.
        detected_dims = [feat.shape[1] for feat in stage_embs]
        print(f"Detected {len(stage_embs)} stages with dimensions: {detected_dims}")
        
        searcher = SparseHierarchicalSearcher(detected_dims).cuda()
        
        # 3. Encode Text
        text_emb = searcher.encode_text(TARGET_TEXT) 

        # 4. Hierarchical Search Loop (Top -> Down)
        top_stage_idx = len(stage_embs) - 1 
        valid_mask = torch.ones(len(stage_embs[top_stage_idx]), dtype=torch.bool, device='cuda')
        
        for stage_idx in range(top_stage_idx, -1, -1):
            print(f"\n--- Processing Stage {stage_idx} (Dim: {stage_embs[stage_idx].shape[1]}) ---")
            
            curr_feats = stage_embs[stage_idx]
            curr_coords = stage_coords[stage_idx]
            
            # Stop if search space empty
            if stage_idx < top_stage_idx and valid_mask.sum() == 0:
                print("Search space collapsed. Stopping.")
                break
            
            # A. Similarity
            scores = searcher.compute_similarity(text_emb, curr_feats, stage_idx)
            
            # B. Masking (Only search within valid parents)
            scores[~valid_mask] = -float('inf')
            
            # C. Selection (Top-K)
            # Determine ratio based on depth
            ratio_idx = top_stage_idx - stage_idx
            ratio = TOP_K_RATIO[ratio_idx] if ratio_idx < len(TOP_K_RATIO) else 0.01
            
            # Calculate K
            k = int(len(scores) * ratio)
            k = max(k, 1)
            num_valid = valid_mask.sum().item()
            actual_k = min(k, num_valid)
            
            # Select
            values, top_indices = torch.topk(scores, actual_k)
            selection_mask = torch.zeros_like(scores, dtype=torch.bool)
            selection_mask[top_indices] = True
            
            print(f"Selected {actual_k}/{len(scores)} voxels using ratio {ratio}")
            save_visualization(curr_coords, selection_mask, f"stage_{stage_idx}_proposal", OUTPUT_DIR)
            
            # D. Propagate to Next Stage
            if stage_idx > 0:
                next_stage_coords = stage_coords[stage_idx - 1]
                valid_mask = get_children_indices(curr_coords, next_stage_coords, selection_mask)
                print(f"Propagating to Stage {stage_idx-1}: {valid_mask.sum()} children candidates.")

    print("\nSearch Complete.")

if __name__ == "__main__":
    main()