import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import clip
from scipy.spatial import cKDTree
from tqdm import tqdm
from torch.utils.data import DataLoader

# Imports
import sonata
from sonata.scannet_text import ScanNetTextDataset, training_collate_fn

# --- CONFIGURATION ---
CONFIG = {
    'batch_size': 8,  # CRITICAL FIX: Reduced from 16 to 1 to prevent OOM
    'num_workers': 2,
    'data_root': 'data/scannet_data/val',
    'output_dir': 'outputs/batch_inference',
    'top_k_ratios': [0.4, 0.3, 0.2, 0.1, 0.05], # From S4 down to S0
    'device': 'cuda'
}

# Sonata Config
CUSTOM_SONATA_CONFIG = dict(
    enc_patch_size=[1024 for _ in range(5)],
    enable_flash=False,
    enc_mode=True,
    freeze_encoder=True, 
)

# --- SEARCHER MODEL ---
class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
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

class SparseHierarchicalSearcher(nn.Module):
    def __init__(self, stage_dims):
        super().__init__()
        print(f"Initializing Searcher for dims: {stage_dims}")
        self.clip_model, _ = clip.load("ViT-B/16", device='cuda')
        self.clip_model.eval()
        
        self.projectors = nn.ModuleList([
            ProjectionLayer(512, dim) for dim in stage_dims
        ])
        
    def encode_text(self, text_list):
        with torch.no_grad():
            tokenized = clip.tokenize(text_list).cuda()
            text_emb = self.clip_model.encode_text(tokenized)
            return F.normalize(text_emb, dim=-1).float() # Explicit Float32

    def compute_similarity(self, text_emb, stage_feats, stage_idx):
        projector = self.projectors[stage_idx]
        text_query = projector(text_emb) # (1, Dim)
        text_query = F.normalize(text_query, dim=-1)
        
        stage_feats = F.normalize(stage_feats, dim=-1).to(text_query.dtype)
        
        # Cosine Sim: (N, C) @ (C, 1) -> (N, 1)
        sim_scores = torch.mm(stage_feats, text_query.T).squeeze()
        return sim_scores

# --- UTILS ---
def get_children_mask(parent_coords, child_coords, parent_mask):
    """Filter child voxels that are spatially near selected parent voxels."""
    if parent_mask.sum() == 0:
        return torch.zeros(len(child_coords), dtype=torch.bool, device=parent_mask.device)

    p_coords_np = parent_coords.detach().cpu().numpy()
    c_coords_np = child_coords.detach().cpu().numpy()
    p_mask_np = parent_mask.detach().cpu().numpy()
    
    selected_parents = p_coords_np[p_mask_np]
    tree = cKDTree(selected_parents)
    
    # Radius search: If child is within 0.15 units of a selected parent, keep it
    dists, _ = tree.query(c_coords_np, k=1)
    keep = dists < 0.15 
    
    return torch.tensor(keep, device=parent_mask.device)

def save_pcd(coords, mask, filename):
    coords_np = coords.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    
    colors = np.ones_like(coords_np) * 0.7  # Grey background
    colors[mask_np] = [1.0, 0.0, 0.0]       # Red selected
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

# --- MAIN INFERENCE LOOP ---
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    sonata.utils.set_seed(42)
    
    # Optional: Set memory management env var if fragmentation persists
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # 1. Load Data
    print("Loading Dataset...")
    dataset = ScanNetTextDataset(CONFIG['data_root'], transform=sonata.transform.default())
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], 
                        collate_fn=training_collate_fn, shuffle=False)

    # 2. Load Backbone
    print("Loading Backbone...")
    backbone = sonata.load("sonata", repo_id="facebook/sonata", custom_config=CUSTOM_SONATA_CONFIG).to(CONFIG['device'])
    backbone.eval()

    # 3. Searcher Placeholder (Init after first batch)
    searcher = None
    
    print("Starting Batch Inference...")
    
    # Iterate Batches
    for batch_idx, (data_dict, meta_list) in enumerate(tqdm(loader)):
        
        # Cleanup from previous iteration
        torch.cuda.empty_cache()

        # Move Data to GPU
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(CONFIG['device'], non_blocking=True)

        # A. Run Backbone
        with torch.no_grad():
            out = backbone(data_dict)
            stage_embs = out.get('stage_embeddings')
            stage_coords = out.get('stage_coords')
            
            if not stage_embs: continue

        # B. Init Searcher (Once)
        if searcher is None:
            dims = [f.shape[1] for f in stage_embs]
            searcher = SparseHierarchicalSearcher(dims).to(CONFIG['device'])
            # Load checkpoint here if you have one!
            # searcher.load_state_dict(torch.load("path/to/best_model.pth"))

        # C. Encode Texts
        texts = [m['text'] for m in meta_list]
        text_embs = searcher.encode_text(texts) # (B, 512)

        # D. Per-Scene Drill Down
        # Note: We must slice the sparse tensors per scene because Drill-Down is sequential
        
        # Handle 'batch' key missing (Sonata sometimes uses only 'offset')
        if 'batch' in data_dict:
            batch_indices = data_dict['batch']
        elif 'offset' in data_dict:
            # Reconstruct batch indices from offset
            offset = data_dict['offset'] # (B,)
            total_points = offset[-1].item()
            batch_indices = torch.zeros(total_points, dtype=torch.long, device=CONFIG['device'])
            start = 0
            for i, end in enumerate(offset):
                batch_indices[start:end] = i
                start = end
        else:
            raise KeyError("Data dict missing 'batch' and 'offset'. Cannot separate scenes.")
        
        # We need to find which stage coords belong to which batch index.
        # Simple heuristic: Use the dense batch index and propagate (KDTree) 
        # OR just run the similarity globally and then mask.
        
        # Pre-calculate Batch IDs for all stages (Propagate Dense -> Sparse)
        # Using KDTree to assign every sparse voxel to a batch index
        dense_coords_cpu = data_dict['coord'].detach().cpu().numpy()
        dense_batch_cpu = batch_indices.detach().cpu().numpy() # Use the reconstructed/retrieved batch indices
        tree = cKDTree(dense_coords_cpu)
        
        stage_batch_ids = []
        for s_coord in stage_coords:
            s_coord_np = s_coord.detach().cpu().numpy()
            _, indices = tree.query(s_coord_np, k=1)
            # Assign batch ID of nearest dense point
            stage_batch_ids.append(torch.tensor(dense_batch_cpu[indices], device=CONFIG['device']))

        # --- SCENE LOOP ---
        for b_i in range(len(meta_list)):
            scene_name = meta_list[b_i]['name']
            target_text = texts[b_i]
            target_emb = text_embs[b_i].unsqueeze(0) # (1, 512)
            
            print(f"\nProcessing Scene: {scene_name} | Query: '{target_text}'")
            
            # --- STAGE LOOP (Top -> Bottom) ---
            top_stage_idx = len(stage_embs) - 1
            valid_mask = None # Defined in first iter
            
            for s_idx in range(top_stage_idx, -1, -1):
                # 1. Get Scene Data for this stage
                s_feat = stage_embs[s_idx]
                s_coord = stage_coords[s_idx]
                s_batch = stage_batch_ids[s_idx]
                
                # Mask: Points belonging to this scene
                scene_mask = (s_batch == b_i)
                
                # If valid_mask exists (from parent), apply it
                if valid_mask is not None:
                    # valid_mask is only for points IN THIS SCENE. 
                    pass
                else:
                    # Initialize: All points in this scene are candidates
                    valid_mask = scene_mask.clone()

                # If no points left, stop
                if valid_mask.sum() == 0:
                    break

                # 2. Compute Sim
                scores = searcher.compute_similarity(target_emb, s_feat, s_idx)
                
                # 3. Masking
                # Set scores of non-candidates (wrong scene OR not child of winner) to -inf
                scores[~valid_mask] = -float('inf')
                
                # 4. Selection
                # Ratio logic
                ratio = CONFIG['top_k_ratios'][top_stage_idx - s_idx]
                num_scene_points = scene_mask.sum().item()
                k = max(int(num_scene_points * ratio), 1)
                
                # Top K
                _, top_indices = torch.topk(scores, k)
                
                # Create next mask
                new_selection = torch.zeros_like(scores, dtype=torch.bool)
                new_selection[top_indices] = True
                
                # Save Vis
                out_name = os.path.join(CONFIG['output_dir'], f"{scene_name}_{target_text}_S{s_idx}.pcd")
                save_pcd(s_coord, new_selection, out_name)
                
                # 5. Propagate Down
                if s_idx > 0:
                    next_coord = stage_coords[s_idx - 1]
                    next_batch = stage_batch_ids[s_idx - 1]
                    
                    # Find children of selected parents
                    # Note: We must ensure children also belong to the scene
                    children_in_scene = (next_batch == b_i)
                    
                    # Logic: If child is in scene AND close to selected parent -> Keep
                    child_is_near = get_children_mask(s_coord, next_coord, new_selection)
                    valid_mask = child_is_near & children_in_scene
                    
        # Limit to 1 batch for testing
        # break 
    
    print(f"Done. Results in {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()