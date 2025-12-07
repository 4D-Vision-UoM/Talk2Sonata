import os
import glob
import time
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open3d as o3d
import clip
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- CONFIGURATION ---
CONFIG = {
    'train_data_root': 'outputs/scannet_cache_train', # Updated to match your cache path
    'val_data_root': 'outputs/scannet_cache_val',     # Updated to match your cache path
    'output_dir': 'outputs/ov_sonata',
    'batch_size': 8, 
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Randomize seed at STARTUP so every run is different, 
    # but the value stays fixed during the loop (ensures consistent validation targets).
    'validation_seed': random.randint(0, 1000000)
}

# --- LOGGING SETUP ---
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- DATASET ---
class CachedScanNetDataset(Dataset):
    def __init__(self, data_root, is_train=True):
        self.files = sorted(glob.glob(os.path.join(data_root, "*.pth")))
        self.is_train = is_train
        if len(self.files) == 0:
            raise ValueError(f"No .pth files found in {data_root}")
        print(f"Found {len(self.files)} cached scenes in {data_root} (Train={is_train})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load data (CPU to save VRAM)
        data = torch.load(self.files[idx], map_location='cpu')
        
        # --- Target Selection Logic ---
        # Note: Your cache inspection showed 'meta_data' inside the dict
        meta = data.get('meta_data', data) # Handle nested or flat structure
        
        available_texts = meta['text']
        available_cids = meta['target_cid']
        scene_name = meta['name']
        
        # Ensure cids is a list
        if isinstance(available_cids, int): available_cids = [available_cids]
        if isinstance(available_texts, str): available_texts = [available_texts]

        if self.is_train:
            # Training: Pick Randomly every time
            choice_idx = random.randint(0, len(available_texts) - 1)
        else:
            # Validation: Deterministic pick based on scene name & fixed seed
            # This ensures Scene X always evaluates on Object Y for the whole run
            state = random.Random(scene_name + str(CONFIG['validation_seed']))
            choice_idx = state.randint(0, len(available_texts) - 1)
            
        selected_text = available_texts[choice_idx]
        selected_cid = available_cids[choice_idx]
        
        # Inject selection back into data dict for the Trainer
        data['selected_text'] = selected_text
        data['selected_cid'] = selected_cid
        
        return data

def custom_collate_fn(batch):
    return batch

# --- MODELS ---
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
        for p in self.clip_model.parameters(): p.requires_grad = False
        
        self.projectors = nn.ModuleList([
            ProjectionLayer(512, dim) for dim in stage_dims
        ])
        
    def encode_text(self, text):
        with torch.no_grad():
            tokenized = clip.tokenize(text).cuda()
            text_emb = self.clip_model.encode_text(tokenized)
            return F.normalize(text_emb, dim=-1).float() 

    def forward_stage(self, text_emb, stage_feats, stage_idx):
        projector = self.projectors[stage_idx]
        text_query = projector(text_emb) # (1, Dim)
        text_query = F.normalize(text_query, dim=-1)
        stage_feats = F.normalize(stage_feats, dim=-1).to(text_query.dtype)
        # (N, C) @ (C, 1) -> (N, 1)
        return torch.mm(stage_feats, text_query.T).squeeze()

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean() if self.reduction == "mean" else loss

# --- CORE UTILS ---
def get_hierarchical_masks(dense_labels, dense_inverse, stage_inverses, target_cid, device):
    """
    Propagates Ground Truth from Dense -> S0 -> S4 using cached inverse maps.
    Returns a list of binary masks [Mask_S0, Mask_S1, ..., Mask_S4]
    """
    masks = []
    
    # 1. Dense -> Stage 0
    # dense_inverse maps: Original_Point_Index -> S0_Voxel_Index
    # Find all original points belonging to target class
    target_indices_dense = torch.nonzero(dense_labels == target_cid, as_tuple=True)[0]
    
    # Find corresponding S0 voxels
    active_s0_indices = dense_inverse[target_indices_dense]
    
    # Determine size of S0 from inverse map range or passed shape?
    # We can infer size from stage_inverses[1] length if it exists, or passed separately.
    # Hack: use max index + 1 or better, use the known size from stage_coords (passed in loop)
    # Let's return Indices instead of Boolean Masks to be size-agnostic here? 
    # No, boolean masks are easier for Loss. We will generate them inside the loop where we know sizes.
    
    # Actually, simpler approach: Return ACTIVE INDICES for each stage.
    active_indices_list = []
    
    # S0 Indices
    current_active_indices = torch.unique(active_s0_indices)
    active_indices_list.append(current_active_indices)
    
    # 2. Propagate Up (S0 -> S1 -> S2...)
    # stage_inverses is [None, Inv_S0->S1, Inv_S1->S2, ...]
    for i in range(1, len(stage_inverses)):
        inv_map = stage_inverses[i] # Maps Stage(i-1) -> Stage(i)
        
        if inv_map is None: break
        
        inv_map = inv_map.to(device)
        
        # Get parents of currently active children
        # Note: inv_map length = num_children. value = parent_index.
        
        # We need to filter inv_map by current_active_indices
        parent_indices = inv_map[current_active_indices]
        current_active_indices = torch.unique(parent_indices)
        active_indices_list.append(current_active_indices)
        
    return active_indices_list

def save_vis(save_dir, epoch, scene_name, text, coords, preds, targets):
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    coords = coords.detach().cpu().numpy()
    preds = torch.sigmoid(preds).detach().cpu().numpy() > 0.5
    targets = targets.detach().cpu().numpy() > 0.5
    
    # Pred (Red)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    colors = np.ones_like(coords) * 0.7
    colors[preds] = [1.0, 0.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(vis_dir, f"E{epoch}_{scene_name}_{text}_pred.pcd"), pcd)
    
    # GT (Green)
    colors = np.ones_like(coords) * 0.7
    colors[targets] = [0.0, 1.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(vis_dir, f"E{epoch}_{scene_name}_{text}_gt.pcd"), pcd)

# --- TRAINER ---
class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config['device']
        
        self.train_ds = CachedScanNetDataset(config['train_data_root'], is_train=True)
        self.val_ds = CachedScanNetDataset(config['val_data_root'], is_train=False)
        
        self.train_loader = DataLoader(self.train_ds, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True, num_workers=config['num_workers'])
        self.val_loader = DataLoader(self.val_ds, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config['num_workers'])
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = SigmoidFocalLoss()
        self.writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'logs'))
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def init_model(self, sample_dims):
        if self.model is None:
            self.model = SparseHierarchicalSearcher(sample_dims).to(self.device)
            self.optimizer = optim.AdamW(self.model.projectors.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['epochs'])
            self.logger.info(f"Model Initialized for dims: {sample_dims}")

    def run_epoch(self, epoch, is_train=True):
        if self.model:
            self.model.train() if is_train else self.model.eval()
            
        loader = self.train_loader if is_train else self.val_loader
        epoch_loss = 0
        num_batches = 0
        best_vis_loss = float('inf')
        best_vis_payload = None
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} {'Train' if is_train else 'Val'}")
        
        for batch_list in pbar:
            batch_loss = 0
            valid_samples = 0
            if is_train and self.model: self.optimizer.zero_grad()
            
            for sample in batch_list:
                # 1. Unpack
                target_text = sample['selected_text']
                target_cid = sample['selected_cid']
                
                # Check for cached None
                if sample['dense_segments'] is None: continue
                
                # Move basic dense data to GPU for mask calc
                # FIX: Convert numpy to tensor if necessary
                dense_labels = torch.as_tensor(sample['dense_segments']).to(self.device)
                dense_inverse = torch.as_tensor(sample['dense_inverse']).to(self.device) # Maps Raw -> S0
                
                # Init Model if needed
                stage_embs = sample['stage_embeddings']
                if self.model is None:
                    self.init_model([f.shape[1] for f in stage_embs])
                    self.model.train() if is_train else self.model.eval()

                # Encode Text
                text_emb = self.model.encode_text(target_text)

                # 2. Propagate Ground Truth (The "Link" Logic)
                # We need a list of active indices for each stage [S0_idx, S1_idx, ..., S4_idx]
                stage_inverses = sample['stage_inverse'] # List of tensors
                
                try:
                    active_indices_per_stage = get_hierarchical_masks(
                        dense_labels, dense_inverse, stage_inverses, target_cid, self.device
                    )
                except Exception as e:
                    # Fallback for empty/mismatched scenes
                    continue
                
                # If object not present in scene, skip (or use negative sampling? skipping for now)
                if len(active_indices_per_stage[0]) == 0: continue

                # 3. Multi-Stage Loss
                scene_loss = 0
                vis_logits, vis_coords, vis_targets = None, None, None
                
                # Loop S0 -> S4
                for s_idx, (s_feat, active_indices) in enumerate(zip(stage_embs, active_indices_per_stage)):
                    s_feat = s_feat.to(self.device)
                    
                    # Create Binary Target Mask for this stage
                    binary_target = torch.zeros(s_feat.shape[0], device=self.device)
                    binary_target[active_indices] = 1.0
                    
                    # Forward
                    logits = self.model.forward_stage(text_emb, s_feat, s_idx)
                    
                    loss = self.criterion(logits, binary_target)
                    scene_loss += loss
                    
                    # Capture S0 for vis
                    if s_idx == 0:
                        vis_logits = logits
                        vis_targets = binary_target
                        vis_coords = sample['stage_coords'][0] # Keep on CPU for vis save

                batch_loss += scene_loss
                valid_samples += 1
                
                # Vis Tracking
                if not is_train and scene_loss.item() < best_vis_loss:
                    best_vis_loss = scene_loss.item()
                    # Safely access name from meta_data
                    s_name = sample['meta_data']['name'] if 'meta_data' in sample else sample.get('name', 'unknown')
                    best_vis_payload = (s_name, target_text, vis_coords, vis_logits, vis_targets)

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                if is_train:
                    batch_loss.backward()
                    self.optimizer.step()
                epoch_loss += batch_loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': batch_loss.item()})

        avg_loss = epoch_loss / max(num_batches, 1)
        self.writer.add_scalar(f'Loss/{"Train" if is_train else "Val"}', avg_loss, epoch)
        
        if not is_train and best_vis_payload:
            self.logger.info(f"Saving Vis for {best_vis_payload[0]}")
            save_vis(self.config['output_dir'], epoch, *best_vis_payload)
            
        return avg_loss

    def train(self):
        self.logger.info("Starting Training...")
        self.logger.info(f"Validation Seed: {self.config['validation_seed']}")
        for epoch in range(self.config['epochs']):
            train_loss = self.run_epoch(epoch, is_train=True)
            self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
            
            if self.scheduler: self.scheduler.step()
            
            if epoch % 1 == 0:
                val_loss = self.run_epoch(epoch, is_train=False)
                self.logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss and self.model:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'best_model.pth'))
                    self.logger.info(">>> Best Model Saved")
                else:
                    self.patience_counter += 1
                
                if self.model:
                    torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'last_model.pth'))
                
                if self.patience_counter >= self.config['patience']:
                    self.logger.info("Early stop.")
                    break

if __name__ == "__main__":
    logger = setup_logging(CONFIG['output_dir'])
    trainer = Trainer(CONFIG, logger)
    trainer.train()