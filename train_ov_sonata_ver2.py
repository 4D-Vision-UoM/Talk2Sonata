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
    'train_data_root': 'outputs/scannet_cache_train', 
    'val_data_root': 'outputs/scannet_cache_val',     
    'output_dir': 'outputs/ov_sonata_ver2',
    'batch_size': 32, 
    'lr': 1e-3,          
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 90,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'clip_model_name': 'ViT-L/14',
    # FIX: Randomize seed at startup for different targets across runs
    'validation_seed': 12435,
    # FIX: Weights for Deep Supervision (S0 is most important)
    # [Stage 0, Stage 1, Stage 2, Stage 3, Stage 4]
    'stage_loss_weights': [1.0, 0.8, 0.4, 0.2, 0.1] 
}

# --- LOGGING ---
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(os.path.join(log_dir, "training.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)

# --- DATASET ---
class CachedScanNetDataset(Dataset):
    def __init__(self, data_root, is_train=True):
        self.files = sorted(glob.glob(os.path.join(data_root, "*.pth")))
        self.is_train = is_train
        if len(self.files) == 0: raise ValueError(f"No .pth files in {data_root}")
        print(f"Found {len(self.files)} scenes in {data_root} (Train={is_train})")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu')
        meta = data.get('meta_data', data)
        
        texts, cids = meta['text'], meta['target_cid']
        if isinstance(cids, int): cids = [cids]
        if isinstance(texts, str): texts = [texts]

        if self.is_train:
            idx = random.randint(0, len(texts) - 1)
        else:
            state = random.Random(meta['name'] + str(CONFIG['validation_seed']))
            idx = state.randint(0, len(texts) - 1)
            
        data['selected_text'] = texts[idx]
        data['selected_cid'] = cids[idx]
        return data

def custom_collate_fn(batch): return batch

# --- MODELS ---
class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), # Extra depth
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
    def __init__(self, stage_dims, clip_name='ViT-L/14'):
        super().__init__()
        print(f"Initializing Searcher. Dims: {stage_dims}")
        self.clip_model, _ = clip.load(clip_name, device='cuda')
        self.clip_model.eval()
        for p in self.clip_model.parameters(): p.requires_grad = False
        
        clip_dim = 768 if 'ViT-L' in clip_name else 512
        
        self.projectors = nn.ModuleList([
            ProjectionLayer(clip_dim, dim) for dim in stage_dims
        ])
        
        # FIX: Learnable Temperature (Logit Scale) like CLIP
        # Initialize to log(1/0.07) ~= 2.65
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text):
        with torch.no_grad():
            tokenized = clip.tokenize(text).cuda()
            emb = self.clip_model.encode_text(tokenized)
            return F.normalize(emb, dim=-1).float() 

    def forward_stage(self, text_emb, stage_feats, stage_idx):
        projector = self.projectors[stage_idx]
        text_query = projector(text_emb) 
        text_query = F.normalize(text_query, dim=-1)
        
        stage_feats = F.normalize(stage_feats, dim=-1).to(text_query.dtype)
        
        # FIX: Apply Temperature Scaling
        # (N, C) @ (C, 1) -> (N, 1) * scalar
        logit_scale = self.logit_scale.exp()
        return torch.mm(stage_feats, text_query.T).squeeze() * logit_scale

# --- LOSS FUNCTIONS ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs are logits, convert to prob
        probs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = SigmoidFocalLoss(alpha=0.75, gamma=2.0)
        self.dice = DiceLoss()
        
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        # Combine: 1.0 Focal + 1.0 Dice
        return focal_loss + dice_loss

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
            loss = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * loss
        return loss.mean() if self.reduction == "mean" else loss

# --- UTILS ---
def get_hierarchical_masks(dense_labels, dense_inverse, stage_inverses, target_cid, device):
    masks = []
    # 1. Dense -> Stage 0
    target_idx = torch.nonzero(dense_labels == target_cid, as_tuple=True)[0]
    active_s0 = dense_inverse[target_idx]
    current_active = torch.unique(active_s0)
    active_indices_list = [current_active]
    
    # 2. Propagate Up
    for i in range(1, len(stage_inverses)):
        inv_map = stage_inverses[i]
        if inv_map is None: break
        # FIX: Ensure inv_map is a tensor before moving to device (handles Numpy cache)
        if not isinstance(inv_map, torch.Tensor):
            inv_map = torch.as_tensor(inv_map)
        inv_map = inv_map.to(device)
        
        current_active = torch.unique(inv_map[current_active])
        active_indices_list.append(current_active)
    return active_indices_list

def save_vis(save_dir, epoch, name, text, coords, preds, targets):
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    coords = coords.detach().cpu().numpy()
    preds = torch.sigmoid(preds).detach().cpu().numpy() > 0.5
    targets = targets.detach().cpu().numpy() > 0.5
    
    # Pred (Red)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    cols = np.ones_like(coords) * 0.7
    cols[preds] = [1.0, 0.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(os.path.join(vis_dir, f"E{epoch}_{name}_{text}_pred.pcd"), pcd)
    
    # GT (Green)
    cols = np.ones_like(coords) * 0.7
    cols[targets] = [0.0, 1.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(os.path.join(vis_dir, f"E{epoch}_{name}_{text}_gt.pcd"), pcd)

# --- TRAINER ---
class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config['device']
        
        self.train_ds = CachedScanNetDataset(config['train_data_root'], True)
        self.val_ds = CachedScanNetDataset(config['val_data_root'], False)
        
        self.train_loader = DataLoader(self.train_ds, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True, num_workers=config['num_workers'])
        self.val_loader = DataLoader(self.val_ds, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config['num_workers'])
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = CombinedLoss() # FIX: Using Focal + Dice
        self.writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'logs'))
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def init_model(self, sample_dims):
        if self.model is None:
            self.model = SparseHierarchicalSearcher(sample_dims, self.config['clip_model_name']).to(self.device)
            # Optimize Projectors AND Logit Scale
            self.optimizer = optim.AdamW([
                {'params': self.model.projectors.parameters()},
                {'params': [self.model.logit_scale], 'lr': 1e-2} # Higher LR for temp
            ], lr=self.config['lr'], weight_decay=self.config['weight_decay'])
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['epochs'])
            self.logger.info(f"Model Initialized. Dims: {sample_dims}")

    def run_epoch(self, epoch, is_train=True):
        if self.model: self.model.train() if is_train else self.model.eval()
        loader = self.train_loader if is_train else self.val_loader
        
        total_loss = 0
        total_inter, total_union = 0, 0
        num_batches = 0
        best_vis_loss = float('inf')
        best_vis = None
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} {'Train' if is_train else 'Val'}")
        
        for batch in pbar:
            batch_loss = 0
            valid_samples = 0
            if is_train and self.model: self.optimizer.zero_grad()
            
            for sample in batch:
                if sample['dense_segments'] is None: continue
                
                # FIX: Robust conversion of Numpy arrays to Tensor before .to()
                # Handles both cases if cache is mixed format
                dense_lbl = torch.as_tensor(sample['dense_segments']).long().to(self.device)
                dense_inv = torch.as_tensor(sample['dense_inverse']).long().to(self.device)
                
                # Init
                stage_embs = sample['stage_embeddings']
                if not self.model: self.init_model([f.shape[1] for f in stage_embs])
                
                target_cid = sample['selected_cid']
                
                # Get Active Indices
                try:
                    active_idcs = get_hierarchical_masks(dense_lbl, dense_inv, sample['stage_inverse'], target_cid, self.device)
                except: continue
                if len(active_idcs[0]) == 0: continue

                text_emb = self.model.encode_text(sample['selected_text'])
                scene_loss = 0
                
                # S0 -> S4 Loop
                for s_idx, (feat, active) in enumerate(zip(stage_embs, active_idcs)):
                    feat = feat.to(self.device)
                    target = torch.zeros(feat.shape[0], device=self.device)
                    target[active] = 1.0
                    
                    logits = self.model.forward_stage(text_emb, feat, s_idx)
                    loss = self.criterion(logits, target)
                    
                    # FIX: Correct Stage Weighting (S0=1.0, S4=0.1)
                    w = self.config['stage_loss_weights'][s_idx]
                    scene_loss += loss * w
                    
                    if s_idx == 0:
                        with torch.no_grad():
                            preds = (torch.sigmoid(logits) > 0.5).float()
                            total_inter += (preds * target).sum().item()
                            total_union += torch.max(preds, target).sum().item()
                            
                            # Vis tracking
                            if not is_train and loss.item() < best_vis_loss:
                                best_vis_loss = loss.item()
                                name = sample.get('meta_data', {}).get('name', 'unk')
                                best_vis = (name, sample['selected_text'], sample['stage_coords'][0], logits, target)

                batch_loss += scene_loss
                valid_samples += 1

            if valid_samples > 0:
                batch_loss /= valid_samples
                if is_train:
                    batch_loss.backward()
                    self.optimizer.step()
                total_loss += batch_loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': batch_loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        miou = total_inter / (total_union + 1e-6)
        
        prefix = "Train" if is_train else "Val"
        self.writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        self.writer.add_scalar(f'mIoU/{prefix}', miou, epoch)
        self.writer.add_scalar('Temp', self.model.logit_scale.exp().item(), epoch)
        
        self.logger.info(f"{prefix} E{epoch}: Loss={avg_loss:.4f} mIoU={miou:.4f}")
        
        if not is_train and best_vis:
            save_vis(self.config['output_dir'], epoch, *best_vis)
            
        return avg_loss

    def train(self):
        self.logger.info("Training...")
        for epoch in range(self.config['epochs']):
            self.run_epoch(epoch, True)
            if self.scheduler: self.scheduler.step()
            
            if epoch % 1 == 0:
                val_loss = self.run_epoch(epoch, False)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], 'best_model.pth'))
                    self.logger.info(">>> Best Saved")
                else:
                    self.patience_counter += 1
                if self.patience_counter >= self.config['patience']: break

if __name__ == "__main__":
    logger = setup_logging(CONFIG['output_dir'])
    Trainer(CONFIG, logger).train()