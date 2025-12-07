"""
SonataText: Open-Vocabulary 3D Segmentation Model
Combines Sonata (3D point cloud backbone) with CLIP (text encoder)
Inspired by Talk2DINO architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import yaml
import os
import sys

# Add sonata to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../sonata'))
import sonata

from src.model import ProjectionLayer
from datasets import get_template


class SonataProjectionLayer(nn.Module):
    """
    Maps CLIP text embeddings to Sonata point feature space
    Similar to ProjectionLayer in Talk2DINO but adapted for 3D
    """
    def __init__(self, 
                 sonata_embed_dim=512,
                 clip_embed_dim=512,
                 hidden_layer=True,
                 act=nn.Tanh(),
                 cosine=True):
        super().__init__()
        self.sonata_embed_dim = sonata_embed_dim
        self.clip_embed_dim = clip_embed_dim
        self.cosine = cosine
        
        # Projection from CLIP to Sonata space
        self.linear_layer = nn.Linear(clip_embed_dim, sonata_embed_dim)
        if hidden_layer:
            self.linear_layer2 = nn.Linear(sonata_embed_dim, sonata_embed_dim)
        self.act = act
    
    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['projection']
        
        # Load activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            sonata_embed_dim=config.get('sonata_embed_dim', 512),
            clip_embed_dim=config.get('clip_embed_dim', 512),
            hidden_layer=config.get('hidden_layer', True),
            act=act,
            cosine=config.get('cosine', True)
        )
        return model
    
    def project_clip_txt(self, text_embedding):
        """Project CLIP text embeddings to Sonata space"""
        text_embedding = text_embedding.float()
        
        x = self.linear_layer(text_embedding)
        if self.act:
            x = self.act(x)
        if hasattr(self, 'linear_layer2'):
            x = self.linear_layer2(x)
        
        return x
    
    def forward(self, point_embedding, textual_embedding, ret_embeds=True):
        """
        Args:
            point_embedding: (N, sonata_dim) - Sonata point features
            textual_embedding: (K, clip_dim) - CLIP text features
            ret_embeds: return normalized embeddings
        
        Returns:
            Normalized embeddings or similarity matrix
        """
        if textual_embedding is not None:
            textual_embedding = self.project_clip_txt(textual_embedding)
        
        if point_embedding is not None and textual_embedding is not None:
            if self.cosine:
                point_embedding = F.normalize(point_embedding, p=2, dim=1)
                textual_embedding = F.normalize(textual_embedding, p=2, dim=1)
            
            if ret_embeds:
                return point_embedding, textual_embedding
            else:
                # Compute similarity matrix
                sim = point_embedding @ textual_embedding.T
                return sim
        
        # Return only the processed embedding that was provided
        if point_embedding is not None:
            if self.cosine:
                point_embedding = F.normalize(point_embedding, p=2, dim=1)
            return point_embedding, None
        
        if textual_embedding is not None:
            if self.cosine:
                textual_embedding = F.normalize(textual_embedding, p=2, dim=1)
            return None, textual_embedding


class SonataTextMasker(nn.Module):
    """
    Generates per-point class predictions for 3D point clouds
    Based on similarity between Sonata features and text embeddings
    """
    def __init__(self, similarity_type="cosine", temperature=1.0):
        super().__init__()
        self.similarity_type = similarity_type
        self.temperature = temperature
    
    def forward_seg(self, point_feat, text_emb, hard=False):
        """
        Args:
            point_feat: (N, C) - Sonata point features (normalized)
            text_emb: (K, C) - Projected text embeddings (normalized)
            hard: if True, return hard assignments (argmax)
        
        Returns:
            predictions: (N, K) or (N,) - Per-point class scores or labels
            simmap: (N, K) - Similarity scores
        """
        # Compute similarity: (N, C) @ (K, C).T → (N, K)
        simmap = point_feat @ text_emb.T
        simmap = simmap / self.temperature
        
        if hard:
            predictions = simmap.argmax(dim=-1)
        else:
            predictions = simmap  # Soft scores
        
        return predictions, simmap


class SonataText(nn.Module):
    """
    Open-vocabulary 3D segmentation model combining Sonata + CLIP
    
    Architecture:
        Point Cloud → Sonata → Point Features → Projection ← CLIP Text
                                                      ↓
                                              Similarity Matching
                                                      ↓
                                              3D Point-wise Labels
    """
    def __init__(self, 
                 sonata_model_name="sonata",
                 sonata_repo_id="facebook/sonata",
                 clip_model_name="ViT-B/16",
                 proj_config=None,
                 freeze_sonata=True,
                 freeze_clip=True,
                 custom_sonata_config=None,
                 device="cuda"):
        super().__init__()
        
        self.device = device
        self.clip_model_name = clip_model_name
        
        # Load Sonata backbone
        print(f"Loading Sonata model: {sonata_model_name}...")
        try:
            import flash_attn
            self.sonata_model = sonata.load(sonata_model_name, repo_id=sonata_repo_id)
        except ImportError:
            print("FlashAttention not available, using standard attention...")
            if custom_sonata_config is None:
                custom_sonata_config = dict(
                    enc_patch_size=[1024 for _ in range(5)],
                    enable_flash=False,
                )
            self.sonata_model = sonata.load(
                sonata_model_name, 
                repo_id=sonata_repo_id, 
                custom_config=custom_sonata_config
            )
        
        self.sonata_model.to(device)
        self.sonata_model.requires_grad_(not freeze_sonata)
        if freeze_sonata:
            self.sonata_model.eval()
        
        # Get Sonata feature dimension
        # Assuming output feature dimension is 512 (standard for Sonata)
        self.sonata_feat_dim = 512
        
        # Load CLIP text encoder
        print(f"Loading CLIP model: {clip_model_name}...")
        self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model.requires_grad_(not freeze_clip)
        if freeze_clip:
            self.clip_model.eval()
        
        # Projection layer
        print("Initializing projection layer...")
        if proj_config is None:
            proj_config = {
                'sonata_embed_dim': self.sonata_feat_dim,
                'clip_embed_dim': 512,  # ViT-B/16
                'hidden_layer': True,
                'act': 'tanh',
                'cosine': True
            }
        self.proj = SonataProjectionLayer.from_config(proj_config)
        self.proj.to(device)
        
        # Masker for 3D point-wise classification
        self.masker = SonataTextMasker(similarity_type="cosine", temperature=1.0)
        self.masker.eval()
        
        # Transform pipeline for preprocessing
        self.transform = sonata.transform.default()
        
        print("SonataText model initialized successfully!")
    
    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames):
        """
        Build text tokens for a set of class names using templates
        
        Args:
            template_set: name of template set (e.g., 'imagenet_template')
            classnames: list of class names
        
        Returns:
            tokens: (num_classes, num_templates, seq_len)
        """
        tokens = []
        templates = get_template(template_set)
        
        for classname in classnames:
            tokens.append(
                clip.tokenize([template.format(classname) for template in templates])
            )
        
        tokens = torch.stack(tokens)
        return tokens
    
    @torch.no_grad()
    def build_text_embedding(self, text_tokens):
        """
        Encode text descriptions and project to Sonata space
        
        Args:
            text_tokens: (num_classes, num_templates, seq_len)
        
        Returns:
            text_emb: (num_classes, embed_dim) - Projected text embeddings
        """
        text_tokens = text_tokens.to(self.device)
        num_classes, num_templates = text_tokens.shape[:2]
        
        # Flatten: (num_classes, num_templates, seq_len) → (N, seq_len)
        text_tokens_flat = text_tokens.reshape(-1, text_tokens.shape[-1])
        
        # Encode with CLIP (chunked for memory efficiency)
        chunk_size = 32
        N = text_tokens_flat.size(0)
        text_features = torch.cat([
            self.clip_model.encode_text(text_tokens_flat[i:i + chunk_size])
            for i in range(0, N, chunk_size)
        ])
        
        # Reshape and average templates: (N, C) → (num_classes, num_templates, C) → (num_classes, C)
        text_features = text_features.reshape(num_classes, num_templates, -1)
        text_features = text_features.mean(dim=1).float()
        
        # Project to Sonata space
        _, text_emb = self.proj(None, text_features)
        
        return text_emb
    
    @torch.no_grad()
    def generate_masks(self, point_dict, text_emb, classnames, return_features=False):
        """
        Generate 3D segmentation predictions for point cloud
        
        Args:
            point_dict: dict with 'coord', 'color', 'normal', etc.
            text_emb: (K, C) - Text embeddings for K classes
            classnames: list of class names (for reference)
            return_features: if True, also return point features
        
        Returns:
            predictions: (N, K) - Per-point class scores
            simmap: (N, K) - Similarity scores
            point_feat: (N, C) - Point features (if return_features=True)
        """
        # Preprocess point cloud
        point = self.transform(point_dict)
        
        # Move to device
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(self.device, non_blocking=True)
        
        # Extract Sonata features
        with torch.inference_mode():
            point = self.sonata_model(point)
        
        # Get point features
        point_feat = point.feat  # (N, C)
        
        # Project point features to aligned space
        point_feat, _ = self.proj(point_feat, None)
        
        # Compute per-point class predictions
        predictions, simmap = self.masker.forward_seg(point_feat, text_emb)
        
        if return_features:
            return predictions, simmap, point_feat
        else:
            return predictions, simmap
    
    def forward(self, point_dict, text_tokens=None, classnames=None):
        """
        Full forward pass for training or inference
        
        Args:
            point_dict: point cloud data
            text_tokens: optional text tokens
            classnames: optional class names
        
        Returns:
            predictions, simmap, text_emb
        """
        if text_tokens is not None:
            text_emb = self.build_text_embedding(text_tokens)
        elif classnames is not None:
            text_tokens = self.build_dataset_class_tokens('imagenet_template', classnames)
            text_emb = self.build_text_embedding(text_tokens)
        else:
            raise ValueError("Either text_tokens or classnames must be provided")
        
        predictions, simmap = self.generate_masks(point_dict, text_emb, classnames or [])
        
        return predictions, simmap, text_emb


# Builder function for model registry
def build_sonatatext(cfg):
    """Build SonataText model from config"""
    return SonataText(
        sonata_model_name=cfg.get('sonata_model_name', 'sonata'),
        sonata_repo_id=cfg.get('sonata_repo_id', 'facebook/sonata'),
        clip_model_name=cfg.get('clip_model_name', 'ViT-B/16'),
        proj_config=cfg.get('projection', None),
        freeze_sonata=cfg.get('freeze_sonata', True),
        freeze_clip=cfg.get('freeze_clip', True),
        device=cfg.get('device', 'cuda')
    )
