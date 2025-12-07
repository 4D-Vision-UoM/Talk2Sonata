# Talk2Sonata Integration Guide

## Overview
This document outlines the integration strategy for combining Sonata (3D point cloud backbone) with CLIP (text encoder) to achieve open-vocabulary 3D segmentation, inspired by Talk2DINO's approach.

## Architecture Comparison

### Talk2DINO (2D)
```
Image → DINOv2 → 2D Patch Features → Projection Layer ← CLIP Text
                                          ↓
                                   Similarity Matching
                                          ↓
                                   2D Segmentation Mask
```

### Talk2Sonata (3D) - Target Architecture
```
Point Cloud → Sonata → 3D Point Features → Projection Layer ← CLIP Text
                                               ↓
                                        Similarity Matching
                                               ↓
                                        3D Point-wise Labels
```

## Key Components to Adapt

### 1. Sonata Feature Extractor
**Location:** `sonata/model.py`

**Current Usage in Sonata:**
```python
import sonata
model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
point = transform(point)  # preprocessing
point = model(point)  # forward pass
# Output: point.feat (N x 512 or similar dimension)
```

**Integration Strategy:**
- Use Sonata as frozen backbone (similar to DINOv2 in Talk2DINO)
- Extract point features before the final classification head
- These features will be aligned with CLIP text embeddings

### 2. CLIP Text Encoder
**Location:** Already in Talk2DINO - reuse directly

**No changes needed:**
```python
import clip
clip_model, _ = clip.load("ViT-B/16", device=device)
text_features = clip_model.encode_text(tokenized_text)
```

### 3. Projection Layer Adaptation
**Location:** `src/model.py` - modify for 3D

**Changes Required:**
```python
class SonataProjectionLayer(nn.Module):
    """
    Maps CLIP text embeddings to Sonata point feature space
    """
    def __init__(self, 
                 sonata_embed_dim=512,  # Sonata output dimension
                 clip_embed_dim=512,     # CLIP text dimension
                 hidden_layer=True,
                 act=nn.Tanh(),
                 cosine=True):
        super().__init__()
        self.linear_layer = nn.Linear(clip_embed_dim, sonata_embed_dim)
        if hidden_layer:
            self.linear_layer2 = nn.Linear(sonata_embed_dim, sonata_embed_dim)
        self.act = act
        self.cosine = cosine
    
    def forward(self, point_embedding, textual_embedding):
        # point_embedding: (N, sonata_dim) - from Sonata
        # textual_embedding: (K, clip_dim) - from CLIP
        
        # Project text to Sonata space
        text_projected = self.linear_layer(textual_embedding)
        if self.act:
            text_projected = self.act(text_projected)
        if hasattr(self, 'linear_layer2'):
            text_projected = self.linear_layer2(text_projected)
        
        # Normalize for cosine similarity
        if self.cosine:
            point_embedding = F.normalize(point_embedding, p=2, dim=1)
            text_projected = F.normalize(text_projected, p=2, dim=1)
        
        return point_embedding, text_projected
```

### 4. 3D Masker (Point-wise Classification)
**Location:** Create new `src/open_vocabulary_segmentation/models/sonatatext/masker.py`

**New Component:**
```python
class SonataTextMasker(nn.Module):
    """
    Generates per-point class predictions for 3D point clouds
    """
    def __init__(self, similarity_type="cosine"):
        super().__init__()
        self.similarity_type = similarity_type
    
    def forward_seg(self, point_feat, text_emb):
        """
        Args:
            point_feat: (N, C) - Sonata point features
            text_emb: (K, C) - Projected text embeddings (K classes)
        
        Returns:
            predictions: (N, K) - Per-point class scores
            simmap: (N, K) - Similarity scores
        """
        # Compute similarity: (N, C) @ (K, C).T → (N, K)
        simmap = point_feat @ text_emb.T
        
        # Optional: temperature scaling
        predictions = simmap  # Can apply softmax later
        
        return predictions, simmap
```

### 5. Main Model Class
**Location:** Create `src/open_vocabulary_segmentation/models/sonatatext/sonatatext.py`

**New Model:**
```python
class SonataText(nn.Module):
    """
    Open-vocabulary 3D segmentation model combining Sonata + CLIP
    """
    def __init__(self, 
                 sonata_model_name="sonata",
                 clip_model_name="ViT-B/16",
                 proj_config=None,
                 freeze_backbones=True):
        super().__init__()
        
        # Load Sonata backbone
        self.sonata_model = sonata.load(sonata_model_name, 
                                        repo_id="facebook/sonata")
        self.sonata_model.requires_grad_(not freeze_backbones)
        
        # Load CLIP text encoder
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.requires_grad_(not freeze_backbones)
        
        # Projection layer
        self.proj = SonataProjectionLayer.from_config(proj_config)
        
        # Masker for 3D
        self.masker = SonataTextMasker(similarity_type="cosine")
        
        # Transform pipeline
        self.transform = sonata.transform.default()
    
    @torch.no_grad()
    def build_text_embedding(self, text_tokens):
        """
        Encode text descriptions
        Args:
            text_tokens: (num_classes, num_templates, seq_len)
        Returns:
            text_emb: (num_classes, embed_dim)
        """
        num_classes, num_templates = text_tokens.shape[:2]
        text_tokens = text_tokens.reshape(-1, text_tokens.shape[-1])
        
        # Encode with CLIP
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features.reshape(num_classes, num_templates, -1)
        text_features = text_features.mean(dim=1)  # Average templates
        
        # Project to Sonata space
        _, text_emb = self.proj(None, text_features)
        
        return text_emb
    
    @torch.no_grad()
    def generate_masks(self, point_dict, text_emb, classnames):
        """
        Generate 3D segmentation for point cloud
        Args:
            point_dict: dict with 'coord', 'color', 'normal'
            text_emb: (K, C) - Text embeddings for K classes
            classnames: list of class names
        
        Returns:
            predictions: (N, K) - Per-point class predictions
        """
        # Preprocess point cloud
        point = self.transform(point_dict)
        
        # Move to GPU
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda()
        
        # Extract Sonata features
        point = self.sonata_model(point)
        point_feat = point.feat  # (N, C)
        
        # Project features
        point_feat, _ = self.proj(point_feat, None)
        
        # Compute similarities
        predictions, simmap = self.masker.forward_seg(point_feat, text_emb)
        
        return predictions, simmap
```

## Required Modifications

### File Structure
```
Talk2Sonata/
├── sonata/                      # ✅ Already copied
├── src/
│   ├── model.py                 # ✅ Has ProjectionLayer (adapt for 3D)
│   ├── open_vocabulary_segmentation/
│   │   ├── models/
│   │   │   ├── sonatatext/      # ⚠️ CREATE NEW
│   │   │   │   ├── __init__.py
│   │   │   │   ├── sonatatext.py
│   │   │   │   └── masker.py
│   │   │   └── builder.py       # ✅ Update to include SonataText
│   │   └── configs/
│   │       └── sonatatext/      # ⚠️ CREATE NEW CONFIG FILES
├── configs/
│   └── sonatatext_vitb.yaml     # ⚠️ CREATE NEW
├── demo_3d.py                   # ⚠️ CREATE NEW (3D demo)
└── train_3d.py                  # ⚠️ CREATE NEW (3D training)
```

## Configuration File Example

**configs/sonatatext_vitb.yaml:**
```yaml
model:
  type: SonataText
  sonata_model_name: sonata
  clip_model_name: ViT-B/16
  freeze_backbones: true
  
  projection:
    sonata_embed_dim: 512
    clip_embed_dim: 512
    hidden_layer: true
    act: tanh
    cosine: true

train:
  lr: 0.0001
  batch_size: 32
  num_epochs: 100
  loss_type: infonce

data:
  dataset: scannet  # or other 3D dataset
  num_points: 8192
```

## Next Steps

1. ✅ **Understand existing codebases** (DONE)
2. 🔄 **Create SonataText model class**
3. 🔄 **Create 3D masker**
4. 🔄 **Create demo script for 3D inference**
5. 🔄 **Prepare 3D training data (point clouds + text)**
6. 🔄 **Training pipeline with contrastive loss**
7. 🔄 **Evaluation on 3D semantic segmentation benchmarks**

## Dataset Requirements

For training Talk2Sonata, you need:
- **3D Point Clouds:** ScanNet, S3DIS, or custom datasets
- **Text Annotations:** Scene descriptions, object labels
- **Pairing:** Point cloud regions matched with text descriptions

Common datasets:
- **ScanNet:** Indoor scenes with semantic labels
- **S3DIS:** Stanford indoor spaces
- **Refer3D:** 3D referring expression dataset
- **ScanRefer:** Referring expressions in 3D scenes

## Training Strategy

1. **Stage 1: Projection Layer Training**
   - Freeze Sonata and CLIP
   - Train only projection layer with contrastive loss
   - Use point cloud-text pairs

2. **Stage 2: Fine-tuning (Optional)**
   - Fine-tune Sonata encoder with projection layer
   - Keep CLIP frozen

3. **Evaluation:**
   - Zero-shot 3D semantic segmentation
   - Open-vocabulary object detection in 3D
