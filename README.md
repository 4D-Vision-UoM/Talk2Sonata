# Talk2Sonata: Open-Vocabulary 3D Segmentation

A hierarchical open-vocabulary 3D point cloud segmentation framework built on the Sonata backbone and CLIP text encoder. This project enables natural language-guided semantic segmentation of ScanNet scenes using sparse hierarchical features.

## Features

- **Open-Vocabulary Segmentation**: Query 3D scenes using natural language descriptions
- **Hierarchical Deep Supervision**: Multi-stage feature learning with weighted loss aggregation
- **ScanNet20 & ScanNet200 Support**: Train on either 20-class or 200-class ScanNet datasets
- **Cached Training Pipeline**: Pre-compute Sonata features for efficient training
- **Interactive Web Viewer**: Browser-based 3D point cloud visualization with split-view comparison
- **Feature Visualization**: PCA/UMAP analysis of hierarchical stage embeddings

## Architecture

```
Text Query (CLIP) → Projection Layers → Sparse Hierarchical Search
                                              ↓
ScanNet Scene → Sonata Encoder → Stage Features [S0...S4]
                                              ↓
                                    Deep Supervision Loss
```

### Key Components

- **Sonata Backbone**: 5-stage hierarchical encoder with patch sizes [1024, 1024, 1024, 1024, 1024]
- **CLIP Text Encoder**: ViT-L/14 for text feature extraction
- **Projection Layers**: Per-stage MLP networks (3-layer with GELU, LayerNorm)
- **Learnable Temperature**: Logit scale parameter for similarity calibration
- **Combined Loss**: Focal Loss (α=0.75, γ=2.0) + Dice Loss

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy
open3d
scikit-learn
umap-learn
clip  # OpenAI CLIP
tqdm
tensorboard
PyYAML

# Install Sonata (from Facebook Research)
# Follow instructions at: https://github.com/facebookresearch/sonata
```

### Setup

```bash
git clone https://github.com/4D-Vision-UoM/Talk2Sonata.git
cd Talk2Sonata
pip install -r requirements.txt
```

## Dataset Preparation

### ScanNet Dataset Structure

Place ScanNet data in the following structure:

```
data/scannet_data/
├── train/
│   ├── scene0000_00/
│   │   ├── coord.npy          # Point coordinates (N, 3)
│   │   ├── color.npy          # RGB colors (N, 3)
│   │   ├── normal.npy         # Surface normals (N, 3)
│   │   ├── segment20.npy      # 20-class labels (N,)
│   │   └── segment200.npy     # 200-class labels (N,)
│   └── ...
└── val/
    └── ...
```

### Label Configuration

**ScanNet20**: 20 common indoor object classes  
**ScanNet200**: 200 fine-grained classes with exclusions:
- Wall (ID: 0)
- Floor (ID: 2)  
- Ceiling (ID: 35)

**Important**: `segment200.npy` contains pre-processed labels in range [0, 199], not NYU40 IDs.

## Usage

### 1. Feature Caching (Recommended)

Pre-compute Sonata features to speed up training:

```bash
# Cache validation set
python scannet_cache.py

# Edit CONFIG in scannet_cache.py for training set:
# 'data_root': 'data/scannet_data/train'
# 'output_dir': 'data/scannet_cache_train'
python scannet_cache.py
```

**Cache Contents**:
- Stage embeddings (S0-S4)
- Stage coordinates  
- Pooling inverse mappings
- Dense segment labels (both segment20 and segment200)
- Text templates and class IDs

### 2. Training

```bash
python talk2sonata_train.py
```

**Configuration** (`CONFIG` in `talk2sonata_train.py`):
```python
CONFIG = {
    'train_data_root': 'data/scannet_cache_train',
    'val_data_root': 'data/scannet_cache_val',
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 100,
    'use_scannet200': False,  # Set True for 200-class training
    'stage_loss_weights': [1.0, 0.8, 0.4, 0.2, 0.1]  # S0 → S4
}
```

**Training Features**:
- Deep supervision across all stages
- Learnable temperature scaling
- Combined Focal + Dice loss
- Best & random scene visualization per epoch
- TensorBoard logging
- Early stopping with patience

**Outputs**:
- `outputs/ov_sonata_seg200/best_model.pth` - Best checkpoint
- `outputs/ov_sonata_seg200/visualizations/` - Prediction visualizations
- `outputs/ov_sonata_seg200/logs/` - TensorBoard logs

### 3. Visualization

#### Feature Analysis

```bash
python visualize_sonata.py
```

Generates PCA-colored point clouds for each Sonata stage:
- `stage_X_sparse_pca.pcd` - Sparse features at native resolution
- `stage_X_projected_to_dense_pca.pcd` - Features upsampled to dense resolution

#### Evaluation & Visualization Scripts

```bash
# Stage-wise PCA analysis (saves original RGB + class highlighting)
python evaluvations/stage_pca.py

# Text-image similarity analysis
python evaluvations/similarity.py

# Semantic segmentation evaluation
python evaluvations/2_sem_seg.py
```

#### Web Viewer

Launch the interactive 3D viewer:

```bash
python serve_viewer.py
```

Navigate to `http://localhost:8000/viewer.html`

**Features**:
- Split-view comparison (GT vs Prediction, Sparse vs Dense)
- Auto-pairing of related files
- Synchronized camera controls
- Point size adjustment
- File browser with folder navigation

## Project Structure

```
Talk2Sonata/
├── sonata/                      # Sonata model integration
│   ├── scannet_text.py         # Text-guided ScanNet dataset
│   ├── scannet_labels.py       # Label definitions & exclusions
│   └── ...
├── scannet_cache.py            # Feature pre-computation
├── talk2sonata_train.py        # Training script
├── visualize_sonata.py         # Feature visualization
├── evaluvations/               # Evaluation scripts
│   ├── stage_pca.py           # PCA feature analysis
│   ├── similarity.py          # Text-image similarity
│   └── 2_sem_seg.py           # Semantic segmentation
├── viewer.html                 # Web-based 3D viewer
├── serve_viewer.py            # HTTP server for viewer
├── configs/                    # Model configurations
└── weights/                    # Pre-trained checkpoints
```

## Key Scripts

### `scannet_cache.py`
Pre-computes Sonata features and saves them to disk for faster training.

### `talk2sonata_train.py`
Main training script with:
- Hierarchical sparse search
- Deep supervision (5 stages)
- Text-guided segmentation
- Automatic visualization

### `visualize_sonata.py`
Visualizes Sonata stage features using PCA dimensionality reduction.

### `evaluvations/stage_pca.py`
Analyzes features at each stage, saves:
- Original RGB point cloud
- Random class highlighted (red on gray)
- PCA-colored sparse features
- PCA-colored dense features

### `viewer.html` + `serve_viewer.py`
Interactive web viewer for 3D point clouds with synchronized dual viewports.

## Coordinate System

**Important**: The project uses Y-Z swapped coordinates to align the floor with the XZ plane (Y-up convention).

All visualization scripts automatically apply this transformation:
```python
coords_aligned[:, [1, 2]] = coords[:, [2, 1]]  # Swap Y and Z
```

## Model Weights

Pre-trained Sonata weights are located in `weights/`:
- `vitb_mlp_infonce.pth`
- `vitl_mlp_infonce.pth`
- `dinov3_vitb_mlp_infonce.pth`
- `dinov3_vitl_mlp_infonce.pth`

## Configuration Files

YAML configs in `configs/` specify model architecture:
```yaml
model:
  enc_patch_size: [1024, 1024, 1024, 1024, 1024]
  enable_flash: false
  enc_mode: true
  freeze_encoder: true
```

## Training Tips

1. **Start with ScanNet20**: Easier to train and debug
2. **Monitor Temperature**: Check `Temp` in TensorBoard (should converge around 10-30)
3. **Validate Coordinates**: Ensure floor is horizontal in saved visualizations
4. **Cache First**: Always regenerate cache after modifying dataset code
5. **Stage Weights**: Higher weights on S0 for better dense predictions

## Troubleshooting

### "Cache missing segment200 targets"
Re-run `scannet_cache.py` with updated code that caches both segment types.

### "No valid objects after exclusion"
Scene only contains wall/floor/ceiling. This is normal for some ScanNet scenes.

### Vertical point clouds in viewer
Coordinate swap not applied. Check visualization code for Y-Z swap.

### Poor segmentation quality
- Verify labels are correct (use `verify_segment200_labels.py`)
- Check temperature scaling (should be trainable)
- Increase training epochs or adjust loss weights

## Citation

If you use this code, please cite:

```bibtex
@misc{talk2sonata2024,
  title={Talk2Sonata: Open-Vocabulary 3D Segmentation},
  author={University of Moratuwa 4D Vision Lab},
  year={2024}
}
```

## License

This project is released under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- [Sonata](https://github.com/facebookresearch/sonata) - Facebook Research
- [CLIP](https://github.com/openai/CLIP) - OpenAI
- [ScanNet](http://www.scan-net.org/) - Dataset
- [Open3D](http://www.open3d.org/) - 3D visualization
- [Three.js](https://threejs.org/) - WebGL rendering

## Contact

For questions or issues, please open an issue on GitHub or contact the 4D Vision Lab at University of Moratuwa.
