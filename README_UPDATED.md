# Talk2Sonata: Open-Vocabulary 3D Point Cloud Segmentation

## Overview
Talk2Sonata combines **Sonata** (self-supervised 3D point cloud backbone) with **CLIP** (text encoder) to achieve open-vocabulary semantic segmentation for 3D point clouds, inspired by the Talk2DINO architecture.

## Architecture

```
Point Cloud → Sonata Encoder → 3D Point Features
                                       ↓
                          Projection Layer ← CLIP Text Encoder
                                       ↓
                            Similarity Matching
                                       ↓
                         Per-Point Class Predictions
```

## Key Features
- **Open-Vocabulary**: Segment 3D scenes with arbitrary text descriptions
- **No Training on Target Classes**: Zero-shot segmentation capabilities
- **Flexible**: Works with any 3D point cloud dataset
- **Based on Proven Architectures**: Leverages Sonata (PTv3) and CLIP

## Installation

### 1. Environment Setup
```bash
# Create conda environment
conda create -n talk2sonata python=3.10
conda activate talk2sonata

# Install PyTorch (adjust CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install Talk2Sonata requirements
pip install -r requirements_3d.txt

# Install spconv (for 3D sparse convolutions)
pip install spconv-cu118  # Adjust CUDA version

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Optional: Flash Attention for faster inference
pip install flash-attn
```

### 2. Verify Installation
```bash
python -c "import torch; import sonata; import clip; print('✓ All dependencies loaded')"
```

## Quick Start

### 1. Download Sample Data
```python
import sonata
point = sonata.data.load("sample1")
```

### 2. Run 3D Segmentation Demo
```bash
python demo_3d.py \
    --input data/sample1.npz \
    --output results/scene_seg.ply \
    --textual_categories "chair,table,floor,wall,door,window" \
    --visualize
```

### 3. Use in Your Code
```python
import torch
import sonata
from src.open_vocabulary_segmentation.models.sonatatext import SonataText

# Initialize model
model = SonataText(
    sonata_model_name="sonata",
    clip_model_name="ViT-B/16",
    device="cuda"
)
model.eval()

# Load point cloud
point = sonata.data.load("sample1")

# Define classes
classnames = ["chair", "table", "floor", "wall"]

# Build text embeddings
text_tokens = model.build_dataset_class_tokens("imagenet_template", classnames)
text_emb = model.build_text_embedding(text_tokens)

# Run inference
with torch.no_grad():
    predictions, simmap = model.generate_masks(point, text_emb, classnames)

# Get labels
labels = predictions.argmax(dim=-1)
```

## Model Components

### 1. Sonata Backbone
- **Architecture**: Point Transformer V3 (PTv3)
- **Pretraining**: Self-supervised on large-scale 3D data
- **Output**: 512-dim point features

### 2. CLIP Text Encoder
- **Models**: ViT-B/16 (default) or ViT-L/14
- **Output**: 512-dim or 768-dim text features

### 3. Projection Layer
- **Purpose**: Aligns CLIP text space with Sonata feature space
- **Architecture**: MLP with tanh activation
- **Training**: Contrastive learning (InfoNCE loss)

### 4. 3D Masker
- **Function**: Computes point-text similarity
- **Output**: Per-point class predictions

## Configuration

Edit `configs/sonatatext_vitb.yaml` to customize:

```yaml
model:
  sonata_model_name: sonata  # or sonata_small
  clip_model_name: ViT-B/16  # or ViT-L/14
  freeze_sonata: true        # Freeze during training
  freeze_clip: true
  
  projection:
    sonata_embed_dim: 512
    clip_embed_dim: 512
    hidden_layer: true
    act: tanh
```

## Training (Coming Soon)

To train the projection layer on your own data:

```bash
python train_3d.py \
    --config configs/sonatatext_vitb.yaml \
    --dataset scannet \
    --output_dir checkpoints/
```

### Training Data Format
Point cloud-text pairs:
```python
{
    'coord': np.array,    # (N, 3)
    'color': np.array,    # (N, 3)
    'normal': np.array,   # (N, 3)
    'caption': str        # Text description
}
```

## Datasets

Compatible with:
- **ScanNet**: Indoor scene understanding
- **S3DIS**: Stanford indoor spaces
- **ScanRefer**: 3D referring expressions
- **Custom datasets**: Any point cloud with coordinates + features

## Comparison with Talk2DINO

| Aspect | Talk2DINO (2D) | Talk2Sonata (3D) |
|--------|----------------|------------------|
| Backbone | DINOv2 (ViT) | Sonata (PTv3) |
| Input | RGB Images | Point Clouds |
| Features | 2D Patches | 3D Points |
| Output | 2D Masks | 3D Labels |
| Attention | Image patches | Serialized 3D patches |

## Project Structure

```
Talk2Sonata/
├── sonata/                           # Sonata backbone (copied)
├── src/
│   ├── model.py                      # Projection layers
│   └── open_vocabulary_segmentation/
│       └── models/
│           └── sonatatext/           # Main model
│               ├── __init__.py
│               └── sonatatext.py     # SonataText class
├── configs/
│   └── sonatatext_vitb.yaml         # Model configuration
├── demo_3d.py                       # 3D inference demo
├── train_3d.py                      # Training script (TODO)
├── requirements_3d.txt              # Dependencies
├── INTEGRATION_GUIDE.md             # Technical details
└── README.md                        # This file
```

## Citation

If you use this code, please cite:

```bibtex
@article{talk2dino,
  title={Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation},
  author={...},
  journal={ICCV},
  year={2025}
}

@article{sonata,
  title={Sonata: Self-Supervised Learning of Reliable Point Representations},
  author={...},
  journal={CVPR},
  year={2025}
}
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `num_points` in config
- Use smaller Sonata model: `sonata_small`
- Process point cloud in chunks

### Missing Dependencies
```bash
# Re-install core dependencies
pip install torch torchvision
pip install spconv-cu118
pip install torch-scatter
```

### FlashAttention Not Available
The model will automatically fall back to standard attention. Performance may be slower but functionality is preserved.

## License

This project combines code from:
- **Sonata**: Apache 2.0 License (Meta Platforms)
- **Talk2DINO**: Check original repository
- **CLIP**: MIT License (OpenAI)

## Acknowledgments

- **Talk2DINO** for the open-vocabulary segmentation approach
- **Sonata** for the powerful 3D backbone
- **CLIP** for bridging vision and language

## Next Steps

1. ✅ **Architecture designed and implemented**
2. ✅ **Demo script created**
3. 🔄 **Collect 3D training data (point clouds + text)**
4. 🔄 **Train projection layer**
5. 🔄 **Evaluate on benchmark datasets**
6. 🔄 **Release pretrained weights**

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Status**: 🚧 Under Active Development
