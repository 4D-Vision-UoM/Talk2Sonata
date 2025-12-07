"""
Demo script for Talk2Sonata - 3D Open-Vocabulary Segmentation

Usage:
    python demo_3d.py --input data/sample_scene.npz \
                      --output results/scene_seg.ply \
                      --textual_categories "chair,table,floor,wall"
"""

import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add sonata to path
sys.path.insert(0, "sonata")
sys.path.insert(0, "src/open_vocabulary_segmentation")

import sonata
from models.sonatatext import SonataText

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Warning: open3d not installed. Visualization will be disabled.")
    HAS_OPEN3D = False


def generate_colors(num_classes):
    """Generate distinct colors for each class"""
    colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [0, 255, 128],    # Spring Green
        [255, 192, 203],  # Pink
    ]
    
    # Generate random colors if needed
    while len(colors) < num_classes:
        colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    return np.array(colors[:num_classes]) / 255.0


def save_colored_point_cloud(coords, labels, colors, output_path):
    """Save point cloud with colored segments"""
    if not HAS_OPEN3D:
        print(f"Cannot save visualization without open3d. Saving raw labels to {output_path}.npy")
        np.save(f"{output_path}.npy", labels)
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # Color by labels
    point_colors = colors[labels]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Save
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved colored point cloud to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo for Talk2Sonata - 3D Open-Vocabulary Segmentation")
    parser.add_argument("--input", type=str, default="data/sample1.npz", 
                        help="Input point cloud file (.npz or .npy format)")
    parser.add_argument("--output", type=str, default="results/scene_seg.ply",
                        help="Output file path for segmented point cloud")
    parser.add_argument("--textual_categories", type=str, default="chair,table,floor,wall",
                        help="Comma-separated list of object categories")
    parser.add_argument("--config", type=str, default="configs/sonatatext_vitb.yaml",
                        help="Model configuration file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the result interactively (requires open3d)")
    
    args = parser.parse_args()
    
    # Parse class names
    classnames = [name.strip().replace("_", " ") for name in args.textual_categories.split(",")]
    print(f"Classes: {classnames}")
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. This may be slow.")
    
    # Load model
    print("Loading SonataText model...")
    model = SonataText(
        sonata_model_name="sonata",
        sonata_repo_id="facebook/sonata",
        clip_model_name="ViT-B/16",
        freeze_sonata=True,
        freeze_clip=True,
        device=device
    )
    model.eval()
    
    # Load point cloud data
    print(f"Loading point cloud from {args.input}...")
    if args.input.endswith('.npz'):
        data = dict(np.load(args.input))
    elif args.input.endswith('.npy'):
        # Assume format: (N, 6) with [x, y, z, r, g, b]
        points = np.load(args.input)
        data = {
            'coord': points[:, :3],
            'color': points[:, 3:6] if points.shape[1] >= 6 else np.ones((points.shape[0], 3)) * 0.5,
        }
    else:
        raise ValueError(f"Unsupported file format: {args.input}")
    
    # Ensure required keys
    if 'coord' not in data:
        raise ValueError("Point cloud must have 'coord' key")
    
    # Add default values if missing
    if 'color' not in data:
        data['color'] = np.ones((data['coord'].shape[0], 3)) * 0.5
    if 'normal' not in data:
        data['normal'] = np.zeros((data['coord'].shape[0], 3))
    
    print(f"Point cloud loaded: {data['coord'].shape[0]} points")
    
    # Store original coordinates for visualization
    original_coords = data['coord'].copy()
    
    # Build text embeddings
    print("Building text embeddings...")
    with torch.no_grad():
        text_tokens = model.build_dataset_class_tokens("imagenet_template", classnames)
        text_emb = model.build_text_embedding(text_tokens)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions, simmap = model.generate_masks(data, text_emb, classnames)
    
    # Get hard labels
    labels = predictions.argmax(dim=-1).cpu().numpy()
    
    # Get prediction confidence
    confidence = predictions.softmax(dim=-1).max(dim=-1)[0].cpu().numpy()
    
    # Print statistics
    print("\nSegmentation Results:")
    print("-" * 50)
    for i, classname in enumerate(classnames):
        count = (labels == i).sum()
        percentage = (count / len(labels)) * 100
        avg_conf = confidence[labels == i].mean() if count > 0 else 0
        print(f"{classname:20s}: {count:6d} points ({percentage:5.2f}%) - Avg Conf: {avg_conf:.3f}")
    print("-" * 50)
    
    # Generate colors
    class_colors = generate_colors(len(classnames))
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Save results
    save_colored_point_cloud(original_coords, labels, class_colors, args.output)
    
    # Save raw predictions
    output_dir = Path(args.output).parent
    np.save(output_dir / "predictions.npy", predictions.cpu().numpy())
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "confidence.npy", confidence)
    print(f"Saved raw predictions to {output_dir}")
    
    # Visualize if requested
    if args.visualize and HAS_OPEN3D:
        print("\nVisualizing result...")
        pcd = o3d.io.read_point_cloud(args.output)
        
        # Create legend text
        legend = "Classes:\n"
        for i, classname in enumerate(classnames):
            color_str = f"RGB({int(class_colors[i][0]*255)}, {int(class_colors[i][1]*255)}, {int(class_colors[i][2]*255)})"
            legend += f"  {classname}: {color_str}\n"
        print(legend)
        
        o3d.visualization.draw_geometries([pcd], 
                                         window_name="Talk2Sonata - 3D Segmentation Result")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
