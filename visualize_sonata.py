"""
Visualization script for Sonata model features using PCA and Open3D.
This script loads a ScanNet dataset sample, processes it through the Sonata model,
and visualizes the stage embeddings as point clouds colored by PCA-reduced features.
"""

import os
import numpy as np
import torch
import open3d as o3d
from sklearn.decomposition import PCA

from sonata.scannet import ScanNetDataset
import sonata


# Configuration
SEED = 53124
DATA_PATH = 'data/scannet_data/val'
OUTPUT_DIR = "demo_outputs/stage_feature_visualizations"
CUSTOM_CONFIG = dict(
    enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
    enable_flash=False,
    enc_mode=True,
    freeze_encoder=False,
)


def compute_pca_colors(features):
    """
    Compresses high-dim features (N, D) to (N, 3) RGB using PCA.
    Colors are normalized min-max per channel to fit 0-1 range.

    Args:
        features: High-dimensional features, either torch.Tensor or numpy array.

    Returns:
        numpy array: RGB colors normalized to [0, 1].
    """
    # Convert to numpy if necessary
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    # Fit PCA to get top 3 principal components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)

    # Normalize each channel to [0, 1] for RGB visualization
    for i in range(3):
        v_min = pca_features[:, i].min()
        v_max = pca_features[:, i].max()
        if v_max > v_min:
            pca_features[:, i] = (pca_features[:, i] - v_min) / (v_max - v_min)

    return pca_features


import umap
import numpy as np
import open3d as o3d

def compute_umap_colors(features, n_neighbors=15, min_dist=0.1):
    """
    Reduces features to 3D using UMAP for better cluster separation.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
        
    print("Fitting UMAP... (this may take a moment)")
    # n_neighbors: controls how local the focus is (low = distinct parts, high = global shape)
    # min_dist: controls how tightly points are packed
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=42,
        n_jobs=1,  # Explicitly set to avoid warning
        init='random'  # Use random init to avoid spectral failures
    )
    embedding = reducer.fit_transform(features)
    
    # Normalize to 0-1 for RGB
    for i in range(3):
        v_min, v_max = embedding[:, i].min(), embedding[:, i].max()
        if v_max > v_min:
            embedding[:, i] = (embedding[:, i] - v_min) / (v_max - v_min)
        
    return embedding

# Usage in your loop:
# 


def save_pcd(coords, colors, filename):
    """
    Saves coordinates and colors as a PCD file using Open3D.

    Args:
        coords: numpy array of shape (N, 3) for point coordinates.
        colors: numpy array of shape (N, 3) for RGB colors.
        filename: Output filename.
    """
    # Fix coordinate alignment - swap Y and Z to put floor on XZ plane
    coords_aligned = coords.copy()
    coords_aligned[:, [1, 2]] = coords_aligned[:, [2, 1]]  # Swap Y and Z
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_aligned)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved: {filename}")


def main():
    """Main execution function."""
    # Set seed for reproducibility
    sonata.utils.set_seed(SEED)

    # Load dataset
    dataset = ScanNetDataset(data_root=DATA_PATH)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[2]

    # Load model
    model = sonata.load(
        "sonata", repo_id="facebook/sonata", custom_config=CUSTOM_CONFIG
    ).cuda()

    # Load default data transform pipeline
    transform = sonata.transform.default()  # voxelize and downsample points

    # Prepare point data
    point = sample.copy()
    # Handle segment data: use segment20 instead of segment200
    if "segment20" in point:
        segment = point.pop("segment20")
        point["segment"] = segment
    original_coord = point["coord"].copy()
    original_point = point.copy()

    # Apply transform
    point = transform(point)

    # Move tensors to GPU
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        # Model forward pass
        point = model(point)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract stage data
    stage_embeddings = point['stage_embeddings']
    stage_coords = point['stage_coords']
    pooling_inverses = point['stage_pooling_inverses']

    # Visualize deeper stages projected to original resolution
    print("\n--- Saving Dense (Projected) Feature Visualizations ---")
    coord_s0 = stage_coords[0].detach().cpu().numpy()

    for target_stage in range(1, len(stage_embeddings)):
        curr_feats = stage_embeddings[target_stage]

        # Upsample features to stage 0 resolution
        for k in range(target_stage, 0, -1):
            inverse_indices = pooling_inverses[k]
            if inverse_indices is not None:
                curr_feats = curr_feats[inverse_indices]

        pca_rgb = compute_pca_colors(curr_feats)
        # umap_rgb = compute_umap_colors(curr_feats)
        save_pcd(coord_s0, pca_rgb, f"{OUTPUT_DIR}/stage_{target_stage}_projected_to_dense_pca.pcd")
        # save_pcd(coord_s0, umap_rgb, f"{OUTPUT_DIR}/stage_{target_stage}_projected_to_dense_umap.pcd")

    # Visualize sparse points at each stage
    print("\n--- Saving Sparse Feature Visualizations ---")
    for stage_idx, (coords, feats) in enumerate(zip(stage_coords, stage_embeddings)):
        coords_np = coords.detach().cpu().numpy()
        pca_rgb = compute_pca_colors(feats)
        save_pcd(coords_np, pca_rgb, f"{OUTPUT_DIR}/stage_{stage_idx}_sparse_pca.pcd")

    print("Done. You can open the .pcd files in Open3D or CloudCompare.")


if __name__ == "__main__":
    main()