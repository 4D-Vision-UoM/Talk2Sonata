"""
Visualization script for Sonata model features using Smooth PCA Interpolation.
Removes blocky voxel artifacts by using Inverse Distance Weighted (IDW) interpolation
from the deep (coarse) stage to the dense (original) stage.
"""

import os
import numpy as np
import torch
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

from custom.scannet import ScanNetDataset
import custom as sonata

# Configuration
SEED = 53124
DATA_PATH = 'data/scannet_data/val'
OUTPUT_DIR = "outputs/visualizations/smooth_pca"
CUSTOM_CONFIG = dict(
    enc_patch_size=[1024 for _ in range(5)],
    enable_flash=False,
    enc_mode=True,
    freeze_encoder=False,
)


def compute_pca_colors(features):
    """
    Compresses high-dim features (N, D) to (N, 3) RGB using PCA.
    Colors are normalized min-max per channel to fit 0-1 range.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    # Fit PCA to get top 3 principal components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)

    # Normalize each channel to [0, 1] for RGB visualization
    for i in range(3):
        v_min = pca_features[:, i].min()
        v_max = pca_features[:, i].max()
        # Avoid division by zero
        if v_max - v_min > 1e-6:
            pca_features[:, i] = (pca_features[:, i] - v_min) / (v_max - v_min)
        else:
            pca_features[:, i] = 0.5

    return pca_features


def interpolate_colors(source_coords, source_colors, target_coords, k=3):
    """
    Interpolates RGB colors from sparse source points to dense target points
    using Inverse Distance Weighting (KNN regression).
    
    Args:
        source_coords: (M, 3) Coords of the deep/coarse features
        source_colors: (M, 3) PCA colors of the deep features
        target_coords: (N, 3) Coords of the original dense point cloud
        k: Number of neighbors to interpolate from (smoothing factor)
    """
    if isinstance(source_coords, torch.Tensor): source_coords = source_coords.detach().cpu().numpy()
    if isinstance(target_coords, torch.Tensor): target_coords = target_coords.detach().cpu().numpy()
    
    print(f"  > Interpolating {len(source_coords)} coarse points -> {len(target_coords)} dense points (k={k})...")
    
    # We use KNeighborsRegressor with 'distance' weights.
    # This means closer neighbors have more influence on the color, 
    # creating a smooth gradient between voxels.
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(source_coords, source_colors)
    
    smoothed_colors = knn.predict(target_coords)
    return smoothed_colors


def save_pcd(coords, colors, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved: {filename}")


def main():
    # Set seed for reproducibility
    sonata.utils.set_seed(SEED)

    # Load dataset
    print("Loading Dataset...")
    dataset = ScanNetDataset(data_root=DATA_PATH)
    sample = dataset[2] # Use same index as before

    # Load model
    print("Loading Model...")
    model = sonata.load(
        "sonata", repo_id="facebook/sonata", custom_config=CUSTOM_CONFIG
    ).cuda()

    # Load transform
    transform = sonata.transform.default()

    # Prepare point data
    point = sample.copy()
    if "segment20" in point:
        segment = point.pop("segment20")
        point["segment"] = segment
    
    # IMPORTANT: Keep a copy of the raw original mesh/points before voxelization
    # We will project features onto THIS, not the voxelized version.
    original_mesh_coords = point["coord"].copy() 

    # Apply transform (Voxelization happens here)
    point = transform(point)

    # Move tensors to GPU
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        point = model(point)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract stage data
    stage_embeddings = point['stage_embeddings']
    stage_coords = point['stage_coords']

    print("\n--- Generating Smooth Feature Visualizations ---")
    
    # We project everything onto the original dense coordinates
    # Note: original_mesh_coords might be numpy, ensure compatibility
    if isinstance(original_mesh_coords, torch.Tensor):
        dense_coords = original_mesh_coords.detach().cpu().numpy()
    else:
        dense_coords = original_mesh_coords

    # Iterate through stages (skip stage 0 as it's the input)
    for i in range(1, len(stage_embeddings)):
        print(f"Processing Stage {i}...")
        
        # 1. Get Deep Features and Coords
        curr_feats = stage_embeddings[i]     # (M, C)
        curr_coords = stage_coords[i]        # (M, 3)
        
        # 2. Compute PCA Colors on the sparse/coarse level first
        # Doing PCA *before* interpolation ensures the colors represent 
        # the feature manifold structure, not spatial smoothing artifacts.
        coarse_colors = compute_pca_colors(curr_feats)
        
        # 3. Interpolate Colors to Dense Resolution
        # This removes the "Cube" look by blending the PCA colors spatially
        smooth_colors = interpolate_colors(
            source_coords=curr_coords, 
            source_colors=coarse_colors, 
            target_coords=dense_coords,
            k=4  # Higher k = smoother/blurrier, Lower k = sharper
        )

        # 4. Save
        save_pcd(dense_coords, smooth_colors, f"{OUTPUT_DIR}/stage_{i}_smooth_pca.pcd")

    print("\nDone. Check folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()