from custom.scannet import ScanNetDataset

train_path = 'data/scannet_data/val'
dataset = ScanNetDataset(data_root=train_path)
print(f"Dataset length: {len(dataset)}")
sample = dataset[0]



import open3d as o3d
import custom as sonata
import torch


flash_attn = None


sonata.utils.set_seed(53124)


custom_config = dict(
    enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
    enable_flash=False,
    enc_mode=True,
    freeze_encoder=False,
)
model = sonata.load(
    "sonata", repo_id="facebook/sonata", custom_config=custom_config
).cuda()
    
    
# Load default data transform pipeline
transform = sonata.transform.default()  # voxelize and downsample points

point = sample


# point.pop("segment200")
segment = point.pop("segment20")
point["segment"] = segment  # two kinds of segment exist in ScanNet, only use one
original_coord = point["coord"].copy()

original_point = point.copy()
point = transform(point)




with torch.inference_mode():
    for key in point.keys():
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].cuda(non_blocking=True)
    # model forward:
    point = model(point)
    
import pdb; pdb.set_trace()    
#     # upcast point feature
#     # Point is a structure contains all the information during forward
#     for _ in range(2):
#         assert "pooling_parent" in point.keys()
#         assert "pooling_inverse" in point.keys()
#         parent = point.pop("pooling_parent")
#         inverse = point.pop("pooling_inverse")
#         parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
#         point = parent
#     while "pooling_parent" in point.keys():
#         assert "pooling_inverse" in point.keys()
#         parent = point.pop("pooling_parent")
#         inverse = point.pop("pooling_inverse")
#         parent.feat = point.feat[inverse]
#         point = parent

#     # here point is down-sampled by GridSampling in default transform pipeline
#     # feature of point cloud in original scale can be acquired by:
#     _ = point.feat[point.inverse]

#     # PCA
#     pca_color = get_pca_color(point.feat, brightness=1.2, center=True)

# # inverse back to original scale before grid sampling
# # point.inverse is acquired from the GirdSampling transform
# original_pca_color = pca_color[point.inverse]
# pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(original_coord)
# # pcd.colors = o3d.utility.Vector3dVector(original_pca_color.cpu().detach().numpy())
# # o3d.visualization.draw_geometries([pcd])
# # or
# o3d.visualization.draw_plotly([pcd])

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
# pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
# o3d.io.write_point_cloud("pca.ply", pcd)
