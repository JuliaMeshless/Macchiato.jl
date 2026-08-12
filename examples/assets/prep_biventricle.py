# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["pyvista>=0.44", "trimesh>=4.5", "numpy", "scipy", "scikit-image", "fast-simplification"]
# ///
# Prepares the biventricular STL used by examples/biventricle_monodomain.jl.
#
# Source: Bai et al. (2015) UK Biobank / UK Digital Heart Project biventricular
# cardiac atlas, PCA instance 010, volumetric tetrahedral myocardium mesh
# (~26k tets, mm coordinates, Cobiveco point fields), from the
# DerangedIons/ArmyHeart.jl repo (git-LFS). Fetch and copy:
#   brew install git-lfs && git lfs install
#   cd ~/dev/ArmyHeart && git lfs pull --include "data/meshes/bai_etal_2015-instance_010_e26073.vtu"
#   cp ~/dev/ArmyHeart/data/meshes/bai_etal_2015-instance_010_e26073.vtu examples/assets/
# (Without git-lfs, the blob can also be fetched from the GitHub LFS batch API
# using the oid/size in the pointer file.) The raw .vtu is not committed — only
# the processed biventricle.stl is. Then run:
#   uv run examples/assets/prep_biventricle.py
#
# The smooth atlas geometry is deliberately "re-segmented": myocardial occupancy
# is rasterized at 1 mm by point-in-tet queries against the volumetric mesh, and
# a marching-cubes isosurface is extracted with NO smoothing — exactly the
# stair-step artifact a real CT/MR segmentation pipeline produces. Occupancy
# comes from the tets themselves (no surface flood-fill anywhere), so the LV/RV
# blood pools stay outside the solid by construction. Mild quadric decimation
# shrinks the STL to a committable size while keeping the blocky character.
#
# Output convention: long axis along z, apex down at z = 0, centered in x/y,
# coordinates are millimetre numbers (unitless in the STL).

import os

import numpy as np
import pyvista as pv
import trimesh
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "bai_etal_2015-instance_010_e26073.vtu")
PITCH = 1.0  # voxel size, mm — the fake scanner resolution
FACE_TARGET = 40_000

if not os.path.exists(RAW) or os.path.getsize(RAW) < 10_000:
    raise SystemExit(
        f"{RAW} is missing or still a git-LFS pointer stub — "
        "fetch it first (see the header of this script)."
    )

grid = pv.read(RAW)
lo = grid.points.min(axis=0)
hi = grid.points.max(axis=0)
print(f"raw: {grid.n_cells} tets, extents {np.round(hi - lo, 1)}")
assert 60.0 < (hi - lo).max() < 160.0, "bounding box not heart-sized in mm — unit problem?"

# --- myocardial occupancy: a voxel center is tissue iff it lies inside some tet.
# No surface fill anywhere, so the LV/RV cavities stay outside by construction —
# open valve orifices or a closed basal lid are both irrelevant. (This is the
# load-bearing difference from prep_moka.py, whose dilate/fill/erode would have
# flooded the cavities shut.)
pad = 2
dims = np.ceil((hi - lo) / PITCH).astype(int) + 2 * pad
origin = lo - pad * PITCH
axes = [origin[i] + (np.arange(dims[i]) + 0.5) * PITCH for i in range(3)]
X, Y, Z = np.meshgrid(*axes, indexing="ij")
centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
occ = (grid.find_containing_cell(centers) >= 0).reshape(dims)
print(f"occupancy: {occ.sum()} of {occ.size} voxels at {PITCH} mm pitch")

# largest connected component (drops stray voxel islands)
labels, n_labels = ndimage.label(occ)
if n_labels > 1:
    counts = np.bincount(labels.ravel())[1:]
    occ = labels == (np.argmax(counts) + 1)
    print(f"kept largest of {n_labels} components ({counts.max()} voxels)")

# marching cubes with NO smoothing — the stair-steps are the point
surf = trimesh.voxel.VoxelGrid(occ).marching_cubes
surf.apply_scale(PITCH)  # index space -> mm; absolute offset is fixed by re-centering below

bodies = surf.split(only_watertight=False)
surf = max(bodies, key=lambda b: len(b.faces))
print(f"marching cubes: {len(surf.faces)} faces ({len(bodies)} bodies)")

surf = surf.simplify_quadric_decimation(face_count=FACE_TARGET)
surf.process(validate=True)
surf.fix_normals()

# --- orient: apex-base axis -> +z (apex down at z = 0), centered in x/y.
# The anatomical long axis comes from the atlas Cobiveco apicobasal coordinate
# ("ab": 0 at apex, 1 at base): the least-squares gradient of ab over the mesh
# points is the apex->base direction. (A vertex-PCA long axis is wrong here —
# the two side-by-side ventricles make the LATERAL direction the widest.)
ab = np.asarray(grid.point_data["ab"]).ravel()
A = np.column_stack([grid.points, np.ones(grid.n_points)])
grad_ab = np.linalg.lstsq(A, ab, rcond=None)[0][:3]
axis = grad_ab / np.linalg.norm(grad_ab)
print(f"apex->base direction (from ab field): {np.round(axis, 3)}")
R = trimesh.geometry.align_vectors(axis, [0.0, 0.0, 1.0])
surf.apply_transform(R)

blo, bhi = surf.bounds
surf.apply_translation([-(blo[0] + bhi[0]) / 2, -(blo[1] + bhi[1]) / 2, -blo[2]])

assert surf.is_watertight, "surface is not watertight"
assert surf.is_winding_consistent, "surface has inconsistent winding"
assert len(surf.split(only_watertight=False)) == 1, "surface is not a single body"
assert 0.7e5 < surf.volume < 2.6e5, f"myocardial volume {surf.volume:.3e} mm^3 implausible"
# apex-base height along z; the side-by-side ventricles are wider laterally
assert 70.0 < surf.extents[2] < 130.0, f"apex-base extent {surf.extents[2]:.1f} mm implausible"

print(f"final: {len(surf.faces)} faces, extents {np.round(surf.extents, 1)} mm")
print(f"myocardial volume: {surf.volume / 1e3:.1f} cm^3")

out = os.path.join(HERE, "biventricle.stl")
surf.export(out)
print(f"saved:  {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
