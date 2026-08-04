# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["trimesh>=4.5", "manifold3d>=2.5", "numpy", "scipy", "scikit-image", "fast-simplification"]
# ///
# Prepares the moka-pot STL used by examples/moka_pot.jl.
#
# Source model: "3D Printable Moka Pot" by CNC Kitchen (Stefan Hermann),
# https://www.thingiverse.com/thing:2490679, licensed CC BY 4.0. Download the
# assembled MokaPot.stl from the Files tab into this directory (it is not
# committed — only the processed moka.stl is), then run:
#   uv run examples/assets/prep_moka.py
#
# The raw model is nine overlapping printable parts with functional internals —
# boiler and collector chambers, funnel, filter plates, a deliberate safety hole
# in the boiler wall. Point sampling needs wall thickness >= ~2x the spacing and
# the steady solver is fragile on thin shells, so this script closes every
# cavity and emits a single solid body: voxelize, morphological closing (seals
# the safety hole and pour-lip gap so the flood fill can't hollow the chambers),
# fill, marching cubes, smooth, decimate. Exterior is visually unchanged at the
# 3-4 mm sampling the hero uses; numbers are millimetres, unitless in the STL.

import os

import numpy as np
import trimesh
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PITCH = 0.8  # voxel size, mm
CLOSE_R = 7  # closing radius, voxels (~5.6 mm) — must exceed the largest leak

raw = trimesh.load(os.path.join(HERE, "MokaPot.stl"))
print(f"raw: {len(raw.faces)} faces, extents {np.round(raw.extents, 1)}")

vox = raw.voxelized(PITCH)
occ = vox.matrix.copy()
pad = CLOSE_R + 2
occ = np.pad(occ, pad)  # room to dilate without touching the border

ball = ndimage.generate_binary_structure(3, 1)
occ = ndimage.binary_dilation(occ, ball, iterations=CLOSE_R)
occ = ndimage.binary_fill_holes(occ)
occ = ndimage.binary_erosion(occ, ball, iterations=CLOSE_R)

solid = trimesh.voxel.VoxelGrid(occ).marching_cubes
# marching cubes runs in padded voxel coordinates; map back to model mm
solid.apply_scale(PITCH)
solid.apply_translation(vox.translation - pad * PITCH)

# keep the largest body in case closing left floaters
bodies = solid.split(only_watertight=False)
solid = max(bodies, key=lambda b: len(b.faces))

trimesh.smoothing.filter_taubin(solid, iterations=10)
solid = solid.simplify_quadric_decimation(face_count=24_000)
solid.process(validate=True)
solid.fix_normals()

# base flat on z = 0 so the Julia script can partition bottom vs walls by height
solid.apply_translation([0.0, 0.0, -solid.bounds[0][2]])

assert solid.is_watertight, "solid mesh is not watertight"
assert solid.is_winding_consistent, "solid mesh has inconsistent winding"
assert solid.volume > 2.0e5, f"suspiciously small volume {solid.volume:.3e} — a chamber leaked"

print(f"solid: {len(solid.faces)} faces, extents {np.round(solid.extents, 1)}")
print(f"volume: {solid.volume:.3e} (raw parts {raw.volume:.3e})")

out = os.path.join(HERE, "moka.stl")
solid.export(out)
print(f"saved:  {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
