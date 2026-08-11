# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["trimesh>=4.5", "numpy", "scipy", "fast-simplification"]
# ///
# Prepares the armadillo STL used by examples/mms_armadillo.jl.
#
# Source model: "Armadillo" from the Stanford Computer Graphics Laboratory
# 3D Scanning Repository, https://graphics.stanford.edu/data/3Dscanrep/
# (free for research and scholarly use with attribution). This script downloads
# the raw Armadillo.ply.gz on first run (it is not committed — only the
# processed armadillo.stl is), then run:
#   uv run examples/assets/prep_armadillo.py
#
# The raw scan reconstruction is already watertight, so unlike the moka pot no
# voxel surgery is needed: rotate the graphics y-up convention to z-up, decimate
# 172k faces to ~30k (plenty for 3-4 mm point sampling), and set the feet on
# z = 0 with the body centered in x/y. Numbers are millimetres, unitless in the
# STL — the raw model is ~152 units tall, which reads naturally as a desk-statue
# sized ~152 mm.

import gzip
import os
import shutil
import urllib.request

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
RAW = os.path.join(HERE, "Armadillo.ply")

if not os.path.exists(RAW):
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp, open(RAW + ".gz", "wb") as f:
        shutil.copyfileobj(resp, f)
    with gzip.open(RAW + ".gz", "rb") as fin, open(RAW, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(RAW + ".gz")

mesh = trimesh.load(RAW)
print(f"raw: {len(mesh.faces)} faces, extents {np.round(mesh.extents, 1)}")

# y-up (graphics convention) -> z-up: (x, y, z) -> (x, -z, y)
R = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
mesh.apply_transform(R)

mesh = mesh.simplify_quadric_decimation(face_count=30_000)
mesh.process(validate=True)
mesh.fix_normals()

# feet on z = 0, body centered in x/y
lo, hi = mesh.bounds
mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

assert mesh.is_watertight, "mesh is not watertight"
assert mesh.is_winding_consistent, "mesh has inconsistent winding"

print(f"solid: {len(mesh.faces)} faces, extents {np.round(mesh.extents, 1)}")
print(f"volume: {mesh.volume:.3e} mm^3")

out = os.path.join(HERE, "armadillo.stl")
mesh.export(out)
print(f"saved:  {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
