# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["trimesh>=4.5", "manifold3d>=2.5", "numpy", "scipy"]
# ///
# Authors the coffee-mug STL used by examples/coffee_cup.jl.
# Run: uv run examples/assets/make_mug.py
#
# Numbers are millimetres by convention; the STL itself is unitless and the Julia
# side reinterprets it with `import_mesh(path, mm)`. Proportions are deliberately
# chunky: the mug is sampled at dx = 4 mm for the docs hero image, and the
# Poisson-disk surface sampler needs wall/base thickness >= ~2x the spacing so
# opposite faces of a wall don't block each other's samples.

import os

import numpy as np
import trimesh
from trimesh.creation import cylinder, torus
from trimesh.transformations import rotation_matrix

R = 40.0  # outer radius
H = 80.0  # height, base at z = 0
WALL = 10.0  # wall thickness (cavity radius R - WALL)
BASE = 10.0  # base thickness (cavity floor at z = BASE)

body = cylinder(radius=R, height=H, sections=128)
body.apply_translation([0.0, 0.0, H / 2])

# Handle: torus ring in the x-z plane. The inner arc reaches into the cavity
# region on purpose -- subtracting the cavity last amputates it, leaving a
# C-shaped handle whose cut ends are flush with the cavity wall.
handle = torus(major_radius=22.0, minor_radius=8.0, major_sections=96, minor_sections=48)
handle.apply_transform(rotation_matrix(np.pi / 2, [1, 0, 0]))
handle.apply_translation([R - 2.0, 0.0, H / 2])

# Cavity spans z in [BASE, H + BASE]: it pokes out the top so the rim stays open.
cavity = cylinder(radius=R - WALL, height=H, sections=128)
cavity.apply_translation([0.0, 0.0, BASE + H / 2])

mug = trimesh.boolean.union([body, handle], engine="manifold")
mug = trimesh.boolean.difference([mug, cavity], engine="manifold")
mug.process(validate=True)
mug.fix_normals()

assert mug.is_watertight, "mug mesh is not watertight"
assert mug.is_winding_consistent, "mug mesh has inconsistent winding"
assert mug.volume > 0, "mug mesh has non-positive signed volume"

print(f"faces:   {len(mug.faces)}")
print(f"extents: {mug.extents}")
print(f"volume:  {mug.volume:.3e}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mug.stl")
mug.export(out)
print(f"saved:   {out}")
