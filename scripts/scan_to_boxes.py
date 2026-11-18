#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["embreex", "networkx", "numpy", "rtree", "scipy", "trimesh"]
# ///
"""Cover a chair scan with axis-aligned boxes for generate_layers.py's seat.

The planner's world is boxes only, so a scanned chair reaches it as
`seat.boxes` (`frame: arm_mount`) in the generator's inputs. This prints
that YAML block. Run it on a workstation (it needs trimesh, which the image
lacks) and paste its output into the robot's mebot_layers.yaml:

    uv run scripts/scan_to_boxes.py chair.stl --origin X Y Z [--cut X Y Z ...]

The scan is a closed mesh in millimetres, z up, the chair facing -y and its
left (the arm's side on flatbot) +x. --origin is the arm's base (arm_mount's
origin) in the scan; --cut is a point on each smooth surface patch to drop,
with the solid it bounds: an arm fused into the scan. The surface is split
into patches at edges sharper than 30 degrees.

Every cell of a --pitch grid (aligned on the origin) that the solid or its
surface touches is occupied; the occupied cells are swept into boxes, which
are merged pairwise down to --boxes, the pair that adds the least volume
first. A merge only grows a box, so the result covers every occupied cell,
and every vertex and surface sample not on a cut patch is checked to be
inside a box. The boxes are written in arm_mount, metres, rounded outward to
0.1 mm; the generator carves them around the arm's fixed spheres.

--solid NAME=X0,Y0,Z0,X1,Y1,Z1 (arm_mount, metres) is written as one box as
given, snapped outward onto the grid: a volume the arm never enters (under
the seat, the backrest). The scan inside it is not boxed again.
"""

import argparse
import hashlib
import sys

import numpy as np
import trimesh
from trimesh.graph import connected_components


def arm_faces(mesh, cuts):
    """Faces of the smooth patches (split at > 30 degree edges) through `cuts`."""
    smooth = mesh.face_adjacency[mesh.face_adjacency_angles < np.radians(30)]
    patch = np.empty(len(mesh.faces), int)
    for k, c in enumerate(
        connected_components(smooth, nodes=np.arange(len(mesh.faces)), min_len=1)
    ):
        patch[c] = k
    if not cuts:
        return np.zeros(len(mesh.faces), bool)
    _, _, tri = trimesh.proximity.closest_point(mesh, np.asarray(cuts, float))
    return np.isin(patch, patch[tri])


def occupancy(mesh, origin, pitch, cut, samples):
    """Occupied cells (bool grid), the grid's first cell index, and the cut's cells removed."""
    first = np.floor((mesh.bounds[0] - origin) / pitch).astype(int) - 1
    shape = np.ceil((mesh.bounds[1] - origin) / pitch).astype(int) + 1 - first
    occ = np.zeros(shape, bool)
    cell = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1
    ).reshape(-1, 3)
    occ.reshape(-1)[mesh.contains(origin + (first + cell + 0.5) * pitch)] = True
    occ[tuple((np.floor((samples - origin) / pitch).astype(int) - first).T)] = True
    if cut.any():
        # a cell whose nearest surface is a cut patch belongs to what is cut
        idx = np.argwhere(occ)
        centre = origin + (first + idx + 0.5) * pitch
        tri_cut = mesh.triangles[cut].reshape(-1, 3)
        near = np.all(
            (centre > tri_cut.min(0) - 2 * pitch)
            & (centre < tri_cut.max(0) + 2 * pitch),
            axis=1,
        )
        _, _, tri = trimesh.proximity.closest_point(mesh, centre[near])
        occ[tuple(idx[near][cut[tri]].T)] = False
    return occ, first


def sweep(occ):
    """Occupied cells as boxes [lo, hi) of cell indices: runs along x, then y, then z."""
    rem = occ.copy()
    boxes = []
    nx, ny, nz = occ.shape
    for i, j, k in np.argwhere(occ):
        if not rem[i, j, k]:
            continue
        i2 = i + 1
        while i2 < nx and rem[i2, j, k]:
            i2 += 1
        j2 = j + 1
        while j2 < ny and rem[i:i2, j2, k].all():
            j2 += 1
        k2 = k + 1
        while k2 < nz and rem[i:i2, j:j2, k2].all():
            k2 += 1
        rem[i:i2, j:j2, k:k2] = False
        boxes.append((i, j, k, i2, j2, k2))
    return np.array(boxes, dtype=np.int64)


def merge(boxes, budget):
    """Merge pairwise to `budget` boxes, least added volume first (ties: first pair)."""
    lo, hi = boxes[:, :3].copy(), boxes[:, 3:].copy()
    n = len(boxes)
    alive = np.ones(n, bool)

    def vol(a, b):
        return np.prod(np.maximum(b - a, 0), axis=-1)

    def added(i):
        both = vol(np.minimum(lo[i], lo), np.maximum(hi[i], hi))
        overlap = vol(np.maximum(lo[i], lo), np.minimum(hi[i], hi))
        cost = (both - vol(lo[i], hi[i]) - vol(lo, hi) + overlap).astype(float)
        cost[~alive] = np.inf
        cost[i] = np.inf
        return cost

    cost = np.stack([added(i) for i in range(n)])
    while alive.sum() > budget:
        i, j = sorted(divmod(int(np.argmin(cost)), n))
        lo[i], hi[i] = np.minimum(lo[i], lo[j]), np.maximum(hi[i], hi[j])
        alive[j] = False
        cost[j, :] = cost[:, j] = np.inf
        cost[i, :] = cost[:, i] = added(i)
    return lo[alive], hi[alive]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stl", help="the scan: a closed mesh, mm")
    ap.add_argument(
        "--origin",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="the arm's base in the scan, mm",
    )
    ap.add_argument(
        "--cut",
        type=float,
        nargs=3,
        action="append",
        default=[],
        metavar=("X", "Y", "Z"),
        help="a point on a patch to drop, mm",
    )
    ap.add_argument("--pitch", type=float, default=10.0, help="grid, mm (default 10)")
    ap.add_argument("--boxes", type=int, default=30, help="box budget (default 30)")
    ap.add_argument("--name", default="scan", help="box name prefix")
    ap.add_argument(
        "--solid",
        action="append",
        default=[],
        metavar="NAME=X0,Y0,Z0,X1,Y1,Z1",
        help="one box written as given (arm_mount, m)",
    )
    args = ap.parse_args(argv)

    mesh = trimesh.load(args.stl, force="mesh")
    if not mesh.is_watertight:
        sys.exit("scan_to_boxes: %s is not a closed mesh" % args.stl)
    origin, pitch = np.asarray(args.origin), args.pitch
    cut = arm_faces(mesh, args.cut)
    pts, face = trimesh.sample.sample_surface_even(
        mesh, 2_000_000, radius=pitch / 6, seed=0
    )
    occ, first = occupancy(mesh, origin, pitch, cut, np.r_[pts, mesh.vertices])
    solids = []
    for spec in args.solid:
        name, nums = spec.split("=")
        v = [float(t) for t in nums.split(",")]
        solids.append((name, v[:3], v[3:]))
    # each solid in cells, snapped outward onto the grid (scan mm: x = X0 + y,
    # y = Y0 - x, z = Z0 + z): the cells inside it are the solid's
    s_cells = []
    for _, s_lo, s_hi in solids:
        a = origin + 1000 * np.array([s_lo[1], -s_hi[0], s_lo[2]])
        b = origin + 1000 * np.array([s_hi[1], -s_lo[0], s_hi[2]])
        a = np.floor((a - origin) / pitch + 1e-6).astype(int) - first
        b = np.ceil((b - origin) / pitch - 1e-6).astype(int) - first
        a, b = np.maximum(a, 0), np.minimum(b, occ.shape)
        occ[a[0] : b[0], a[1] : b[1], a[2] : b[2]] = False
        s_cells.append(np.r_[a, b])
    lo, hi = merge(sweep(occ), args.boxes)
    n_solid = len(solids)
    if n_solid:
        lo = np.r_[np.array(s_cells)[:, :3], lo]
        hi = np.r_[np.array(s_cells)[:, 3:], hi]
    lo_mm, hi_mm = origin + (first + lo) * pitch, origin + (first + hi) * pitch

    keep = np.r_[mesh.vertices[np.unique(mesh.faces[~cut])], pts[~cut[face]]]
    inside = np.zeros(len(keep), bool)
    for a, b in zip(lo_mm, hi_mm):
        inside |= np.all((keep >= a - 1e-6) & (keep <= b + 1e-6), axis=1)
    if not inside.all():
        sys.exit("scan_to_boxes: %d surface points are in no box" % (~inside).sum())

    cells = occ.sum()
    union = np.zeros(occ.shape, bool)
    for a, b in zip(lo[n_solid:], hi[n_solid:]):
        union[a[0] : b[0], a[1] : b[1], a[2] : b[2]] = True
    # scan (x left, y back) -> arm_mount (x forward, y left), metres, outward to 0.1 mm
    am_lo = np.c_[
        -(hi_mm[:, 1] - origin[1]), lo_mm[:, 0] - origin[0], lo_mm[:, 2] - origin[2]
    ]
    am_hi = np.c_[
        -(lo_mm[:, 1] - origin[1]), hi_mm[:, 0] - origin[0], hi_mm[:, 2] - origin[2]
    ]
    am_lo, am_hi = np.floor(am_lo * 10 + 1e-6) / 1e4, np.ceil(am_hi * 10 - 1e-6) / 1e4
    order = n_solid + np.argsort(
        -np.prod(am_hi[n_solid:] - am_lo[n_solid:], axis=1), kind="stable"
    )

    sha = hashlib.sha256(open(args.stl, "rb").read()).hexdigest()
    print(
        "  # GENERATED by scripts/scan_to_boxes.py from %s (sha256 %s):"
        % (args.stl.split("/")[-1], sha)
    )
    print(
        "  #   --origin %s%s%s --pitch %g --boxes %d"
        % (
            " ".join("%g" % v for v in args.origin),
            "".join(" --cut %s" % " ".join("%g" % v for v in c) for c in args.cut),
            "".join(" --solid %s" % spec for spec in args.solid),
            pitch,
            args.boxes,
        )
    )
    print(
        "  # %d surface points checked inside; outside the solids %d boxes hold %d"
        " cells of %g mm, %.1f %% over"
        % (len(keep), len(order), cells, pitch, 100.0 * (union.sum() - cells) / cells)
    )
    print("  boxes:")
    names = [name for name, _, _ in solids]
    names += ["%s_%02d" % (args.name, n) for n in range(1, len(order) + 1)]
    for name, i in zip(names, [*range(n_solid), *order]):
        print(
            "    %s: [[%s], [%s]]"
            % (
                name,
                ", ".join("%.4f" % (v + 0.0) for v in am_lo[i]),
                ", ".join("%.4f" % (v + 0.0) for v in am_hi[i]),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
