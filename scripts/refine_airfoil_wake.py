#!/usr/bin/env python3
"""Conformingly refine a downstream band while preserving all RANS quads.

This is isotropic triangle refinement for a controlled wake-sensitivity study.
It does not extend the boundary-layer quads or alter their trailing-edge fan.
All lengths in the wake settings are chord multiples; size_factor is not used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from generate_airfoil_mesh import (cell_blocks, cross, plot_mesh, positive,
                                   require, keys_only, read_su2, signed_area,
                                   validate_mesh)


DEFAULT_WAKE = dict(start=0.02, length=2.0, half_width=0.025, spread=0.025,
                    slope=0.0, size=0.002, downstream_growth=4.0, transition_rate=0.4)


def read_settings(path):
    config = json.loads(path.read_text())
    keys_only(config, {"input", "output", "chord", "wake"}, "configuration")
    for key in ("input", "output"):
        require(isinstance(config.get(key), str) and config[key].strip(), f"{key} is required")
        config[key] = str((path.parent/config[key]).resolve())
    config["chord"] = positive(config.get("chord", 1), "chord")
    wake = config.get("wake", {})
    keys_only(wake, DEFAULT_WAKE, "wake")
    wake = DEFAULT_WAKE | wake
    for key in ("start", "length", "half_width", "size", "downstream_growth", "transition_rate"):
        wake[key] = positive(wake[key], f"wake.{key}")
    for key in ("slope", "spread"):
        value = wake[key]
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value), f"wake.{key} must be finite")
    require(wake["spread"] >= 0, "wake.spread must be nonnegative")
    require(wake["start"] < wake["length"], "wake.start must be smaller than length")
    config["wake"] = wake
    return config


def split_triangle(cell, mids, points):
    """Red/green subdivision with shared edge midpoints and no hanging nodes."""
    a, b, c = map(int, cell)
    m = [mids.get(tuple(sorted(e))) for e in ((a, b), (b, c), (c, a))]
    count = sum(v is not None for v in m)
    if count == 0:
        return [(a, b, c)]
    if count == 3:
        u, v, w = m
        return [(a, u, w), (u, b, v), (w, v, c), (u, v, w)]
    if count == 1:
        i = next(i for i, v in enumerate(m) if v is not None)
        a, b, c = np.roll(cell, -i).tolist()
        return [(a, m[i], c), (m[i], b, c)]
    # Rotate so the two bisected sides are a-b and b-c. Choose the better diagonal.
    i = next(i for i in range(3) if m[i] is not None and m[(i+1) % 3] is not None)
    a, b, c = np.roll(cell, -i).tolist()
    u, v = m[i], m[(i+1) % 3]
    choices = [[(u, b, v), (a, u, c), (u, v, c)],
               [(u, b, v), (a, u, v), (a, v, c)]]

    def score(triangles):
        p = points[np.asarray(triangles)]
        area2 = cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        lengths2 = np.sum((np.roll(p, -1, axis=1)-p)**2, axis=(1, 2))
        return np.min(area2/lengths2)

    return max(choices, key=score)


def refine(points, cells, markers, chord, wake, max_passes=12, max_nodes=2000000):
    blocks = cell_blocks(cells)
    triangles = blocks.get(3, np.empty((0, 3), dtype=int)).copy()
    quads = blocks.get(4, np.empty((0, 4), dtype=int)).copy()
    require(len(triangles) > 0, "Wake refinement requires triangles")
    points = points.copy()
    original_nodes, original_triangles = len(points), len(triangles)
    fixed = {tuple(sorted(e)) for edges in markers.values() for e in edges}
    fixed.update(tuple(sorted(e)) for q in quads for e in zip(q, np.roll(q, -1)))
    passes = []
    for iteration in range(max_passes+1):
        edges = np.unique(np.sort(np.stack((triangles, np.roll(triangles, -1, axis=1)),
                                           axis=2).reshape(-1, 2), axis=1), axis=0)
        p = points[edges]/chord
        middle = p.mean(axis=1)
        s = middle[:, 0]-1
        clipped = np.clip(s, wake["start"], wake["length"])
        transverse = np.maximum(abs(middle[:, 1]-wake["slope"]*clipped)
                                -wake["half_width"]-wake["spread"]*clipped, 0)
        # A continuous size transition avoids repeated green splits producing
        # a row of thin triangles against an abruptly unrefined band boundary.
        distance = np.hypot(s-clipped, transverse)
        target = (wake["size"]*(1+wake["downstream_growth"]*(clipped-wake["start"]))
                  +wake["transition_rate"]*distance)
        selected = edges[(s >= 0) & (np.linalg.norm(p[:, 1]-p[:, 0], axis=1) > target*(1+1e-10))]
        selected = [tuple(e) for e in selected if tuple(e) not in fixed]
        if not selected:
            break
        require(iteration < max_passes, "Wake refinement did not reach the requested edge size")
        require(len(points)+len(selected) <= max_nodes, "Wake refinement exceeds the node limit")
        mids = {edge: len(points)+i for i, edge in enumerate(selected)}
        points = np.vstack((points, points[np.asarray(selected)].mean(axis=1)))
        triangles = np.asarray([t for cell in triangles for t in split_triangle(cell, mids, points)], dtype=int)
        passes.append(dict(split_edges=len(selected), nodes=len(points), triangles=len(triangles)))
    # All original coordinates, marker edges, and quads retain their IDs exactly.
    return points, {3: triangles, 4: quads}, dict(
        original_nodes=original_nodes, added_nodes=len(points)-original_nodes,
        original_triangles=original_triangles, preserved_quads=len(quads), passes=passes,
        note="Only triangle edges are bisected. Marker and quad edges remain fixed; "
             "the size bound excludes these protected edges. The trailing-edge fan is unchanged.")


def write_su2(path, points, cells, markers):
    with path.open("w") as stream:
        stream.write(f"NDIME= 2\nNELEM= {sum(len(c) for c in cells.values())}\n")
        for n, block in cells.items():
            for c in block:
                stream.write(f"{5 if n == 3 else 9} " + " ".join(map(str, c)) + "\n")
        stream.write(f"NPOIN= {len(points)}\n")
        for i, (x, y) in enumerate(points):
            stream.write(f"{x:.17g} {y:.17g} {i}\n")
        stream.write(f"NMARK= {len(markers)}\n")
        for name, edges in markers.items():
            stream.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {len(edges)}\n")
            for a, b in edges:
                stream.write(f"3 {a} {b}\n")


def wall_contour(points, edges):
    neighbors = {}
    for a, b in edges:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    require(all(len(v) == 2 for v in neighbors.values()), "Invalid wall loop")
    first = next(iter(neighbors)); order = [first]; previous = None
    while True:
        nxt = next(i for i in neighbors[order[-1]] if i != previous)
        if nxt == first:
            break
        require(nxt not in order, "Invalid wall cycle")
        previous = order[-1]; order.append(nxt)
    require(len(order) == len(neighbors), "Disconnected wall")
    contour = points[order]
    return contour if signed_area(contour) > 0 else contour[::-1]


def run(path):
    settings = read_settings(path.resolve())
    source, output = Path(settings["input"]), Path(settings["output"])
    require(not output.exists(), f"Output already exists: {output}")
    points, cells, markers = read_su2(source)
    require(set(markers) == {"airfoil", "farfield"}, "Expected airfoil and farfield markers")
    contour = wall_contour(points, markers["airfoil"])
    original = points.copy()
    original_quads = cell_blocks(cells).get(4, np.empty((0, 4), dtype=int)).copy()
    points, cells, summary = refine(points, cells, markers, settings["chord"], settings["wake"])
    require(np.array_equal(points[:len(original)], original), "Original coordinates changed")
    expected = dict(nodes=len(points), cells=sum(len(c) for c in cells.values()),
                    markers={k: len(v) for k, v in markers.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wake-refine-", dir=output.parent) as tmp:
        directory = Path(tmp)
        os.environ.setdefault("MPLCONFIGDIR", str(directory/"mpl"))
        os.environ.setdefault("XDG_CACHE_HOME", str(directory/"cache"))
        write_su2(directory/"mesh.su2", points, cells, markers)
        p, c, m = read_su2(directory/"mesh.su2")
        require(np.array_equal(p[:len(original)], original), "Export changed original coordinates")
        require(np.array_equal(cell_blocks(c).get(4, np.empty((0, 4), dtype=int)), original_quads),
                "Export changed original quadrilaterals")
        require(all(np.array_equal(m[k], v) for k, v in markers.items()), "Export changed markers")
        quality = validate_mesh(p, c, m, contour, expected, settings["chord"])
        quality["wake_refinement"] = summary
        quality["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        quality["checks"].append("original_nodes_and_quad_connectivity_preserved")
        radius = np.max(np.linalg.norm(p[m["farfield"]].reshape(-1, 2)/settings["chord"]-[.5, 0], axis=1))
        plot_settings = dict(chord=settings["chord"], mesh=dict(farfield_radius=radius))
        plot_mesh(p, c, contour, plot_settings, directory)
        (directory/"quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False)+"\n")
        (directory/"settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False)+"\n")
        output.mkdir()
        for artifact in directory.iterdir():
            if artifact.is_file():
                shutil.move(str(artifact), output/artifact.name)
    return output, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        output, quality = run(args.config)
    except (ValueError, OSError, RuntimeError, StopIteration, ImportError) as error:
        parser.exit(1, f"Wake refinement failed: {error}\n")
    print(f"Created {output}: {quality['nodes']} nodes, {quality['cells']} cells")
    for warning in quality["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
