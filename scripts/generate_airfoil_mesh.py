#!/usr/bin/env python3
"""Generate and validate a 2D Euler or wall-resolved RANS airfoil mesh.

Lengths in the mesh section are chord multiples. Coordinate files describe a
single contour, with trailing-edge endpoints (Selig), or two LE/TE branches
(Lednicer). See README.md for the complete input contract.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np


DEFAULT_MESH = dict(farfield_radius=20.0, surface_size=0.01, edge_size=0.002,
                    farfield_size=1.0, wake_size=0.02, wake_length=5.0,
                    wake_half_width=0.25, size_factor=1.0)
DEFAULT_BOUNDARY_LAYER = dict(first_height=1e-6, growth_ratio=1.18,
                              thickness=0.02, fan_elements=40)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys_only(value, allowed, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(not (set(value) - set(allowed)),
            f"Unknown {label} keys: {sorted(set(value) - set(allowed))}")


def positive(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0,
            f"{label} must be a finite positive number")
    return float(value)


def airfoil_settings(airfoil, directory):
    """Validate an airfoil source shared by the mixed and structured meshers."""
    keys_only(airfoil, {"naca", "file", "format", "points_per_side", "trailing_edge"}, "airfoil")
    require(("naca" in airfoil) != ("file" in airfoil),
            "Specify exactly one of airfoil.naca and airfoil.file")
    airfoil = dict(airfoil)
    if "naca" in airfoil:
        require(not ({"format", "file"} & set(airfoil)), "File options cannot be used with NACA")
        airfoil.setdefault("points_per_side", 201)
        airfoil.setdefault("trailing_edge", "closed")
    else:
        require(not ({"points_per_side", "trailing_edge"} & set(airfoil)),
                "NACA options cannot be used with a coordinate file")
        require(airfoil.get("format") in ("selig", "lednicer"),
                "airfoil.format must explicitly be selig or lednicer")
        require(isinstance(airfoil["file"], str) and airfoil["file"].strip(),
                "airfoil.file must be a nonempty path")
        airfoil["file"] = str((directory / airfoil["file"]).resolve())
    return airfoil


def read_settings(path):
    with path.open() as stream:
        config = json.load(stream)
    keys_only(config, {"airfoil", "chord", "mesh", "output", "mode", "boundary_layer"}, "configuration")
    mode = config.get("mode", "euler")
    require(mode in ("euler", "rans"), "mode must be euler or rans")
    require(mode == "rans" or "boundary_layer" not in config,
            "boundary_layer requires mode=rans")
    boundary_layer = None
    if mode == "rans":
        boundary_layer = config.get("boundary_layer", {})
        keys_only(boundary_layer, DEFAULT_BOUNDARY_LAYER, "boundary_layer")
        boundary_layer = DEFAULT_BOUNDARY_LAYER | boundary_layer
        fan = boundary_layer["fan_elements"]
        require(isinstance(fan, int) and not isinstance(fan, bool) and 4 <= fan <= 200,
                "boundary_layer.fan_elements must be an integer from 4 to 200")
        for key in ("first_height", "growth_ratio", "thickness"):
            boundary_layer[key] = positive(boundary_layer[key], f"boundary_layer.{key}")
        h, ratio, thickness = (boundary_layer[k] for k in ("first_height", "growth_ratio", "thickness"))
        require(1.01 <= ratio <= 1.5, "boundary_layer.growth_ratio must be between 1.01 and 1.5")
        require(5*h <= thickness <= 0.1, "Require 5*first_height <= thickness <= 0.1 chord")
    airfoil = airfoil_settings(config.get("airfoil"), path.parent)
    mesh = config.get("mesh", {})
    keys_only(mesh, DEFAULT_MESH, "mesh")
    defaults = DEFAULT_MESH if mode == "euler" else DEFAULT_MESH | dict(
        surface_size=0.005, edge_size=0.0005, wake_size=0.01)
    mesh = defaults | mesh
    mesh = {key: positive(value, f"mesh.{key}") for key, value in mesh.items()}
    require(mesh["farfield_radius"] > 2, "farfield_radius must be greater than 2 chords")
    require(mesh["edge_size"] <= mesh["surface_size"] <= mesh["farfield_size"],
            "Require edge_size <= surface_size <= farfield_size")
    require(mesh["wake_size"] <= mesh["farfield_size"], "wake_size must not exceed farfield_size")
    require(math.hypot(0.5 + mesh["wake_length"], mesh["wake_half_width"])
            < mesh["farfield_radius"], "Wake refinement must fit inside the farfield")
    output = config.get("output")
    require(isinstance(output, str) and output.strip(), "output must be a nonempty path")
    result = dict(airfoil=airfoil, chord=positive(config.get("chord", 1.0), "chord"),
                  mode=mode, mesh=mesh, output=str((path.parent / output).resolve()))
    if boundary_layer is not None:
        require(boundary_layer["first_height"] < mesh["edge_size"]*mesh["size_factor"],
                "first_height must be smaller than the scaled edge_size")
        result["boundary_layer"] = boundary_layer
    return result


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def signed_area(points):
    return float(np.sum(cross(points, np.roll(points, -1, axis=0))) / 2)


def clean_contour(points):
    points = np.asarray(points, dtype=float)
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3,
            "At least three 2D coordinates are required")
    require(np.isfinite(points).all(), "Coordinates must be finite")
    span = float(np.max(np.ptp(points, axis=0)))
    require(span > 0, "Coordinate extent / chord is zero")
    tolerance = 1e-10 * span
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > tolerance:
            kept.append(point)
    if len(kept) > 1 and np.linalg.norm(kept[-1] - kept[0]) <= tolerance:
        kept.pop()
    points = np.asarray(kept)
    require(len(points) >= 3, "Too few distinct coordinates")
    # Work in dimensionless coordinates so intersection tolerances scale with input.
    p = (points - points[0]) / span
    q = np.roll(p, -1, axis=0)
    for i in range(len(p)):
        indices = np.arange(i + 2, len(p))
        if i == 0:
            indices = indices[indices != len(p) - 1]
        if not len(indices):
            continue
        a, b, c, d = p[i], q[i], p[indices], q[indices]
        o1, o2 = cross(b - a, c - a), cross(b - a, d - a)
        o3, o4 = cross(d - c, a - c), cross(d - c, b - c)
        eps = 1e-12
        overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d))
                         <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + eps, axis=1)
        opposite1 = ((o1 <= eps) & (o2 >= -eps)) | ((o2 <= eps) & (o1 >= -eps))
        opposite2 = ((o3 <= eps) & (o4 >= -eps)) | ((o4 <= eps) & (o3 >= -eps))
        require(not np.any(overlap & opposite1 & opposite2),
                f"Self-intersection or nonadjacent touching near contour edge {i}")
    require(abs(signed_area(p)) > 1e-10, "Airfoil has zero or negligible enclosed area")
    # Also reject adjacent edges that backtrack over each other.
    incoming, outgoing = p - np.roll(p, 1, axis=0), q - p
    backtrack = (np.abs(cross(incoming, outgoing)) < 1e-14) & (np.sum(incoming * outgoing, axis=1) < 0)
    require(not np.any(backtrack), "Adjacent contour edges overlap")
    return points if signed_area(points) > 0 else points[::-1].copy()


def naca4(code, points_per_side=201, trailing_edge="closed"):
    require(isinstance(code, str) and len(code) == 4 and code.isascii() and code.isdigit(),
            "NACA must be a four-digit string, e.g. \"0012\"")
    require(isinstance(points_per_side, int) and not isinstance(points_per_side, bool)
            and points_per_side >= 21, "points_per_side must be an integer >= 21")
    require(trailing_edge in ("closed", "finite"), "trailing_edge must be closed or finite")
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    require(t > 0 and (m == 0 or p > 0), "NACA requires positive thickness and a valid camber position")
    x = (1 - np.cos(np.linspace(0, np.pi, points_per_side))) / 2
    a4 = -0.1036 if trailing_edge == "closed" else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260*x - 0.3516*x*x + 0.2843*x**3 + a4*x**4)
    yc, slope = np.zeros_like(x), np.zeros_like(x)
    if m:
        front = x < p
        yc[front] = m / p**2 * (2*p*x[front] - x[front]**2)
        yc[~front] = m / (1-p)**2 * ((1-2*p) + 2*p*x[~front] - x[~front]**2)
        slope[front] = 2*m / p**2 * (p-x[front])
        slope[~front] = 2*m / (1-p)**2 * (p-x[~front])
    angle = np.arctan(slope)
    upper = np.column_stack((x - yt*np.sin(angle), yc + yt*np.cos(angle)))
    lower = np.column_stack((x + yt*np.sin(angle), yc - yt*np.cos(angle)))
    if trailing_edge == "closed":
        upper[-1] = lower[-1] = (1, 0)
    return np.vstack((upper[::-1], lower[1:]))


def read_coordinates(path, file_format):
    rows = []
    header_seen = False
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            values = [float(v.replace("D", "E").replace("d", "e")) for v in line.split()]
        except ValueError:
            require(not rows and not header_seen, f"Invalid coordinate text on line {number}")
            header_seen = True
            continue
        require(len(values) == 2, f"Expected two columns on line {number}")
        require(all(math.isfinite(v) for v in values), f"Nonfinite coordinate on line {number}")
        rows.append(values)
    require(len(rows) >= 3, "Coordinate file contains too few rows")
    points = np.asarray(rows)
    if file_format == "selig":
        return points
    require(file_format == "lednicer", "Unknown coordinate format")
    counts = points[0]
    require(np.all(counts == np.floor(counts)) and np.all(counts >= 2),
            "Lednicer first numeric row must contain two integer point counts")
    n_upper, n_lower = map(int, counts)
    require(len(points) == 1 + n_upper + n_lower, "Lednicer point counts do not match the data")
    upper, lower = points[1:1+n_upper], points[1+n_upper:]
    # The common LE is the closest endpoint pair. Finite TE gaps remain intact.
    candidates = [(float(np.linalg.norm(upper[i] - lower[j])), i, j)
                  for i in (0, -1) for j in (0, -1)]
    distance, i, j = min(candidates, key=lambda item: (item[0], (upper[item[1], 0] + lower[item[2], 0])/2))
    span = float(np.max(np.ptp(points[1:], axis=0)))
    require(span > 0 and distance <= 1e-8 * span, "Lednicer branches must share a leading edge")
    # If both LE and TE coincide, the endpoint with smaller x is the LE.
    shared = [item for item in candidates if item[0] <= 1e-8 * span]
    if len(shared) > 1:
        _, i, j = min(shared, key=lambda item: (upper[item[1], 0] + lower[item[2], 0])/2)
    upper = upper if i == -1 else upper[::-1]
    lower = lower if j == 0 else lower[::-1]
    return np.vstack((upper, lower[1:]))


def normalize_coordinates(points):
    require(np.isfinite(points).all(), "Coordinates must be finite")
    # Selig's first and last points locate the TE, including a finite TE face.
    te = (points[0] + points[-1]) / 2
    le = points[np.argmax(np.linalg.norm(points - te, axis=1))]
    chord = float(np.linalg.norm(te - le))
    require(chord > 0, "Coordinate chord is zero")
    axis = (te - le) / chord
    normal = np.array([-axis[1], axis[0]])
    result = np.column_stack(((points-le) @ axis, (points-le) @ normal)) / chord
    require(np.max(np.linalg.norm(result[[0, -1]] - [1, 0], axis=1)) < 0.1,
            "Contour endpoints must be at the trailing edge (gap < 0.2 chord)")
    return result, dict(leading_edge=le.tolist(), trailing_edge_midpoint=te.tolist(),
                        input_chord=chord, rotation_radians=float(math.atan2(axis[1], axis[0])))


def airfoil_geometry(settings):
    source = settings["airfoil"]
    if "naca" in source:
        points = naca4(source["naca"], source["points_per_side"], source["trailing_edge"])
        transform = dict(input_chord=1.0, leading_edge=[0, 0], trailing_edge_midpoint=[1, 0], rotation_radians=0.0)
    else:
        path = Path(source["file"])
        points, transform = normalize_coordinates(read_coordinates(path, source["format"]))
        transform["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    points = clean_contour(points) * settings["chord"]
    require(np.max(np.linalg.norm(points / settings["chord"] - [0.5, 0], axis=1))
            < settings["mesh"]["farfield_radius"], "Airfoil is outside the farfield")
    return points, transform


def generate_gmsh(settings, contour, directory):
    import gmsh
    chord, mesh = settings["chord"], settings["mesh"]
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.model.add("airfoil")
        geo = gmsh.model.geo
        vertices = [geo.addPoint(float(x), float(y), 0) for x, y in contour]
        curves = [geo.addLine(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
        wall_loop = geo.addCurveLoop(curves)
        radius = mesh["farfield_radius"] * chord
        center = geo.addPoint(0.5*chord, 0, 0)
        outer = [geo.addPoint(0.5*chord + radius*math.cos(a), radius*math.sin(a), 0)
                 for a in (0, math.pi/2, math.pi, 3*math.pi/2)]
        arcs = [geo.addCircleArc(a, center, b) for a, b in zip(outer, outer[1:] + outer[:1])]
        surface = geo.addPlaneSurface([geo.addCurveLoop(arcs), wall_loop])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(1, curves, name="airfoil")
        gmsh.model.addPhysicalGroup(1, arcs, name="farfield")
        gmsh.model.addPhysicalGroup(2, [surface], name="fluid")
        field = gmsh.model.mesh.field

        def threshold(curve_list, near_size, near_distance, far_distance):
            distance = field.add("Distance")
            field.setNumbers(distance, "CurvesList", curve_list)
            field.setNumber(distance, "Sampling", 30)
            tag = field.add("Threshold")
            for key, value in dict(InField=distance, SizeMin=near_size*chord*mesh["size_factor"],
                                   SizeMax=mesh["farfield_size"]*chord*mesh["size_factor"],
                                   DistMin=near_distance*chord, DistMax=far_distance*chord).items():
                field.setNumber(tag, key, value)
            return tag

        body_field = threshold(curves, mesh["surface_size"], 0.02, 3.0)
        edge_curves = [tag for k, tag in enumerate(curves)
                       if min(contour[k, 0], contour[(k+1) % len(contour), 0]) < 0.03*chord
                       or max(contour[k, 0], contour[(k+1) % len(contour), 0]) > 0.97*chord]
        edge_field = threshold(edge_curves, mesh["edge_size"], 0.005, 0.5)
        wake_field = field.add("Box")
        for key, value in dict(VIn=mesh["wake_size"]*chord*mesh["size_factor"],
                               VOut=mesh["farfield_size"]*chord*mesh["size_factor"],
                               XMin=chord, XMax=(1+mesh["wake_length"])*chord,
                               YMin=-mesh["wake_half_width"]*chord, YMax=mesh["wake_half_width"]*chord,
                               ZMin=-chord, ZMax=chord, Thickness=0.5*chord).items():
            field.setNumber(wake_field, key, value)
        combined = field.add("Min")
        field.setNumbers(combined, "FieldsList", [body_field, edge_field, wake_field])
        field.setAsBackgroundMesh(combined)
        if settings.get("mode") == "rans":
            layer = settings["boundary_layer"]
            bl = field.add("BoundaryLayer")
            field.setNumbers(bl, "CurvesList", curves)
            for key, value in dict(Size=layer["first_height"]*chord,
                                   Ratio=layer["growth_ratio"], Thickness=layer["thickness"]*chord,
                                   SizeFar=mesh["farfield_size"]*mesh["size_factor"]*chord,
                                   Quads=1).items():
                field.setNumber(bl, key, value)
            # Convex sharp corners (including both corners of a blunt TE) need fans.
            incoming = contour - np.roll(contour, 1, axis=0)
            outgoing = np.roll(contour, -1, axis=0) - contour
            turning = np.arctan2(cross(incoming, outgoing), np.sum(incoming*outgoing, axis=1))
            fans = [vertices[i] for i in np.flatnonzero(turning > math.pi/6)]
            field.setNumbers(bl, "FanPointsList", fans)
            field.setNumbers(bl, "FanPointsSizesList", [layer["fan_elements"]]*len(fans))
            field.setAsBoundaryLayer(bl)
        gmsh.model.mesh.generate(2)
        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(2)
        require(set(element_types) <= {2, 3}, "Expected only first-order triangles and quads")
        require(settings.get("mode") != "rans" or 3 in element_types,
                "RANS boundary layer did not produce quadrilaterals")
        counts = dict(cells=sum(len(tags) for tags in element_tags))
        # Count only nodes referenced by fluid elements (exclude circle construction points).
        counts["nodes"] = len(set(map(int, np.concatenate(element_nodes))))
        counts["markers"] = {name: sum(len(gmsh.model.mesh.getElements(1, tag)[1][0]) for tag in tags)
                             for name, tags in (("airfoil", curves), ("farfield", arcs))}
        gmsh.write(str(directory / "mesh.msh"))
        gmsh.write(str(directory / "mesh.su2"))
        return counts, gmsh.__version__
    finally:
        gmsh.finalize()


def read_su2(path):
    with path.open() as stream:
        lines = iter(line.strip() for line in stream if line.strip() and not line.lstrip().startswith("%"))

        def header(key):
            line = next(lines)
            name, separator, value = line.partition("=")
            require(separator and name.strip() == key, f"Expected SU2 {key}, got {line}")
            return value.strip()

        require(header("NDIME") == "2", "SU2 mesh must have NDIME=2")
        triangles, quads = [], []
        for _ in range(int(header("NELEM"))):
            row = next(lines).split()
            require(row[0] in ("5", "9"), "SU2 element is not a linear triangle or quadrilateral")
            n = 3 if row[0] == "5" else 4
            require(len(row) in (n+1, n+2), "Invalid SU2 element row")
            (triangles if n == 3 else quads).append(list(map(int, row[1:n+1])))
        points = []
        for i in range(int(header("NPOIN"))):
            row = next(lines).split()
            require(len(row) in (2, 3), "Invalid SU2 point row")
            if len(row) == 3:
                require(int(row[2]) == i, "SU2 point indices must be contiguous and zero-based")
            points.append(list(map(float, row[:2])))
        markers = {}
        for _ in range(int(header("NMARK"))):
            name = header("MARKER_TAG")
            require(name not in markers, "Duplicate SU2 marker")
            edges = []
            for _ in range(int(header("MARKER_ELEMS"))):
                row = next(lines).split()
                require(row[0] == "3" and len(row) == 3, "Invalid SU2 boundary edge")
                edges.append(list(map(int, row[1:])))
            markers[name] = np.asarray(edges, dtype=int).reshape(-1, 2)
        require(next(lines, None) is None, "Unexpected data after SU2 boundaries")
    cells = np.asarray(triangles, dtype=int).reshape(-1, 3)
    if quads:
        cells = {3: cells, 4: np.asarray(quads, dtype=int).reshape(-1, 4)}
    return np.asarray(points), cells, markers


def cell_blocks(cells):
    """Retain the triangle-only array API; mixed meshes use {3: triangles, 4: quads}."""
    return cells if isinstance(cells, dict) else {3: cells}


def connected(vertices, edges):
    adjacency = defaultdict(list)
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    seen, pending = set(), [int(next(iter(vertices)))]
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(n for n in adjacency[node] if n not in seen)
    return len(seen) == len(vertices)


def validate_mesh(points, triangles, markers, contour, expected, chord):
    from matplotlib.path import Path as PolygonPath
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) > 0
            and np.isfinite(points).all(), "Invalid mesh nodes")
    blocks = cell_blocks(triangles)
    require(set(blocks) <= {3, 4}, "Unsupported cell type")
    total = sum(len(cells) for cells in blocks.values())
    require(total > 0, "No cells")
    areas, centroids, edge_blocks, quad_jacobians, aspect_ratios = [], [], [], [], []
    quality = np.array([])
    for n, cells in blocks.items():
        require(cells.ndim == 2 and cells.shape[1] == n, "Invalid cell connectivity")
        if not len(cells):
            continue
        require(cells.min() >= 0 and cells.max() < len(points), "Element node index out of bounds")
        require(len(np.unique(np.sort(cells, axis=1), axis=0)) == len(cells),
                "Duplicate triangles" if n == 3 else "Duplicate quadrilaterals")
        p = points[cells] / chord
        edges = np.roll(p, -1, axis=1) - p
        # Local differences avoid cancellation for very thin cells far from the origin.
        a = cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])/2
        if n == 4:
            a += cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0])/2
        require(np.all(a > 1e-20), "Mesh has inverted or degenerate cells")
        lengths = np.linalg.norm(edges, axis=2)
        require(np.all(lengths > 0), "Degenerate cell edge")
        if n == 3:
            quality = 4*math.sqrt(3)*a/np.sum(lengths**2, axis=1)
        else:
            jac = cross(edges, np.roll(edges, -1, axis=1)) / (lengths*np.roll(lengths, -1, axis=1))
            require(np.all(jac > 1e-10), "Quadrilateral has inverted / concave corners")
            quad_jacobians.extend(np.min(jac, axis=1))
        aspect_ratios.extend(np.max(lengths, axis=1)/np.min(lengths, axis=1))
        areas.append(a)
        centroids.append(np.mean(p, axis=1))
        edge_blocks.append(np.stack((cells, np.roll(cells, -1, axis=1)), axis=2).reshape(-1, 2))
    area = np.concatenate(areas)
    directed = np.concatenate(edge_blocks)
    require(len(np.unique(directed)) == len(points), "Mesh has isolated nodes")
    require(len(np.unique(points, axis=0)) == len(points), "Duplicate mesh node coordinates")
    edge_counts = Counter(tuple(sorted(edge)) for edge in directed)
    require(max(edge_counts.values()) <= 2, "Non-manifold mesh edge")
    orientation = Counter()
    for a, b in directed:
        orientation[tuple(sorted((a, b)))] += 1 if a < b else -1
    require(all(orientation[edge] == 0 for edge, count in edge_counts.items() if count == 2),
            "Internal edge orientation mismatch")
    boundary = {edge for edge, count in edge_counts.items() if count == 1}
    require(set(markers) == {"airfoil", "farfield"}, "Expected exactly airfoil and farfield markers")
    all_marker_edges = []
    marker_areas = {}
    directed_boundary = {tuple(sorted((a, b))): (a, b) for a, b in directed
                         if tuple(sorted((a, b))) in boundary}
    for name, edges in markers.items():
        require(len(edges) > 2 and edges.min() >= 0 and edges.max() < len(points), f"Invalid {name} marker")
        counts = Counter(map(int, edges.ravel()))
        require(all(value == 2 for value in counts.values()) and connected(counts, edges),
                f"{name} boundary is not one closed loop")
        keys = [tuple(sorted(edge)) for edge in edges]
        require(all(edge in boundary for edge in keys), f"{name} has a non-boundary edge")
        all_marker_edges.extend(keys)
        oriented = np.asarray([directed_boundary[edge] for edge in keys])
        marker_areas[name] = float(np.sum(cross(points[oriented[:, 0]]/chord, points[oriented[:, 1]]/chord))/2)
    require(len(set(all_marker_edges)) == len(all_marker_edges), "Duplicate boundary marker edges")
    require(set(all_marker_edges) == boundary, "Missing or extra boundary marker edges")
    require(connected(range(len(points)), directed), "Fluid mesh is disconnected")
    require(marker_areas["airfoil"] < 0 < marker_areas["farfield"], "Boundary orientations / holes are incorrect")
    require(math.isclose(-marker_areas["airfoil"], signed_area(contour/chord), rel_tol=1e-8, abs_tol=1e-12),
            "Airfoil boundary area differs from input contour")
    require(math.isclose(float(np.sum(area)), sum(marker_areas.values()), rel_tol=1e-10, abs_tol=1e-10),
            "Fluid area differs from boundary area")
    polygon = PolygonPath(np.vstack((contour/chord, contour[0]/chord)))
    require(not np.any(polygon.contains_points(np.concatenate(centroids), radius=-1e-10)),
            "Cell centroids lie inside the airfoil")
    wall_nodes = set(map(int, markers["airfoil"].ravel()))
    nonwall = [i for i in range(len(points)) if i not in wall_nodes]
    require(not np.any(polygon.contains_points(points[nonwall]/chord, radius=-1e-10)),
            "Fluid nodes lie inside the airfoil")
    require(len(points) == expected["nodes"] and total == expected["cells"],
            "SU2 export changed node / element counts")
    require({k: len(v) for k, v in markers.items()} == expected["markers"], "SU2 export changed marker counts")
    warnings = []
    if len(quality) and quality.min() < 0.05:
        warnings.append("Minimum triangle quality < 0.05; inspect the leading/trailing-edge images.")
    result = dict(valid=True, nodes=len(points), cells=total,
                cell_types={"triangles": len(blocks.get(3, [])), "quadrilaterals": len(blocks.get(4, []))},
                markers={k: len(v) for k, v in markers.items()},
                min_area=float(area.min()*chord**2), fluid_area=float(area.sum()*chord**2),
                max_edge_aspect_ratio=float(max(aspect_ratios)), warnings=warnings,
                checks=["positive_areas", "unique_nodes_and_cells", "no_isolated_nodes", "edge_incidence",
                        "closed_markers", "boundary_coverage", "connected_fluid", "area_balance",
                        "airfoil_exclusion", "su2_round_trip_counts"])
    if len(quality):
        result["triangle_quality"] = dict(definition="4*sqrt(3)*area/sum(edge_length^2); equilateral=1",
                                           **distribution(quality))
    if quad_jacobians:
        result["quad_scaled_jacobian"] = distribution(quad_jacobians)
        result["checks"].append("positive_quad_corner_jacobians")
    return result


def distribution(values):
    return dict(minimum=float(np.min(values)), p01=float(np.quantile(values, 0.01)),
                median=float(np.median(values)), p99=float(np.quantile(values, 0.99)),
                maximum=float(np.max(values)))


def validate_boundary_layer(points, cells, markers, settings):
    """Measure wall-edge-to-opposite-edge normal heights, not cell-centre distances."""
    quads = cell_blocks(cells).get(4, np.empty((0, 4), dtype=int))
    owners = {}
    adjacent = defaultdict(list)
    wall = {tuple(sorted(edge)) for edge in markers["airfoil"]}
    for index, cell in enumerate(quads):
        for i in range(4):
            key = tuple(sorted((cell[i], cell[(i+1) % 4])))
            adjacent[key].append((index, i))
            if key in wall:
                owners[key] = np.roll(cell, -i)
    require(set(owners) == wall, "RANS wall edges must all touch a boundary-layer quadrilateral")
    heights, angles = [], []
    for cell in owners.values():
        a, b, c, d = points[cell]/settings["chord"]
        tangent = b-a
        normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
        heights.extend([float((d-a) @ normal), float((c-b) @ normal)])
        for direction in (d-a, c-b):
            angles.append(math.degrees(math.acos(float(np.clip(direction @ normal/np.linalg.norm(direction), -1, 1)))))
    target = settings["boundary_layer"]["first_height"]
    require(min(heights) >= 0.5*target and max(heights) <= 2*target,
            "Actual first-layer normal heights differ from target by more than a factor of two")
    layers, thicknesses = [], []
    for start in wall:
        edge, previous, count, seen = start, None, 0, set()
        while True:
            following = [(index, i) for index, i in adjacent[edge] if index != previous]
            if not following:
                break
            require(len(following) == 1, "Ambiguous boundary-layer column")
            index, i = following[0]
            require(index not in seen, "Boundary-layer column forms a cycle")
            seen.add(index)
            cell = quads[index]
            edge = tuple(sorted((cell[(i+2) % 4], cell[(i+3) % 4])))
            previous, count = index, count+1
        layers.append(count)
        thicknesses.append(np.linalg.norm(points[list(edge)].mean(axis=0)
                                         - points[list(start)].mean(axis=0))/settings["chord"])
    require(min(layers) >= 2, "Boundary-layer columns must contain at least two quadrilaterals")
    return dict(wall_quad_coverage=1.0, first_height_over_chord=distribution(heights),
                target_first_height_over_chord=target, max_wall_normal_deviation_deg=max(angles),
                quad_layers=distribution(layers), column_thickness_over_chord=distribution(thicknesses),
                note="Geometric first-cell height; y+ must be checked from the RANS solution. "
                     "Layer thickness is a Gmsh maximum and may be locally truncated.")


def plot_mesh(points, triangles, contour, settings, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    chord = settings["chord"]
    p, wall = points/chord, contour/chord
    r = settings["mesh"]["farfield_radius"]
    views = [("mesh_overview.png", (0.5-r*1.03, 0.5+r*1.03), (-r*1.03, r*1.03), "Full domain"),
             ("mesh_airfoil.png", (-0.15, 1.25), (-0.45, 0.45), "Airfoil mesh"),
             ("mesh_trailing_edge.png", (0.9, 1.12), (-0.08, 0.08), "Trailing edge")]
    if settings.get("mode") == "rans":
        views.append(("mesh_leading_edge.png", (-0.012, 0.045), (-0.035, 0.035), "Leading-edge boundary layer"))
    edges = np.concatenate([np.stack((cells, np.roll(cells, -1, axis=1)), axis=2).reshape(-1, 2)
                            for cells in cell_blocks(triangles).values()])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    for name, xlim, ylim, title in views:
        fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
        ax.add_collection(LineCollection(p[edges], colors="#587184", linewidths=0.25))
        ax.fill(wall[:, 0], wall[:, 1], color="#f0ede5", edgecolor="#c34531", linewidth=0.8, zorder=3)
        ax.set(xlim=xlim, ylim=ylim, xlabel="x / c", ylabel="y / c", title=title, aspect="equal")
        ax.set_facecolor("#fcfcfa")
        fig.savefig(directory / name, dpi=180)
        plt.close(fig)


def run(config_path):
    settings = read_settings(config_path.resolve())
    output = Path(settings["output"])
    require(not output.exists(), f"Output already exists; choose a new directory: {output}")
    contour, transform = airfoil_geometry(settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".airfoil-mesh-", dir=output.parent) as temporary:
        directory = Path(temporary)
        # Keep font caches out of the user's global configuration directory.
        os.environ.setdefault("MPLCONFIGDIR", str(directory / ".mpl-cache"))
        os.environ.setdefault("XDG_CACHE_HOME", str(directory / ".cache"))
        expected, gmsh_version = generate_gmsh(settings, contour, directory)
        points, triangles, markers = read_su2(directory / "mesh.su2")
        quality = validate_mesh(points, triangles, markers, contour, expected, settings["chord"])
        if settings.get("mode") == "rans":
            quality["boundary_layer"] = validate_boundary_layer(points, triangles, markers, settings)
            quality["checks"].extend(["wall_quad_coverage", "first_layer_normal_height"])
        import matplotlib
        quality["versions"] = dict(python=platform.python_version(), gmsh=gmsh_version,
                                    numpy=np.__version__, matplotlib=matplotlib.__version__)
        quality["geometry"] = dict(points=len(contour), area=signed_area(contour), normalization=transform)
        plot_mesh(points, triangles, contour, settings, directory)
        (directory / "quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False) + "\n")
        (directory / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
        output.mkdir()  # Atomic reservation: never overwrite an existing result.
        for artifact in directory.iterdir():
            if artifact.is_file():
                shutil.move(str(artifact), output / artifact.name)
    return output, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON settings; relative paths resolve against this file")
    args = parser.parse_args()
    try:
        output, quality = run(args.config)
    except (ValueError, OSError, RuntimeError, StopIteration, ImportError) as error:
        parser.exit(1, f"Mesh generation failed: {error}\n")
    print(f"Created {output}: {quality['nodes']} nodes, {quality['cells']} cells ({quality['cell_types']})")
    for warning in quality["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
