#!/usr/bin/env python3
"""Generate, run and inspect two-dimensional Euler or SA-RANS intake cases.

Commands: generate settings.json; run case_dir result_dir; analyze case_dir result_dir.
All lengths are metres, pressure Pa and temperature K. Results are per metre span.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from generate_airfoil_mesh import read_su2, connected, cell_blocks, validate_boundary_layer, distribution
from intake_mesh import (policy_settings, boundary_plan, size_function,
                         termination_quality, transition_quality, policy_assessment,
                         slip_partition, partition_size_function, partition_constraints, layer_rows,
                         wall_spacing_feedback, initial_wall_spacing)


DEFAULTS = {
    "kind": "intake", "mach": 5.0, "temperature": 220.0, "pressure": 2500.0,
    "gamma": 1.4, "gas_constant": 287.05, "ramp_angle_deg": 10.0,
    "ramp_length": 1.0, "outlet_x": 3.0, "upstream_length": 0.5,
    "top_height_factor": 3.0, "cowl_thickness": 0.02, "cowl_nose_length": 0.1,
    "size_factor": 1.0, "far_size": 0.04, "internal_size": 0.01,
    "shock_size": 0.005,
}
LAYER_DEFAULTS = dict(first_height=1e-6, growth_ratio=1.18, thickness=0.01, fan_elements=40)


def is_rans(s):
    return s.get("mode", "euler") == "rans"


def outlet_type(s):
    # Previously generated RANS cases specified pressure without an explicit type.
    return s.get("outlet_type", "pressure" if "outlet_pressure" in s else "extrapolate")


def external_outlet_type(s):
    return s.get("external_outlet_type", outlet_type(s))


def plot_triangles(cells):
    """Split quads for visualization only; retain the original mesh for fluxes."""
    blocks = cell_blocks(cells)
    quads = blocks.get(4, np.empty((0, 4), dtype=int))
    return np.concatenate([blocks.get(3, np.empty((0, 3), dtype=int)),
                           quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def shock(mach, theta_deg, gamma=1.4):
    """Weak attached oblique shock, with stagnation pressure loss."""
    require(mach > 1 and gamma > 1 and 0 < theta_deg < 90, "Invalid shock inputs")
    theta = math.radians(theta_deg)
    def f(b):
        return (2 / math.tan(b) * (mach**2 * math.sin(b)**2 - 1)
                / (mach**2 * (gamma + math.cos(2*b)) + 2) - math.tan(theta))
    grid = np.linspace(math.asin(1/mach) + 1e-10, math.pi/2 - 1e-10, 10001)
    indices = np.flatnonzero(np.array([f(b) for b in grid]) >= 0)
    require(len(indices) > 0, "Ramp angle has no attached weak shock")
    i = int(indices[0])
    a, b = grid[max(0, i-1)], grid[i]
    for _ in range(60):
        mid = (a+b)/2
        if f(mid) >= 0:
            b = mid
        else:
            a = mid
    beta = (a+b)/2
    mn = mach*math.sin(beta)
    pressure_ratio = 1 + 2*gamma/(gamma+1)*(mn**2-1)
    density_ratio = (gamma+1)*mn**2 / ((gamma-1)*mn**2+2)
    mn2 = math.sqrt((1+(gamma-1)/2*mn**2)/(gamma*mn**2-(gamma-1)/2))
    m2 = mn2/math.sin(beta-theta)
    p0ratio = pressure_ratio*((1+(gamma-1)/2*m2*m2)/(1+(gamma-1)/2*mach*mach))**(gamma/(gamma-1))
    return dict(beta_deg=math.degrees(beta), pressure_ratio=pressure_ratio,
                density_ratio=density_ratio, mach_downstream=m2, total_pressure_ratio=p0ratio)


def ramps(s):
    """Canonical ramps; legacy case records remain readable."""
    return s.get("ramps", [dict(angle_deg=s["ramp_angle_deg"], length=s["ramp_length"])])


def ramp_vertices(s):
    x, y = 0.0, 0.0
    points = []
    for stage in ramps(s):
        x += stage["length"]
        y += stage["length"]*math.tan(math.radians(stage["angle_deg"]))
        points.append((x, y))
    if "ramp_end_point" in s:
        points[-1] = tuple(s["ramp_end_point"])
    return points


def incident_shocks(s):
    """Sequential isolated-shock estimates, before shock interactions."""
    mach, angle, x, y = s["mach"], 0.0, 0.0, 0.0
    estimates = []
    for stage in ramps(s):
        theory = shock(mach, stage["angle_deg"]-angle, s["gamma"])
        estimates.append(dict(**theory, origin=[x, y], ray_angle_deg=angle+theory["beta_deg"]))
        mach, angle = theory["mach_downstream"], stage["angle_deg"]
        x += stage["length"]
        y += stage["length"]*math.tan(math.radians(angle))
    return estimates


def settings(path):
    raw = json.loads(path.read_text())
    require(isinstance(raw, dict), "Settings must be an object")
    require(set(raw) <= set(DEFAULTS) | {"output", "ramps", "mode", "boundary_layer",
                                       "freestream_nu_factor", "outlet_pressure", "outlet_type", "upstream_wall",
                                       "external_outlet_type", "cowl_lip_height", "mesh_policy",
                                       "initial_wall_spacing", "lip_refinement", "ramp_end",
                                       "duct_gap_height", "geometry_design"}, "Unknown settings key")
    s = DEFAULTS | raw
    if "geometry_design" in raw:
        provenance = raw["geometry_design"]
        require(isinstance(provenance, dict)
                and set(provenance) == {"gas_model", "upstream_temperature", "solver_gas_model"}
                and provenance["gas_model"] == "nasa7"
                and provenance["solver_gas_model"] == "constant_gamma",
                "Invalid geometry_design provenance")
        require(not isinstance(provenance["upstream_temperature"], bool)
                and isinstance(provenance["upstream_temperature"], (int, float))
                and math.isfinite(provenance["upstream_temperature"])
                and provenance["upstream_temperature"] == s["temperature"],
                "NASA7 geometry temperature must match freestream")
        require("ramp_end" not in raw, "NASA7 geometry must preserve its explicit endpoint")
    s["mode"] = raw.get("mode", "euler")
    require(s["mode"] in ("euler", "rans"), "mode must be euler or rans")
    if is_rans(s):
        if "mesh_policy" in raw:
            s["mesh_policy"] = policy_settings(raw["mesh_policy"])
        s["upstream_wall"] = raw.get("upstream_wall", "no_slip")
        require(s["upstream_wall"] in ("no_slip", "slip"), "Invalid upstream_wall")
        layer = raw.get("boundary_layer", {})
        require(isinstance(layer, dict) and set(layer) <= set(LAYER_DEFAULTS), "Invalid boundary_layer fields")
        layer = LAYER_DEFAULTS | layer
        for key, value in layer.items():
            require(not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and value > 0, f"Invalid boundary_layer.{key}")
        require(1.01 <= layer["growth_ratio"] <= 1.5, "Invalid boundary_layer.growth_ratio")
        require(isinstance(layer["fan_elements"], int) and 4 <= layer["fan_elements"] <= 200,
                "Invalid boundary_layer.fan_elements")
        s["boundary_layer"] = layer
        if 'initial_wall_spacing' in raw:
            seed=raw['initial_wall_spacing']
            require(isinstance(seed,dict) and set(seed)<={'method','target_y_plus'}
                    and seed.get('method')=='freestream_flat_plate_v1', 'Invalid initial_wall_spacing policy')
            require('first_height' not in raw.get('boundary_layer',{}),
                    'Choose explicit first_height or initial_wall_spacing, not both')
            s['initial_wall_spacing']=dict(method=seed['method'],target_y_plus=seed.get('target_y_plus',.5))
        s["freestream_nu_factor"] = raw.get("freestream_nu_factor", 3.0)
        s["outlet_type"] = outlet_type(s)
        require(s["outlet_type"] in ("extrapolate", "pressure"), "Invalid outlet_type")
        if "external_outlet_type" in raw:
            require(s["kind"] == "intake", "external_outlet_type requires an intake")
            require(s["external_outlet_type"] in ("extrapolate", "pressure", "farfield"),
                    "Invalid external_outlet_type")
        require(s["outlet_type"] == "pressure" or "outlet_pressure" not in raw,
                "outlet_pressure cannot be used with extrapolate")
        for key in ("freestream_nu_factor", "outlet_pressure") if s["outlet_type"] == "pressure" else ("freestream_nu_factor",):
            value = s.get(key)
            require(not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and value > 0, f"RANS requires positive {key}")
    else:
        require(not {"boundary_layer", "freestream_nu_factor", "outlet_pressure", "outlet_type", "upstream_wall",
                     "external_outlet_type", "mesh_policy", "initial_wall_spacing"} & set(raw),
                "Viscous settings require mode=rans")
    require(s["kind"] in ("intake", "wedge"), "kind must be intake or wedge")
    if "lip_refinement" in raw:
        require(s["kind"] == "intake", "lip_refinement requires an intake")
        local = raw["lip_refinement"]
        require(isinstance(local, dict) and set(local) == {"size", "radius"},
                "lip_refinement requires size and radius in metres")
        require(all(not isinstance(v, bool) and isinstance(v, (int, float))
                    and math.isfinite(v) and v > 0 for v in local.values()), "Invalid lip_refinement")
        require(local["size"] < local["radius"], "lip_refinement size must be smaller than radius")
    for key in DEFAULTS.keys() - {"kind"}:
        value = s[key]
        require(not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value > 0, f"Invalid {key}")
    require(s["mach"] > 1 and s["gamma"] > 1, "Supersonic perfect gas required")
    if "ramps" in raw:
        require(not {"ramp_angle_deg", "ramp_length"} & set(raw), "Use ramps or legacy ramp fields, not both")
        stages = raw["ramps"]
        require(isinstance(stages, list) and 1 <= len(stages) <= 6, "Specify one to six ramps")
        angle = 0
        for stage in stages:
            require(isinstance(stage, dict) and set(stage) == {"angle_deg", "length"}, "Invalid ramp fields")
            require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                        and v > 0 for v in stage.values()), "Invalid ramp values")
            require(angle < stage["angle_deg"] < 90, "Ramp angles must increase and stay below 90 degrees")
            angle = stage["angle_deg"]
        require(s["kind"] == "intake" or len(stages) == 1, "wedge supports one ramp only")
        s["ramp_angle_deg"] = stages[0]["angle_deg"]
        s["ramp_length"] = sum(stage["length"] for stage in stages)
    s["shock_estimates"] = incident_shocks(s)
    s["theory"] = shock(s["mach"], s["ramp_angle_deg"], s["gamma"])
    s["lip_y"] = s["ramp_length"]*math.tan(math.radians(s["theory"]["beta_deg"]))
    if "cowl_lip_height" in raw:
        height = s["cowl_lip_height"]
        require(s["kind"] == "intake" and not isinstance(height, bool) and isinstance(height, (int, float))
                and math.isfinite(height) and height > 0, "Invalid cowl_lip_height")
        s["lip_y"] = height
    s["ramp_y"] = ramp_vertices(s)[-1][1]
    require(s["lip_y"] > s["ramp_y"], "Ramp closes the intake opening")
    if "ramp_end" in raw:
        require(s["kind"] == "intake", "ramp_end requires an intake")
        require(s["ramp_end"] in ("lip_x", "lip_shock"), "Invalid ramp_end")
        if s["ramp_end"] == "lip_shock":
            angle = ramps(s)[-1]["angle_deg"]
            reflected = shock(s["shock_estimates"][-1]["mach_downstream"], angle, s["gamma"])
            ray_angle = angle-reflected["beta_deg"]
            require(-90 < ray_angle < 0, "Lip shock must point downstream toward the lower wall")
            ramp_slope = math.tan(math.radians(angle))
            shock_slope = math.tan(math.radians(ray_angle))
            dx = (s["lip_y"]-s["ramp_y"])/(ramp_slope-shock_slope)
            end_x = s["ramp_length"]+dx
            end_y = s["ramp_y"]+dx*ramp_slope
            require(math.isfinite(end_x) and s["ramp_length"] < end_x < s["outlet_x"],
                    "Lip shock intersection must lie before outlet_x")
            require(end_y < s["lip_y"], "Ramp closes the intake opening")
            s["ramp_end_point"] = [end_x, end_y]
            s["lip_shock_estimate"] = dict(**reflected, origin=[s["ramp_length"], s["lip_y"]],
                                            ray_angle_deg=ray_angle)
            s["ramp_y"] = end_y
    if "duct_gap_height" in raw:
        height = s["duct_gap_height"]
        require(s["kind"] == "intake" and "ramp_end" not in raw,
                "duct_gap_height requires an intake without ramp_end")
        require(not isinstance(height, bool) and isinstance(height, (int, float))
                and math.isfinite(height) and 0 < height <= s["lip_y"]-s["ramp_y"],
                "duct_gap_height must be positive and no greater than the lip opening")
        slope = math.tan(math.radians(ramps(s)[-1]["angle_deg"]))
        end_y = s["lip_y"]-height
        end_x = s["ramp_length"]+(end_y-s["ramp_y"])/slope
        require(s["ramp_length"] <= end_x < s["outlet_x"], "Duct ramp endpoint must lie before outlet_x")
        s["ramp_end_point"] = [end_x, end_y]
        s["ramp_y"] = end_y
    s["top_y"] = s["lip_y"]*s["top_height_factor"]
    require(s["outlet_x"] > s["ramp_length"] + s["cowl_nose_length"], "Outlet too close")
    require(s["top_y"] > s["lip_y"] + s["cowl_thickness"], "Top intersects cowl")
    if is_rans(s):
        if 'initial_wall_spacing' in s:
            seed=initial_wall_spacing(s['mach'],s['pressure'],s['temperature'],s['ramp_length'],
                                      s['gamma'],s['gas_constant'],s['initial_wall_spacing']['target_y_plus'])
            s['initial_wall_spacing_report']=seed
            s['boundary_layer']['first_height']=seed['first_height_m']
        require(s['boundary_layer']['thickness']>=5*s['boundary_layer']['first_height'], 'Boundary layer too thin')
        gap = min(s["lip_y"]-s["ramp_y"], s["top_y"]-s["lip_y"]-s["cowl_thickness"])
        require(s["boundary_layer"]["thickness"] < 0.25*gap, "Boundary layer too thick for opening")
    if s["kind"] == "wedge":
        require(s["top_y"] > s["outlet_x"]*math.tan(math.radians(s["theory"]["beta_deg"])),
                "Wedge shock intersects top boundary; increase top_height_factor")
    require(isinstance(s.get("output"), str) and s["output"], "output path required")
    s["output"] = str((path.resolve().parent/s["output"]).resolve())
    return s


def polygon(s):
    """Counterclockwise fluid polygon, with one marker per following edge."""
    L, x, h, y, top = (s[k] for k in ("ramp_length", "outlet_x", "lip_y", "ramp_y", "top_y"))
    u = -s["upstream_length"]
    upstream = "upstream" if is_rans(s) and s.get("upstream_wall") == "slip" else "lower"
    if s["kind"] == "wedge":
        return np.array([(u, 0), (0, 0), (x, x*y/L), (x, top), (u, top)]), [
            upstream, "lower", "outlet", "farfield", "inlet"]
    return np.array([(u, 0), (0, 0), *ramp_vertices(s), (x, y), (x, h), (L, h),
                     (L+s["cowl_nose_length"], h+s["cowl_thickness"]),
                     (x, h+s["cowl_thickness"]), (x, top), (u, top)]), [
        upstream, *["lower"]*(len(ramps(s))+1), "outlet", "cowl", "cowl", "cowl",
        "external_outlet", "farfield", "inlet"]


def mesh_quality(points, cells, markers, poly, expected):
    require(np.isfinite(points).all(), "Nonfinite mesh")
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    blocks = cell_blocks(cells)
    require(set(blocks) <= {3, 4} and sum(map(len, blocks.values())) > 0, "Triangles/quads required")
    areas, centroids, edge_blocks, jacobians, shapes = [], [], [], [], []
    for n, block in blocks.items():
        require(block.ndim == 2 and block.shape[1] == n, "Invalid connectivity")
        if not len(block):
            continue
        require(block.min() >= 0 and block.max() < len(points), "Invalid node index")
        require(len(np.unique(np.sort(block, axis=1), axis=0)) == len(block), "Duplicate elements")
        q = points[block]
        delta = np.roll(q, -1, axis=1)-q
        lengths = np.linalg.norm(delta, axis=2)
        area = cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])/2
        if n == 4:
            area += cross(q[:, 2]-q[:, 0], q[:, 3]-q[:, 0])/2
            require(np.all(lengths > 0), "Degenerate edge")
            jac = cross(delta, np.roll(delta, -1, axis=1))/(lengths*np.roll(lengths, -1, axis=1))
            require(np.all(jac > 1e-10), "Inverted or concave quadrilateral")
            jacobians.extend(jac.min(axis=1))
        else:
            shapes.extend(4*math.sqrt(3)*area/np.sum(lengths**2, axis=1))
        require(np.all(area > 0), "Nonpositive element")
        areas.extend(area)
        centroids.extend(q.mean(axis=1))
        edge_blocks.append(np.stack([block, np.roll(block, -1, axis=1)], axis=2).reshape(-1, 2))
    area = np.array(areas)
    edges = np.concatenate(edge_blocks)
    counts = Counter(map(tuple, np.sort(edges, axis=1)))
    require(max(counts.values()) <= 2, "Nonmanifold edge")
    orientation = Counter()
    for a, b in edges:
        orientation[tuple(sorted((a, b)))] += 1 if a < b else -1
    require(all(orientation[e] == 0 for e, n in counts.items() if n == 2), "Internal edge orientation mismatch")
    boundary = {e for e, n in counts.items() if n == 1}
    marked = Counter(tuple(sorted(e)) for es in markers.values() for e in es)
    require(set(markers) == set(expected), "Unexpected boundary markers")
    require(set(marked) == boundary and max(marked.values()) == 1, "Boundary coverage mismatch")
    degrees = Counter(n for e in boundary for n in e)
    require(set(degrees.values()) == {2}, "Boundary is not closed")
    require(connected(set(degrees), boundary), "Disconnected boundary")
    require(connected(set(range(len(points))), edges), "Disconnected fluid mesh")
    require(len(np.unique(points, axis=0)) == len(points), "Duplicate nodes")
    expected_area = abs(np.sum(cross(poly, np.roll(poly, -1, axis=0))))/2
    require(abs(area.sum()-expected_area) < 1e-10*expected_area, "Fluid area mismatch")
    from matplotlib.path import Path as Polygon
    require(Polygon(poly).contains_points(np.array(centroids), radius=1e-10).all(), "Element outside fluid")
    result = dict(nodes=len(points), triangles=len(blocks.get(3, [])), quadrilaterals=len(blocks.get(4, [])),
                  area=float(area.sum()), min_area=float(area.min()),
                  min_triangle_quality=float(min(shapes)) if shapes else None,
                  markers={k: len(v) for k, v in markers.items()})
    if jacobians:
        result["quad_scaled_jacobian"] = distribution(jacobians)
    return result


def wall_layer_quality(points, cells, markers, s):
    wall = np.concatenate([edges for name, edges in markers.items() if name in ("lower", "cowl")])
    quality = validate_boundary_layer(points, cells, {"airfoil": wall},
                                      dict(chord=1.0, boundary_layer=s["boundary_layer"]))
    quality["note"] = ("Geometric first-cell height; check y+ from the RANS solution. "
                       "Corners and open-boundary grading can change column thickness; inspect measured values.")
    return {key.replace("_over_chord", "_m"): value for key, value in quality.items()}


def generate(path):
    import gmsh
    s = settings(path)
    out = Path(s["output"])
    out.mkdir(parents=True, exist_ok=False)
    poly, names = polygon(s)
    constraints = []
    geometry_size = None
    partition = None
    if "mesh_policy" in s:
        geometry_size = size_function(poly, names, s["boundary_layer"], s["mesh_policy"])
        if s["mesh_policy"]["name"] == "geometry_v2":
            poly, names, partition = slip_partition(poly, names, s["boundary_layer"], s["mesh_policy"])
            if partition:
                geometry_size = partition_size_function(geometry_size, partition, s["boundary_layer"], s["mesh_policy"])
        poly, names, constraints = boundary_plan(poly, names, s["boundary_layer"],
                                                 skip_markers={"upstream"} if partition else ())
    factor = s["size_factor"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("intake")
        geo = gmsh.model.geo
        vertices = [geo.addPoint(float(x), float(y), 0) for x, y in poly]
        lines = [geo.addLine(v, vertices[(i+1) % len(vertices)]) for i, v in enumerate(vertices)]
        internal_lines = []
        if partition:
            tip = int(np.argmin(np.linalg.norm(poly-partition["origin"], axis=1)))
            end = int(np.argmin(np.linalg.norm(poly-partition["end"], axis=1)))
            base = int(np.argmin(np.linalg.norm(poly-partition["block_base"], axis=1)))
            mid = geo.addPoint(*partition["layer_end"], 0)
            outer = geo.addPoint(*partition["block_top"], 0)
            internal_lines = [geo.addLine(vertices[tip], mid), geo.addLine(mid, vertices[end])]
            block_top = geo.addLine(mid, outer)
            block_left = geo.addLine(vertices[base], outer)
            def path_edges(start, stop):
                indices = []
                while start != stop:
                    indices.append(lines[start])
                    start = (start+1) % len(lines)
                return indices
            surface = geo.addPlaneSurface([geo.addCurveLoop(path_edges(tip, end)+[-l for l in internal_lines[::-1]])])
            upstream_surface = geo.addPlaneSurface([geo.addCurveLoop(path_edges(end, base)+[block_left, -block_top, internal_lines[1]])])
            block_surface = geo.addPlaneSurface([geo.addCurveLoop(path_edges(base, tip)+[internal_lines[0], block_top, -block_left])])
            surfaces = [surface, upstream_surface, block_surface]
        else:
            surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
            surfaces = [surface]
        geo.synchronize()
        if partition:
            count, ratio = len(layer_rows(s["boundary_layer"])), s["boundary_layer"]["growth_ratio"]
            gmsh.model.mesh.setTransfiniteCurve(internal_lines[0], count, "Progression", ratio)
            gmsh.model.mesh.setTransfiniteCurve(block_left, count, "Progression", ratio)
            gmsh.model.mesh.setTransfiniteCurve(lines[base], partition["block_width_nodes"],
                                               "Progression", 1/ratio)
            gmsh.model.mesh.setTransfiniteCurve(block_top, partition["block_width_nodes"],
                                               "Progression", ratio)
            gmsh.model.mesh.setTransfiniteSurface(block_surface)
            gmsh.model.mesh.setRecombine(2, block_surface)
        for constraint in constraints:
            gmsh.model.mesh.setTransfiniteCurve(lines[constraint["edge"]], constraint["count"],
                                               "Progression", constraint["progression"])
        for name in sorted(set(names)):
            tag = gmsh.model.addPhysicalGroup(1, [l for l, n in zip(lines, names) if n == name])
            gmsh.model.setPhysicalName(1, tag, name)
        gmsh.model.addPhysicalGroup(2, surfaces, name="fluid")
        if partition:
            constraints.extend(partition_constraints(partition, s["boundary_layer"], s["mesh_policy"]))
        # Distance from analytic incident and reflected shock lines; field only, no internal boundary.
        rays = []
        estimates = incident_shocks(s)
        def distance_line(origin, angle):
            slope = math.tan(math.radians(angle))
            return f"Abs(y-({origin[1]:.16g}+({slope:.16g})*(x-{origin[0]:.16g})))/{math.sqrt(1+slope*slope):.16g}"
        for estimate in estimates:
            rays.append(distance_line(estimate["origin"], estimate["ray_angle_deg"]))
        if s["kind"] == "intake":
            # A refinement guide only: actual interactions are resolved by the solver.
            try:
                angle = ramps(s)[-1]["angle_deg"]
                reflected = shock(estimates[-1]["mach_downstream"], angle, s["gamma"])
                rays.append(distance_line([s["ramp_length"], s["lip_y"]], angle-reflected["beta_deg"]))
            except ValueError:
                pass  # Detached reflection: retain incident, corner and interior refinement.
        distance = rays[0]
        for ray in rays[1:]:
            distance = f"Min({distance},{ray})"
        field = gmsh.model.mesh.field
        shock_field = field.add("MathEval")
        field.setString(shock_field, "F", f"Min({s['far_size']*factor}, {s['shock_size']*factor}+{0.18*factor}*({distance}))")
        box = field.add("Box")
        for key, value in dict(VIn=s["internal_size"]*factor, VOut=s["far_size"]*factor,
                               XMin=0, XMax=s["outlet_x"], YMin=0,
                               YMax=s["lip_y"]+0.05*(s["ramp_length"] if geometry_size else 1),
                               Thickness=0.1*(s["ramp_length"] if geometry_size else 1)).items():
            field.setNumber(box, key, value)
        corners = field.add("Distance")
        field.setNumbers(corners, "PointsList", vertices)
        threshold = field.add("Threshold")
        for key, value in dict(InField=corners, SizeMin=s["shock_size"]*factor/2,
                               SizeMax=s["far_size"]*factor,
                               DistMin=0.005*(s["ramp_length"] if geometry_size else 1),
                               DistMax=0.12*(s["ramp_length"] if geometry_size else 1)).items():
            field.setNumber(threshold, key, value)
        minimum = field.add("Min")
        fields = [shock_field, box, threshold]
        if "lip_refinement" in s:
            # Absolute metre sizes, independent of size_factor. Geometry is unchanged.
            local = s["lip_refinement"]
            lip_index = int(np.argmin(np.linalg.norm(poly-[s["ramp_length"], s["lip_y"]], axis=1)))
            lip_distance = field.add("Distance")
            field.setNumbers(lip_distance, "PointsList", [vertices[lip_index]])
            lip_field = field.add("Threshold")
            for key, value in dict(InField=lip_distance, SizeMin=local["size"],
                                   SizeMax=s["far_size"]*factor, DistMin=local["radius"],
                                   DistMax=2*local["radius"], StopAtDistMax=1).items():
                field.setNumber(lip_field, key, value)
            fields.append(lip_field)
        field.setNumbers(minimum, "FieldsList", fields)
        field.setAsBackgroundMesh(minimum)
        if is_rans(s):
            layer = s["boundary_layer"]
            bl = field.add("BoundaryLayer")
            if partition:
                field.setNumbers(bl, "ExcludedSurfacesList", [upstream_surface, block_surface])
            field.setNumbers(bl, "CurvesList", [line for line, name in zip(lines, names)
                                                if name in ("lower", "cowl")])
            for key, value in dict(Size=layer["first_height"], Ratio=layer["growth_ratio"],
                                   Thickness=layer["thickness"], SizeFar=s["far_size"]*factor, Quads=1).items():
                field.setNumber(bl, key, value)
            # Fluid polygon is CCW: negative turns are convex solid corners.
            incoming, outgoing = poly-np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0)-poly
            turning = np.arctan2(incoming[:, 0]*outgoing[:, 1]-incoming[:, 1]*outgoing[:, 0],
                                 np.sum(incoming*outgoing, axis=1))
            fans = [vertices[i] for i, turn in enumerate(turning) if turn < -math.pi/6
                    and names[i] in ("lower", "cowl") and names[i-1] in ("lower", "cowl")]
            field.setNumbers(bl, "FanPointsList", fans)
            field.setNumbers(bl, "FanPointsSizesList", [layer["fan_elements"]]*len(fans))
            field.setAsBoundaryLayer(bl)
            # Grade open boundary curves at wall intersections as well. Otherwise
            # Gmsh stretches the end of a micron-high wall cell to a far-field edge.
            endpoints = {}
            for i, (line, name) in enumerate(zip(lines, names)):
                if name in ("lower", "cowl") or partition and name == "upstream":
                    continue
                ends, projections = [], []
                tangent = poly[(i+1) % len(poly)]-poly[i]
                tangent /= np.linalg.norm(tangent)
                def add_end(point, wall_direction):
                    wall_direction = wall_direction/np.linalg.norm(wall_direction)
                    projection = abs(tangent[0]*wall_direction[1]-tangent[1]*wall_direction[0])
                    require(projection > 1e-8, "Wall-layer termination is tangential to the adjacent boundary")
                    ends.append(point)
                    projections.append(projection)
                if names[i-1] in ("lower", "cowl"):
                    add_end(poly[i], poly[i]-poly[i-1])
                if names[(i+1) % len(names)] in ("lower", "cowl"):
                    add_end(poly[(i+1) % len(poly)], poly[(i+2) % len(poly)]-poly[(i+1) % len(poly)])
                if ends:
                    endpoints[line] = (np.array(ends), np.array(projections))
            def boundary_size(dim, tag, x, y, z, lc):
                if geometry_size is not None:
                    lc = min(lc, geometry_size([x, y]))
                if dim == 1 and tag in endpoints:
                    ends, projections = endpoints[tag]
                    distance = np.linalg.norm(ends-[x, y], axis=1)
                    sizes = layer["first_height"]/projections+(layer["growth_ratio"]-1)*distance
                    return min(lc, sizes.min())
                return lc
            gmsh.model.mesh.setSizeCallback(boundary_size)
        for key, value in {"Mesh.Algorithm": 6, "Mesh.ElementOrder": 1,
                           "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.MeshSizeFromPoints": 0,
                           "Mesh.MeshSizeFromCurvature": 0, "Mesh.RandomSeed": 1}.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out/"mesh.su2"))
        gmsh.write(str(out/"mesh.msh"))
        s["gmsh_version"] = gmsh.__version__
    finally:
        gmsh.finalize()
    points, cells, markers = read_su2(out/"mesh.su2")
    quality = mesh_quality(points, cells, markers, poly, names)
    quality["mesh_sha256"] = hashlib.sha256((out/"mesh.su2").read_bytes()).hexdigest()
    if is_rans(s):
        quality["boundary_layer"] = wall_layer_quality(points, cells, markers, s)
        if 'initial_wall_spacing_report' in s:
            quality['initial_wall_spacing']=s['initial_wall_spacing_report']
        quality["layer_transition"] = transition_quality(points, cells)
        if constraints:
            quality["layer_termination"] = termination_quality(points, markers, constraints, s["boundary_layer"])
            quality["mesh_policy"] = s["mesh_policy"]
            quality["policy_assessment"] = policy_assessment(quality["layer_transition"], s["mesh_policy"])
            if partition:
                quality["internal_partition"] = partition
    (out/"settings.json").write_text(json.dumps(s, indent=2)+"\n")
    replay = {k: s[k] for k in DEFAULTS}
    replay["mode"] = s["mode"]
    if "geometry_design" in s:
        replay["geometry_design"] = s["geometry_design"]
    if "ramp_end" in s:
        replay["ramp_end"] = s["ramp_end"]
    if "duct_gap_height" in s:
        replay["duct_gap_height"] = s["duct_gap_height"]
    if "cowl_lip_height" in s:
        replay["cowl_lip_height"] = s["cowl_lip_height"]
    if "lip_refinement" in s:
        replay["lip_refinement"] = s["lip_refinement"]
    if is_rans(s):
        replay.update({k: s[k] for k in ("boundary_layer", "freestream_nu_factor", "outlet_type", "outlet_pressure",
                                       "upstream_wall", "external_outlet_type", "mesh_policy") if k in s})
        if 'initial_wall_spacing' in s:
            replay['initial_wall_spacing']=s['initial_wall_spacing']
            replay['boundary_layer']=dict(s['boundary_layer'])
            replay['boundary_layer'].pop('first_height')
    if "ramps" in s:
        replay.pop("ramp_angle_deg")
        replay.pop("ramp_length")
        replay["ramps"] = s["ramps"]
    replay["output"] = "../"+out.name+"-recreated"
    (out/"input.json").write_text(json.dumps(replay, indent=2)+"\n")
    (out/"quality.json").write_text(json.dumps(quality, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    from matplotlib.collections import PolyCollection
    for block in cell_blocks(cells).values():
        ax.add_collection(PolyCollection(points[block], facecolors="none", edgecolors="#476b83", linewidths=0.15))
    ax.autoscale()
    ax.set(xlabel="x [m]", ylabel="y [m]", title=f"{s['kind']} mesh: {len(points):,} nodes")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out/"mesh.png", dpi=180)
    if is_rans(s):
        height = s["boundary_layer"]["thickness"]
        ax.set(xlim=(-0.025, 0.05), ylim=(-height*0.1, height*2), title="Ramp boundary-layer mesh")
        fig.tight_layout()
        fig.savefig(out/"mesh-wall.png", dpi=180)
        if s["kind"] == "intake":
            ax.set(xlim=(s["ramp_length"]-3*height, s["ramp_length"]+5*height),
                   ylim=(s["lip_y"]-3*height, s["lip_y"]+3*height), title="Cowl boundary-layer mesh")
            fig.tight_layout()
            fig.savefig(out/"mesh-cowl.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"directory": str(out), **quality}), flush=True)
    return out


def numerical_controls(overrides=None, inherited=None):
    """Explicit experimental controls, inherited on same-mesh continuation."""
    values=dict(inherited or {})
    values.update(overrides or {})
    require(set(values)<= {'flow_limiter','venkat_coefficient','max_update_flow','cfl_reduction_turb'}, 'Unknown numerical control')
    if 'flow_limiter' in values:
        require(values['flow_limiter'] in ('VENKATAKRISHNAN','BARTH_JESPERSEN'), 'Unsupported flow limiter')
    if 'venkat_coefficient' in values:
        coefficient=values['venkat_coefficient']
        require(isinstance(coefficient,(int,float)) and not isinstance(coefficient,bool)
                and math.isfinite(coefficient) and coefficient>0, 'Invalid Venkat coefficient')
    if 'max_update_flow' in values:
        fraction=values['max_update_flow']
        require(isinstance(fraction,(int,float)) and not isinstance(fraction,bool)
                and math.isfinite(fraction) and 0 < fraction <= 1, 'Invalid flow update fraction')
    if 'cfl_reduction_turb' in values:
        fraction=values['cfl_reduction_turb']
        require(isinstance(fraction,(int,float)) and not isinstance(fraction,bool)
                and math.isfinite(fraction) and 0 < fraction <= 1, 'Invalid turbulence CFL factor')
    return values


def configuration(case, s, iterations, second_order, cfl, numerics=None):
    numerics=numerical_controls(numerics)
    require(is_rans(s) or not numerics, 'Experimental numerical controls require RANS')
    ext = ", external_outlet" if s["kind"] == "intake" else ""
    wall = "lower, cowl" if s["kind"] == "intake" else "lower"
    u = s["mach"]*math.sqrt(s["gamma"]*s["gas_constant"]*s["temperature"])
    values = dict(SOLVER="EULER", MATH_PROBLEM="DIRECT", RESTART_SOL="YES" if second_order else "NO",
                  SYSTEM_MEASUREMENTS="SI", FLUID_MODEL="IDEAL_GAS", GAMMA_VALUE=s["gamma"],
                  GAS_CONSTANT=s["gas_constant"], MACH_NUMBER=s["mach"], AOA=0,
                  FREESTREAM_PRESSURE=s["pressure"], FREESTREAM_TEMPERATURE=s["temperature"],
                  REF_DIMENSIONALIZATION="DIMENSIONAL", REF_LENGTH=s["ramp_length"], REF_AREA=1,
                  MARKER_EULER=f"( {wall} )", MARKER_FAR="( farfield )",
                  MARKER_SUPERSONIC_INLET=f"( inlet, {s['temperature']}, {s['pressure']}, {u}, 0, 0 )",
                  MARKER_SUPERSONIC_OUTLET=f"( outlet{ext} )", MARKER_PLOTTING=f"( {wall} )",
                  MARKER_MONITORING=f"( {wall} )", MARKER_ANALYZE=f"( inlet, outlet, farfield{ext} )",
                  MARKER_ANALYZE_AVERAGE="MASSFLUX", NUM_METHOD_GRAD="WEIGHTED_LEAST_SQUARES",
                  CONV_NUM_METHOD_FLOW="HLLC", MUSCL_FLOW="YES" if second_order else "NO",
                  SLOPE_LIMITER_FLOW="VENKATAKRISHNAN", VENKAT_LIMITER_COEFF=0.05,
                  LIMITER_ITER=1500,
                  TIME_DISCRE_FLOW="EULER_IMPLICIT", CFL_NUMBER=cfl, CFL_ADAPT="NO", MGLEVEL=0,
                  LINEAR_SOLVER="FGMRES", LINEAR_SOLVER_PREC="ILU", LINEAR_SOLVER_ERROR=0.01,
                  LINEAR_SOLVER_ITER=20, ITER=iterations, CONV_FIELD="( RMS_DENSITY, RMS_ENERGY )",
                  CONV_RESIDUAL_MINVAL=-16, CONV_STARTITER=iterations+1,
                  MESH_FORMAT="SU2", MESH_FILENAME=str((case/"mesh.su2").resolve()),
                  TABULAR_FORMAT="CSV", CONV_FILENAME="history",
                  HISTORY_OUTPUT="( ITER, RMS_RES, FLOW_COEFF_SURF, NONPHYSICAL_POINTS )",
                  SCREEN_OUTPUT="( INNER_ITER, RMS_DENSITY, RMS_ENERGY, NONPHYSICAL_POINTS )",
                  SCREEN_WRT_FREQ_INNER=100, HISTORY_WRT_FREQ_INNER=1,
                  OUTPUT_FILES="( RESTART, RESTART_ASCII, PARAVIEW, SURFACE_CSV )",
                  OUTPUT_WRT_FREQ=iterations, RESTART_FILENAME="restart_flow",
                  SOLUTION_FILENAME="restart_flow", READ_BINARY_RESTART="YES", VOLUME_FILENAME="flow",
                  SURFACE_FILENAME="surface_flow", VOLUME_OUTPUT="( COORDINATES, SOLUTION, PRIMITIVE )")
    if is_rans(s):
        values.pop("MARKER_EULER")
        if s.get("upstream_wall") == "slip":
            values["MARKER_EULER"] = "( upstream )"
        values.pop("MARKER_SUPERSONIC_OUTLET")
        boundary_groups = {"extrapolate": [], "pressure": [], "farfield": ["farfield"]}
        boundary_groups[outlet_type(s)].append(
            f"outlet, {s['outlet_pressure']}" if outlet_type(s) == "pressure" else "outlet")
        if s["kind"] == "intake":
            boundary_groups[external_outlet_type(s)].append(
                f"external_outlet, {s['pressure']}" if external_outlet_type(s) == "pressure" else "external_outlet")
        for kind, key in (("extrapolate", "MARKER_SUPERSONIC_OUTLET"), ("pressure", "MARKER_OUTLET"),
                          ("farfield", "MARKER_FAR")):
            if boundary_groups[kind]:
                values[key] = "( "+", ".join(boundary_groups[kind])+" )"
        values.update(SOLVER="RANS", KIND_TURB_MODEL="SA", SA_OPTIONS="NONE",
                      # Keep the limiter live across every block/restart. Early
                      # freezing amplified sharp-lip residuals in RANS probes.
                      LIMITER_ITER=iterations+1, LINEAR_SOLVER_ITER=50,
                      NUM_METHOD_GRAD_RECON="WEIGHTED_LEAST_SQUARES",
                      HISTORY_OUTPUT="( ITER, RMS_RES, FLOW_COEFF_SURF, NONPHYSICAL_POINTS, LINSOL )",
                      INIT_OPTION="TD_CONDITIONS", FREESTREAM_OPTION="TEMPERATURE_FS",
                      VISCOSITY_MODEL="SUTHERLAND", MU_REF=1.716e-5, MU_T_REF=273.15, SUTHERLAND_CONSTANT=110.4,
                      CONDUCTIVITY_MODEL="CONSTANT_PRANDTL", PRANDTL_LAM=0.72, PRANDTL_TURB=0.9,
                      FREESTREAM_NU_FACTOR=s["freestream_nu_factor"], REYNOLDS_LENGTH=s["ramp_length"],
                      MARKER_HEATFLUX="( "+", ".join(f"{name}, 0.0" for name in wall.split(", "))+" )",
                      CONV_NUM_METHOD_TURB="SCALAR_UPWIND",
                      MUSCL_TURB="NO", TIME_DISCRE_TURB="EULER_IMPLICIT", CFL_REDUCTION_TURB=0.5,
                      CONV_FIELD="( RMS_DENSITY, RMS_ENERGY, RMS_NU_TILDE )",
                      SCREEN_OUTPUT="( INNER_ITER, RMS_DENSITY, RMS_ENERGY, RMS_NU_TILDE, NONPHYSICAL_POINTS )",
                      OUTPUT_FILES="( RESTART, RESTART_ASCII, PARAVIEW, SURFACE_CSV )",
                      VOLUME_OUTPUT="( COORDINATES, SOLUTION, PRIMITIVE, WALL_Y_PLUS, SKIN_FRICTION, HEAT_FLUX, RESIDUAL, LIMITER )")
    if 'flow_limiter' in numerics:
        values['SLOPE_LIMITER_FLOW']=numerics['flow_limiter']
    if 'venkat_coefficient' in numerics:
        values['VENKAT_LIMITER_COEFF']=numerics['venkat_coefficient']
    if 'max_update_flow' in numerics:
        values['MAX_UPDATE_FLOW']=numerics['max_update_flow']
    if 'cfl_reduction_turb' in numerics:
        values['CFL_REDUCTION_TURB']=numerics['cfl_reduction_turb']
    return "".join(f"{key}= {value}\n" for key, value in values.items())


def physics_signature(s):
    # Preserve the legacy Euler signature, including already completed runs.
    physics = {k: s[k] for k in DEFAULTS if k not in ("size_factor", "far_size", "internal_size", "shock_size")}
    if s.get("ramp_end", "lip_x") != "lip_x":
        physics["ramp_end"] = s["ramp_end"]
    if "duct_gap_height" in s:
        physics["duct_gap_height"] = s["duct_gap_height"]
    if len(ramps(s)) > 1:
        physics["ramps"] = ramps(s)
    if "cowl_lip_height" in s:
        physics["cowl_lip_height"] = s["cowl_lip_height"]
    if is_rans(s):
        physics.update(mode="rans", turbulence_model="SA", wall="adiabatic", viscosity="sutherland")
        if s.get("upstream_wall") == "slip":
            physics["upstream_wall"] = "slip"
        physics["freestream_nu_factor"] = s["freestream_nu_factor"]
        if outlet_type(s) == "pressure":
            physics["outlet_pressure"] = s["outlet_pressure"]
        else:
            physics["outlet_type"] = "extrapolate"
        if s["kind"] == "intake" and external_outlet_type(s) != outlet_type(s):
            physics["external_outlet_type"] = external_outlet_type(s)
    return hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()


def run(case, output, ranks=None, first=500, iterations=5000, cfl=3.0, restart=None, numerics=None, initial_guess=None):
    if ranks is None:
        from mpi_defaults import default_mpi_ranks
        ranks = default_mpi_ranks()
    case, output = case.resolve(), output.resolve()
    s = json.loads((case/"settings.json").read_text())
    require(ranks >= 1 and first >= 1 and iterations >= 500 and math.isfinite(cfl) and cfl > 0, "Invalid run controls")
    require(not (restart and initial_guess), 'Restart and initial guess are mutually exclusive')
    mesh_hash = hashlib.sha256((case/"mesh.su2").read_bytes()).hexdigest()
    if "mesh_policy" in s:
        quality = json.loads((case/"quality.json").read_text())
        require(quality.get("mesh_sha256") == mesh_hash, "Mesh policy quality report does not match mesh")
        require(quality.get("mesh_policy") == s["mesh_policy"], "Mesh policy differs from quality report")
        assessment = policy_assessment(quality["layer_transition"], s["mesh_policy"])
        require(assessment["status"] == "mesh_geometry_passed",
                "Mesh policy requires review before RANS: " + "; ".join(assessment["reasons"]))
    binary = shutil.which("SU2_CFD")
    require(binary is not None, "SU2_CFD not on PATH")
    physics_hash = physics_signature(s)
    inherited={};guess_record=None;restart_check=None
    if initial_guess:
        from intake_transfer import validate_guess
        require(is_rans(s),'Initial guess requires RANS')
        guess_record=validate_guess(case,initial_guess)
        require(guess_record['source']['binary_sha256']==hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                'Initial guess solver differs')
        inherited=guess_record.get('numerics',{})
    if restart:
        previous_record = json.loads((restart/"run.json").read_text())
        require(previous_record["mesh_sha256"] == mesh_hash, "Restart mesh differs")
        require(previous_record.get("physics_sha256") == physics_hash, "Restart physical conditions differ")
        require(previous_record["returncode"] == 0, "Restart run did not succeed")
        require(previous_record['config_sha256'] == hashlib.sha256((restart/'case.cfg').read_bytes()).hexdigest(),
                'Restart configuration provenance differs')
        if is_rans(s):
            restart_check = validate_rans_restart(case, restart, Path(binary))
        inherited=previous_record.get('numerics',{})
    numerics=numerical_controls(numerics,inherited)
    require(is_rans(s) or not numerics, 'Experimental numerical controls require RANS')
    output.mkdir(parents=True, exist_ok=False)
    if restart:
        baseline = restart.parent/"initial-history.csv"
        if not baseline.exists():
            baseline = restart.parent/"first"/"history.csv"
        require(baseline.exists(), "Missing initial convergence history")
        shutil.copy2(baseline, output/"initial-history.csv")
    stages = [("first", first, False), ("second", iterations, True)] if restart is None and initial_guess is None else [("second", iterations, True)]
    previous = restart.resolve() if restart else None
    for name, count, second in stages:
        target = output/name
        target.mkdir()
        if previous:
            shutil.copy2(previous/"restart_flow.dat", target/"restart_flow.dat")
        config_text=configuration(case,s,count,second,cfl,numerics)
        if initial_guess:
            shutil.copy2(initial_guess/'restart_flow.csv',target/'restart_flow.csv')
            config_text=config_text.replace('READ_BINARY_RESTART= YES\n','READ_BINARY_RESTART= NO\n')
        (target/"case.cfg").write_text(config_text)
        command = [binary, "case.cfg"] if ranks == 1 else ["mpirun", "--bind-to", "none", "-np", str(ranks), binary, "case.cfg"]
        start = time.monotonic()
        with (target/"solver.log").open("w") as log:
            result = subprocess.run(command, cwd=target, stdout=log, stderr=subprocess.STDOUT)
        record = dict(returncode=result.returncode, elapsed_s=time.monotonic()-start, ranks=ranks,
                      iterations=count, cfl=cfl, second_order=second,
                      mesh_sha256=mesh_hash, physics_sha256=physics_hash,
                      settings_sha256=hashlib.sha256((case/"settings.json").read_bytes()).hexdigest(),
                      binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                      restart_from=str(previous) if previous else None,
                      config_sha256=hashlib.sha256((target/"case.cfg").read_bytes()).hexdigest())
        record['numerics']=numerics
        if restart_check is not None:
            record['restart_validation'] = restart_check
        if initial_guess:
            record.update(initial_guess_from=str(initial_guess.resolve()),
                initial_guess_sha256=hashlib.sha256((initial_guess/'initial_guess.json').read_bytes()).hexdigest(),
                initial_guess_csv_sha256=guess_record['csv_sha256'])
        (target/"run.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps({"stage": str(target), **record}), flush=True)
        require(result.returncode == 0 and "Exit Success" in (target/"solver.log").read_text(),
                f"Solver failed: {target/'solver.log'}")
        if not second:
            handoff = validate_initial_stage(target, s, count)
            (target/'handoff-check.json').write_text(json.dumps(handoff, indent=2)+'\n')
        if initial_guess:
            shutil.copy2(target/'history.csv',output/'initial-history.csv')
        previous = target
    return output


def validate_rans_restart(case, stage, binary):
    """Apply full source checks to direct CLI restarts as well as controllers."""
    from intake_adapt import solution, digest
    from intake_convergence import check_completed_block
    require(stage.name == 'second', 'RANS restart requires a completed second-order stage')
    *_, record = solution(case, stage.parent)
    require(record.get('second_order') is True, 'RANS restart requires second-order source')
    require(record.get('binary_sha256') == digest(binary), 'Restart solver differs from current solver')
    check_completed_block(read_csv(stage/'history.csv'), record['iterations'])
    return dict(status='validated_source', source_run_sha256=digest(stage/'run.json'),
                history_sha256=digest(stage/'history.csv'),
                restart_ascii_sha256=digest(stage/'restart_flow.csv'),
                restart_binary_sha256=digest(stage/'restart_flow.dat'),
                usable_as_combustor_inlet=False)


def validate_initial_stage(stage, settings, iterations):
    """Do not seed second-order flow from an invalid first-order startup."""
    history = read_csv(stage/'history.csv')
    require(np.array_equal(history['Inner_Iter'], np.arange(iterations)),
            'Incomplete first-order startup history')
    counters = {k:float(np.max(v)) for k, v in history.items() if k.startswith('Nonphysical')}
    require(bool(counters) and all(v == 0 for v in counters.values()),
            'Nonphysical first-order startup; second-order stage not started')
    data = read_csv(stage/'restart_flow.csv')
    primitives(data, settings)
    if is_rans(settings):
        require('Nu_Tilde' in data and np.all(data['Nu_Tilde'] >= 0),
                'Invalid standard-SA startup state')
    return dict(status='valid_for_second_order_startup', iterations=iterations,
                nonphysical_max=counters,
                history_sha256=hashlib.sha256((stage/'history.csv').read_bytes()).hexdigest(),
                restart_sha256=hashlib.sha256((stage/'restart_flow.csv').read_bytes()).hexdigest(),
                usable_as_combustor_inlet=False)


def read_csv(path):
    with path.open() as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [v.strip().strip('"') for v in next(reader)]
        rows = []
        for row in reader:
            if len(row) != len(header):
                break  # SU2 restart metadata follows the point data.
            rows.append([float(v) for v in row])
    data = np.array(rows)
    require(len(rows) > 0 and np.isfinite(data).all(), f"Nonfinite/empty data: {path}")
    return {key: data[:, i] for i, key in enumerate(header)}


SU2_BINARY_RESTART = 535532


def read_dat(path):
    """SU2 binary restart: int32[5] header (magic, fields, points), 33-byte names, float64 point data."""
    raw = Path(path).read_bytes()
    require(len(raw) >= 20, f"Truncated binary restart: {path}")
    magic, nvar, npoint = (int(v) for v in np.frombuffer(raw, "<i4", 3))
    require(magic == SU2_BINARY_RESTART and nvar > 0 and npoint > 0, f"Not an SU2 binary restart: {path}")
    offset = 20+33*nvar
    require(len(raw) >= offset+8*nvar*npoint, f"Truncated binary restart: {path}")
    names = [raw[20+33*i:20+33*(i+1)].split(b"\0")[0].decode() for i in range(nvar)]
    data = np.frombuffer(raw, "<f8", nvar*npoint, offset).reshape(npoint, nvar)
    require(np.isfinite(data).all(), f"Nonfinite data: {path}")
    return {key: data[:, i].copy() for i, key in enumerate(names)}


def read_restart(path):
    """Point data of an SU2 restart, binary (.dat) or ASCII (.csv)."""
    path = Path(path)
    return read_dat(path) if path.suffix == ".dat" else read_csv(path)


def nonphysical_statistics(history):
    """Keep tail diagnostics, but never discard earlier invalid updates."""
    counters={k:np.asarray(v) for k,v in history.items() if 'nonphysical' in k.lower()}
    require(all(len(v)>0 and np.isfinite(v).all() for v in counters.values()),
            'Invalid nonphysical history counters')
    return dict(full_history={k:float(v.max()) for k,v in counters.items()},
                last500={k:float(v[-500:].max()) for k,v in counters.items()},
                all_zero=bool(counters) and all(bool(np.all(v==0)) for v in counters.values()))


def primitives(data, s):
    rho = data["Density"]
    u, v = data["Momentum_x"]/rho, data["Momentum_y"]/rho
    pressure = (s["gamma"]-1)*(data["Energy"]-0.5*rho*(u*u+v*v))
    require(np.all(rho > 0) and np.all(pressure > 0), "Nonpositive physical state")
    a = np.sqrt(s["gamma"]*pressure/rho)
    mach = np.hypot(u, v)/a
    total_pressure = pressure*(1+(s["gamma"]-1)/2*mach*mach)**(s["gamma"]/(s["gamma"]-1))
    total_temperature = pressure/(rho*s["gas_constant"])*(1+(s["gamma"]-1)/2*mach*mach)
    return dict(density=rho, u=u, v=v, pressure=pressure, mach=mach, sound_speed=a,
                total_pressure=total_pressure, temperature=pressure/(rho*s["gas_constant"]),
                total_temperature=total_temperature,
                total_enthalpy=s["gamma"]*s["gas_constant"]/(s["gamma"]-1)*total_temperature)


def volume_output(stage, stem="flow"):
    """The stage's volume file: binary VTU, else legacy ASCII VTK, plain or gzipped by history_archive.py."""
    for name in (stem+".vtu", stem+".vtk", stem+".vtk.gz"):
        if (Path(stage)/name).is_file():
            return Path(stage)/name
    raise ValueError(f"No {stem}.vtu or {stem}.vtk in {stage}")


def content_sha256(path):
    """SHA-256 of the file content; a .gz file hashes as the file it restores, so provenance is unchanged."""
    path = Path(path)
    h = hashlib.sha256()
    with (gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")) as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _legacy_vtk(path, wanted):
    """Points and selected point arrays of SU2 PARAVIEW_ASCII output (.vtk or .vtk.gz)."""
    out, xyz = {}, None
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        require(stream.readline().startswith("# vtk DataFile"), "Expected VTK header")
        stream.readline()
        require(stream.readline().strip() == "ASCII", "Expected ASCII VTK")
        def numbers(count):
            values = []
            while len(values) < count:
                line = stream.readline()
                require(bool(line), "Truncated VTK array")
                values.extend(float(v) for v in line.split())
            require(len(values) == count, "VTK array length mismatch")
            return np.array(values)
        count, point_data = None, False
        for line in stream:
            words = line.split()
            if words[:1] == ["POINTS"]:
                count = int(words[1])
                xyz = numbers(3*count).reshape(-1, 3)
            elif words[:1] == ["POINT_DATA"]:
                require(count is not None and int(words[1]) == count, "VTK point-data count mismatch")
                point_data = True
            elif words[:1] in (["SCALARS"], ["VECTORS"]) and words[1] in wanted:
                require(point_data, "Expected VTK point data")
                if words[0] == "SCALARS":
                    require(len(words) == 4 and words[3] == "1" and
                            stream.readline().strip() == "LOOKUP_TABLE default", f"Invalid {words[1]} header")
                    out[words[1]] = numbers(count)
                else:
                    out[words[1]] = numbers(3*count).reshape(-1, 3)
                if set(out) == set(wanted):
                    break
    return xyz, out


VTU_TYPES = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1"}


def _binary_vtu(path, wanted):
    """Points and selected point arrays of SU2 PARAVIEW output (VTU with raw appended data)."""
    raw = path.read_bytes()
    start = raw.find(b'<AppendedData encoding="raw">')
    require(start > 0, "Expected a VTU with raw appended data")
    head = raw[:start].decode()
    require('byte_order="LittleEndian"' in head, "Expected a little-endian VTU")
    size_type = "<u8" if 'header_type="UInt64"' in head else "<u4"
    data = raw[raw.index(b"_", start)+1:]
    count = int(re.search(r'NumberOfPoints="(\d+)"', head)[1])

    def array(tag):
        kind, comps, offset = tag.group(1), int(tag.group(3)), int(tag.group(4))
        width = np.dtype(size_type).itemsize
        nbytes = int(np.frombuffer(data, size_type, 1, offset)[0])
        values = np.frombuffer(data, VTU_TYPES[kind], nbytes//np.dtype(VTU_TYPES[kind]).itemsize, offset+width)
        require(len(values) == count*comps, f"VTU array length mismatch: {tag.group(2) or 'Points'}")
        return values.astype(float).reshape(-1, comps) if comps > 1 else values.astype(float)
    tags = r'<DataArray type="(\w+)" Name="(\w*)" NumberOfComponents=\s*"(\d+)" offset="(\d+)"'
    points = re.search(r"<Points>\s*"+tags, head)
    require(points is not None, "VTU has no points")
    xyz = array(points)
    out = {}
    section = head[head.index("<PointData>"):head.index("</PointData>")] if "<PointData>" in head else ""
    for tag in re.finditer(tags, section):
        if tag.group(2) in wanted:
            out[tag.group(2)] = array(tag)
    return xyz, out


def read_volume(path, points, wanted, *, rtol=1e-6, atol=1e-9, finite=True):
    """Selected point arrays of an SU2 volume file (.vtu, .vtk or .vtk.gz), checked against the mesh points."""
    path = Path(path)
    if not path.exists() and Path(str(path)+".gz").is_file():
        path = Path(str(path)+".gz")
    wanted = set(wanted)
    xyz, out = (_binary_vtu if path.suffix == ".vtu" else _legacy_vtk)(path, wanted)
    require(xyz is not None and len(xyz) == len(points), "Volume point count differs from mesh")
    require(np.allclose(xyz[:, :2], points, rtol=rtol, atol=atol), "Volume mesh mismatch: coordinates differ")
    require(set(out) == wanted, f"Missing volume arrays: {sorted(wanted-set(out))}")
    require(not finite or all(np.isfinite(v).all() for v in out.values()), "Invalid volume values: nonfinite")
    return out


def read_wall_yplus(path, points):
    """Wall y+ from the volume output (VTU or legacy VTK); restart CSV has solution fields only."""
    values = read_volume(path, points, {"Y_Plus"}, finite=False)["Y_Plus"]
    require(len(values) == len(points) and np.isfinite(values).all() and np.all(values >= 0), "Invalid wall y+")
    return values


def boundary_integrals(points, cells, markers, flow):
    """Outward normals from CCW cell edges; lumped endpoint quadrature (SU2 dual faces)."""
    oriented = {}
    for block in cell_blocks(cells).values():
        for cell in block:
            for a, b in zip(cell, np.roll(cell, -1)):
                oriented[tuple(sorted((a, b)))] = (a, b)
    result = {}
    for name, edges in markers.items():
        ordered = np.array([oriented[tuple(sorted(e))] for e in edges])
        delta = points[ordered[:, 1]]-points[ordered[:, 0]]
        normal = np.column_stack((delta[:, 1], -delta[:, 0]))
        flux = flow["density"][ordered]*(flow["u"][ordered]*normal[:, 0, None]
                                            + flow["v"][ordered]*normal[:, 1, None])/2
        weights = np.abs(flux)
        mass = float(flux.sum())
        entry = dict(massflow_outward=mass, energyflow_outward=float(np.sum(flux*flow["total_enthalpy"][ordered])))
        if weights.sum() > 1e-12:
            entry.update({key: float(np.sum(weights*flow[key][ordered])/weights.sum())
                          for key in ("mach", "pressure", "total_pressure", "temperature", "total_temperature")})
        mn = (flow["u"][ordered]*normal[:, 0, None]+flow["v"][ordered]*normal[:, 1, None]) / (
            np.linalg.norm(normal, axis=1)[:, None]*flow["sound_speed"][ordered])
        entry["min_normal_mach"] = float(mn.min())
        entry["max_normal_mach"] = float(mn.max())
        entry["reverse_massflow_fraction"] = float(-np.minimum(flux, 0).sum()/max(weights.sum(), 1e-30))
        outgoing = np.maximum(flux, 0)
        entry["subsonic_outgoing_massflow_fraction"] = float(np.sum(outgoing*(mn < 1))/max(outgoing.sum(), 1e-30))
        result[name] = entry
    return result


def analyze(case, result):
    case, result = case.resolve(), result.resolve()
    stage = result/"second"
    s = json.loads((case/"settings.json").read_text())
    points, cells, markers = read_su2(case/"mesh.su2")
    data = read_csv(stage/"restart_flow.csv")
    # Output is global point-ID order, independent of MPI partitioning.
    ids = data["PointID"].astype(int)
    require(np.array_equal(np.sort(ids), np.arange(len(points))), "Restart point IDs do not match mesh")
    order = np.argsort(ids)
    data = {k: v[order] for k, v in data.items()}
    require(np.allclose(np.column_stack((data["x"], data["y"])), points, rtol=0, atol=1e-12), "Restart mesh mismatch")
    flow = primitives(data, s)
    history = read_csv(stage/"history.csv")
    boundary = boundary_integrals(points, cells, markers, flow)
    open_names = [n for n in markers if n not in ("lower", "cowl", "upstream")]
    inflow = -sum(min(0, boundary[n]["massflow_outward"]) for n in open_names)
    imbalance = abs(sum(boundary[n]["massflow_outward"] for n in open_names))/inflow
    energy_in = -sum(min(0, boundary[n]["energyflow_outward"]) for n in open_names)
    energy_imbalance = abs(sum(boundary[n]["energyflow_outward"] for n in open_names))/energy_in
    p0in = s["pressure"]*(1+(s["gamma"]-1)/2*s["mach"]**2)**(s["gamma"]/(s["gamma"]-1))
    outlet = boundary["outlet"]
    metrics = dict(massflow=outlet["massflow_outward"], mach=outlet["mach"],
                   pressure=outlet["pressure"], total_pressure_recovery=outlet["total_pressure"]/p0in)
    residuals = ["rms[Rho]", "rms[RhoE]"]
    if is_rans(s):
        residuals.append("rms[nu]")
    baseline = result/"first"/"history.csv"
    if not baseline.exists():
        baseline = result/"initial-history.csv"
    initial_history = read_csv(baseline)
    reduction = {k: float(initial_history[k][0]-history[k][-1]) for k in residuals}
    stability = {}
    for key in ("Avg_Massflow(outlet)", "Avg_Mach(outlet)", "Avg_Press(outlet)", "Avg_TotalPress(outlet)"):
        tail = history[key][-500:]
        stability[key] = float(np.ptp(tail)/max(abs(np.mean(tail)), 1e-30))
    nonphysical = nonphysical_statistics(history)
    checks = dict(residual_reduction=all(v >= 5 for v in reduction.values()),
                  outlet_stability=len(next(iter(history.values()))) >= 500 and max(stability.values()) < 0.001,
                  mass_balance=imbalance < 0.001,
                  energy_balance=energy_imbalance < 0.001,
                  supersonic_outlets=all(boundary[n]["min_normal_mach"] > 1 for n in markers if "outlet" in n),
                  nonphysical_points=nonphysical['all_zero'])
    wall_diagnostics = None
    if is_rans(s):
        # No-slip walls necessarily have M=0; a whole-outlet M>1 test cannot
        # pass. Report subsonic mass flux explicitly for extrapolation studies.
        checks.pop("supersonic_outlets")
        checks.pop("energy_balance")  # Convective flux omits viscous/thermal diffusion.
        outlets = [n for n in markers if "outlet" in n and
                   not (n == "external_outlet" and external_outlet_type(s) == "farfield")]
        checks["outlet_forward_flow"] = all(boundary[n]["massflow_outward"] > 0 and
            boundary[n]["reverse_massflow_fraction"] < 1e-6 for n in outlets)
        wall_diagnostics = {}
        all_yplus = read_wall_yplus(volume_output(stage), points)
        for name in ("lower", "cowl"):
            if name not in markers:
                continue
            nodes = np.unique(markers[name])
            yplus = all_yplus[nodes]
            wall_diagnostics[name] = dict(y_plus=distribution(yplus),
                                          fraction_y_plus_above_one=float(np.mean(yplus > 1)),
                                          max_speed=float(np.hypot(flow["u"][nodes], flow["v"][nodes]).max()))
        checks["wall_y_plus"] = all(v["y_plus"]["maximum"] <= 1 for v in wall_diagnostics.values())
        checks["no_slip"] = all(v["max_speed"] < 1e-8 for v in wall_diagnostics.values())
    # Independent boundary integration must reproduce solver monitoring values.
    agreement = dict(massflow=abs(metrics["massflow"] / -history["Avg_Massflow(outlet)"][-1]-1),
                     mach=abs(metrics["mach"] / history["Avg_Mach(outlet)"][-1]-1),
                     pressure=abs(metrics["pressure"] / history["Avg_Press(outlet)"][-1]-1),
                     total_pressure=abs(outlet["total_pressure"] / history["Avg_TotalPress(outlet)"][-1]-1))
    agreement = {k: float(v) for k, v in agreement.items()}
    checks["integration_agreement"] = max(agreement.values()) < 1e-6
    record = json.loads((stage/"run.json").read_text())
    checks["solver_success"] = record["returncode"] == 0 and "Exit Success" in (stage/"solver.log").read_text()
    checks["mesh_provenance"] = record["mesh_sha256"] == hashlib.sha256((case/"mesh.su2").read_bytes()).hexdigest()
    if "settings_sha256" in record:
        checks["settings_provenance"] = record["settings_sha256"] == hashlib.sha256((case/"settings.json").read_bytes()).hexdigest()
    from benchmark_su2 import read_config
    actual = read_config(stage/"case.cfg")
    checks["physical_conditions"] = all(float(actual[k]) == s[v] for k, v in {
        "MACH_NUMBER": "mach", "FREESTREAM_PRESSURE": "pressure", "FREESTREAM_TEMPERATURE": "temperature",
        "GAMMA_VALUE": "gamma", "GAS_CONSTANT": "gas_constant"}.items())
    expected_config = dict(line.split("= ", 1) for line in configuration(case, s, 1, True, 1).splitlines())
    model_keys = ("SOLVER", "MARKER_EULER", "MARKER_HEATFLUX", "MARKER_OUTLET", "MARKER_SUPERSONIC_OUTLET", "MARKER_FAR",
                  "KIND_TURB_MODEL", "SA_OPTIONS", "INIT_OPTION", "FREESTREAM_OPTION", "VISCOSITY_MODEL",
                  "MU_REF", "MU_T_REF", "SUTHERLAND_CONSTANT", "CONDUCTIVITY_MODEL", "PRANDTL_LAM",
                  "PRANDTL_TURB", "FREESTREAM_NU_FACTOR")
    checks["physical_conditions"] &= all(actual.get(k) == expected_config.get(k) for k in model_keys)
    if "physics_sha256" in record:
        checks["physics_provenance"] = record["physics_sha256"] == physics_signature(s)
    t0 = s["temperature"]*(1+(s["gamma"]-1)/2*s["mach"]**2)
    excess = flow["total_temperature"] > 1.01*t0
    i_max = int(np.argmax(flow["total_temperature"]))
    local_temperature = dict(reference_total_temperature=t0, max_total_temperature=float(flow["total_temperature"][i_max]),
                             max_location=points[i_max].tolist(), nodes_above_one_percent=int(excess.sum()))
    if s["kind"] == "intake" and excess.any():
        local_temperature["max_distance_from_lip_m"] = float(np.max(np.linalg.norm(points[excess]-[s["ramp_length"], s["lip_y"]], axis=1)))
    summary = dict(settings=s, mesh=json.loads((case/"quality.json").read_text()), metrics=metrics,
                   boundary=boundary, mass_imbalance=imbalance, residual_reduction_decades=reduction,
                   outlet_last500_relative_span=stability, nonphysical_last500=nonphysical['last500'],
                   nonphysical_full_history=nonphysical['full_history'], checks=checks,
                   integration_relative_difference=agreement,
                   energy_imbalance=energy_imbalance, local_temperature_diagnostic=local_temperature,
                   requested_iterations=int(actual["ITER"]), completed_iterations=int(history["Inner_Iter"][-1])+1,
                   stop_request=json.loads((stage/"stop-request.json").read_text()) if (stage/"stop-request.json").exists() else None,
                   ranges={k: [float(v.min()), float(v.max())] for k, v in flow.items()})
    if is_rans(s):
        mu = 1.716e-5*(s["temperature"]/273.15)**1.5*(273.15+110.4)/(s["temperature"]+110.4)
        rho = s["pressure"]/(s["gas_constant"]*s["temperature"])
        speed = s["mach"]*math.sqrt(s["gamma"]*s["gas_constant"]*s["temperature"])
        summary["rans"] = dict(turbulence_model="SA", wall="adiabatic_no_slip", viscosity=mu,
                               upstream_wall=s.get("upstream_wall", "no_slip"),
                               outlet_type=outlet_type(s),
                               external_outlet_type=external_outlet_type(s) if s["kind"] == "intake" else None,
                               outlet_note="Extrapolation across a subsonic wall layer requires outlet-location sensitivity checks."
                               if outlet_type(s) == "extrapolate" else "Specified back pressure is used at locally subsonic outlet nodes.",
                               reynolds_number=rho*speed*s["ramp_length"]/mu, walls=wall_diagnostics,
                               energy_balance_note="Convective total-enthalpy flux only; excludes diffusion through open boundaries.",
                               total_temperature_note="Diagnostic only: viscous transport changes local total temperature.")
        summary["rans"]["wall_spacing_feedback"] = wall_spacing_feedback(
            s["boundary_layer"]["first_height"],
            {name: values["y_plus"]["maximum"] for name, values in wall_diagnostics.items()}, checks)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    triangles = plot_triangles(cells)
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    gradient = np.array(mtri.LinearTriInterpolator(tri, flow["density"]).gradient(
        points[triangles].mean(axis=1)[:, 0], points[triangles].mean(axis=1)[:, 1]))
    schlieren = np.log10(1+np.hypot(*gradient))
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    for ax, key, label in zip(axes, ("mach", "pressure", "schlieren"),
                              ("Mach number", "Static pressure [Pa]", "log10(1 + |grad density|)")):
        if key == "schlieren":
            im = ax.tripcolor(tri, facecolors=schlieren, cmap="Greys", shading="flat")
        else:
            im = ax.tripcolor(tri, flow[key], cmap="turbo", shading="gouraud")
        for name in ("lower", "cowl"):
            for edge in markers.get(name, []):
                ax.plot(points[edge, 0], points[edge, 1], color="#182737", lw=1)
        for estimate in incident_shocks(s):
            x0, y0 = estimate["origin"]
            x1 = s["ramp_length"]
            y1 = y0+(x1-x0)*math.tan(math.radians(estimate["ray_angle_deg"]))
            ax.plot([x0, x1], [y0, y1], "--", color="#db3d49", lw=1)
        ax.set(xlabel="x [m]", ylabel="y [m]", title=label, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    fig.savefig(result/"flow.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in ("lower", "cowl"):
        if name not in markers:
            continue
        nodes = np.unique(markers[name])
        if name == "cowl":
            nodes = nodes[np.isclose(points[nodes, 1], s["lip_y"])]
        nodes = nodes[np.argsort(points[nodes, 0])]
        axes[0].plot(points[nodes, 0], flow["pressure"][nodes]/s["pressure"], label=name)
    axes[0].set(xlabel="x [m]", ylabel="p / p_infinity", title="Wall static pressure")
    axes[0].legend()
    for k in residuals:
        axes[1].plot(history["Inner_Iter"], history[k], label=k)
    limiter_iteration = int(actual["LIMITER_ITER"])
    if history["Inner_Iter"][-1] > limiter_iteration:
        axes[1].axvline(limiter_iteration, ls="--", lw=1, color="#888888", label="Limiter fixed")
    axes[1].set(xlabel="Second-order iteration", ylabel="log10 residual", title="Convergence")
    axes[1].legend()
    fig.savefig(result/"wall-convergence.png", dpi=180)
    plt.close(fig)
    if s["kind"] == "wedge" and not is_rans(s):
        theory = s["theory"]
        beta, theta = math.radians(theory["beta_deg"]), math.radians(s["ramp_angle_deg"])
        interp_p = mtri.LinearTriInterpolator(tri, flow["pressure"])
        interp_m = mtri.LinearTriInterpolator(tri, flow["mach"])
        xs = np.linspace(0.4, 0.8, 30)*s["ramp_length"]
        ys = xs*(math.tan(theta)+math.tan(beta))/2
        p = float(np.mean(interp_p(xs, ys)))
        mach = float(np.mean(interp_m(xs, ys)))
        shock_y = []
        for x in xs:
            y = np.linspace(x*math.tan(theta)+0.01, x*math.tan(beta)+0.08, 600)
            vals = interp_p(np.full_like(y, x), y)
            target = s["pressure"]*(1+theory["pressure_ratio"])/2
            crossings = np.flatnonzero((vals[:-1] >= target) & (vals[1:] < target))
            require(len(crossings) > 0, "Unable to locate wedge shock")
            i = crossings[-1]
            shock_y.append(float(y[i]+(target-vals[i])*(y[i+1]-y[i])/(vals[i+1]-vals[i])))
        fitted = math.degrees(math.atan(np.polyfit(xs, shock_y, 1)[0]))
        errors = dict(shock_angle_deg=abs(fitted-theory["beta_deg"]),
                      pressure_relative=abs(p/(s["pressure"]*theory["pressure_ratio"])-1),
                      mach_relative=abs(mach/theory["mach_downstream"]-1))
        summary["wedge_validation"] = dict(measured=dict(beta_deg=fitted, pressure=p, mach=mach), errors=errors,
                                             passed=errors["shock_angle_deg"] <= 0.5 and max(errors["pressure_relative"], errors["mach_relative"]) <= 0.02)
    (result/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    # Checkpoint consumers finalize separately; this summary does not prove completion.
    print(json.dumps({k: summary[k] for k in ("metrics", "mass_imbalance", "checks")}), flush=True)
    return summary


def compare(manifest_path):
    """Build the local study report only from explicitly selected analyzed runs."""
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.resolve().parent
    paths = {k: (base/v).resolve() for k, v in manifest["runs"].items()}
    require({"wedge", "coarse", "medium", "fine", "domain"} <= set(paths), "Missing study runs")
    summaries = {k: json.loads((p/"summary.json").read_text()) for k, p in paths.items()}
    require(not any(is_rans(summary["settings"]) for summary in summaries.values()),
            "compare is an Euler verification report; RANS has no inviscid wedge acceptance criterion")
    reference = summaries["medium"]["settings"]
    physics_keys = ("mach", "pressure", "temperature", "gamma", "gas_constant", "ramp_angle_deg", "ramp_length")
    geometry_keys = ("outlet_x", "upstream_length", "top_height_factor", "cowl_thickness", "cowl_nose_length")
    for name, summary in summaries.items():
        s = summary["settings"]
        require(all(s[k] == reference[k] for k in physics_keys), "Study physical conditions differ")
        require(s["kind"] == ("wedge" if name == "wedge" else "intake"), "Incorrect study geometry type")
        if name != "wedge":
            require(s.get("ramp_end", "lip_x") == reference.get("ramp_end", "lip_x"),
                    "Grid comparison ramp end differs")
            require(ramps(s) == ramps(reference), "Grid comparison ramp geometry differs")
            require(s.get("cowl_lip_height") == reference.get("cowl_lip_height"),
                    "Grid comparison cowl geometry differs")
            keys = [k for k in geometry_keys if name != "domain" or k not in ("upstream_length", "top_height_factor")]
            require(all(s[k] == reference[k] for k in keys), "Grid comparison geometry differs")
            require(all(s[k] == reference[k] for k in ("far_size", "internal_size", "shock_size")),
                    "Grid size distributions differ")
    require(summaries["domain"]["settings"]["size_factor"] == reference["size_factor"], "Domain grid sizes differ")
    require(all(summaries["domain"]["settings"][k] > reference[k] for k in ("upstream_length", "top_height_factor")),
            "Domain study must expand upstream and top boundaries")
    pairs = {"grid": ("medium", "fine"), "domain": ("medium", "domain")}
    if "extra_fine" in summaries:
        pairs["grid_extra"] = ("fine", "extra_fine")
    if "cfl_check" in summaries:
        require(summaries["cfl_check"]["settings"]["size_factor"] == reference["size_factor"], "CFL check grid differs")
        pairs["cfl"] = ("medium", "cfl_check")
    differences = {}
    for label, (a, b) in pairs.items():
        if label.startswith("grid"):
            require(summaries[a]["settings"]["size_factor"] > summaries[b]["settings"]["size_factor"],
                    "Grid comparison must use successively finer grids")
        differences[label] = {k: abs(summaries[a]["metrics"][k]/summaries[b]["metrics"][k]-1)
                              for k in ("massflow", "mach", "pressure", "total_pressure_recovery")}
    grid_key = "grid_extra" if "grid_extra" in differences else "grid"
    checks = dict(all_runs=all(all(s["checks"].values()) for s in summaries.values()),
                  wedge_theory=summaries["wedge"]["wedge_validation"]["passed"],
                  grid=max(differences[grid_key].values()) <= 0.01,
                  domain=max(differences["domain"].values()) <= 0.01)
    if "cfl" in differences:
        checks["cfl_sensitivity"] = max(differences["cfl"].values()) < 0.001
    out = (base/manifest["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"comparison.json").write_text(json.dumps(dict(differences=differences, checks=checks), indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid_names = [n for n in ("coarse", "medium", "fine", "extra_fine") if n in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, key, label in zip(axes.flat, ("massflow", "mach", "pressure", "total_pressure_recovery"),
                              ("Outlet mass flow [kg/(s m)]", "Outlet mean Mach", "Outlet mean pressure [Pa]", "Total pressure recovery")):
        ax.plot([summaries[n]["mesh"]["nodes"] for n in grid_names],
                [summaries[n]["metrics"][key] for n in grid_names], "o-", color="#176e89")
        ax.set(xlabel="Mesh nodes", ylabel=label, xscale="log")
        ticks = sorted({summaries[n]["mesh"]["nodes"] for n in grid_names})
        ax.set_xticks(ticks, [f"{n/1000:.1f}k" for n in ticks])
        ax.minorticks_off()
        ax.grid(alpha=0.2)
    fig.savefig(out/"grid-comparison.png", dpi=180)
    plt.close(fig)
    import os
    link = lambda p: os.path.relpath(p, out)
    ramp_description = " → ".join(f"{r['angle_deg']:g}°" for r in ramps(reference))
    lines = [f"# Mach {reference['mach']:g}・2DインテークのEuler解析", "",
             "判定: **"+("設定した収束・理論解・出口指標比較の基準を満たした" if all(checks.values()) else "未達の検証項目あり")+"**。", "",
             f"Mach {reference['mach']:g}、静温{reference['temperature']:g} K、静圧{reference['pressure']:g} Pa、迎角0°、γ={reference['gamma']:g}の熱量的完全気体。{ramp_description}ランプとカウルを含む2D定常非粘性流れ。",
             "形状は第1衝撃波の理論延長線上にカウル先端を置く基礎ケース（2段での同時入射は保証しない）で、実機形状・性能の検証ではない。", "",
             "## 出口指標", "",
             "平均値は質量流束加重。全圧回復率は出口全圧平均／一様流全圧。流量は単位奥行き当たり。", "",
             "| ケース | 節点数 | 流量 kg/(s·m) | Mach | 静圧 Pa | 全圧回復率 | 質量不釣合い |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in summaries.items():
        m = s["metrics"]
        lines.append(f"| [{name}]({link(paths[name]/'summary.json')}) | {s['mesh']['nodes']:,} | {m['massflow']:.6f} | {m['mach']:.6f} | {m['pressure']:.3f} | {m['total_pressure_recovery']:.6f} | {100*s['mass_imbalance']:.6g}% |")
    lines += ["", "くさびケースの出口は衝撃波の上下流をともに含むため、インテーク出口とは比較しない。", "",
              "## 理論解・収束・格子依存性", ""]
    e = summaries["wedge"]["wedge_validation"]["errors"]
    lines += [f"- くさび理論解との差：衝撃波角 {e['shock_angle_deg']:.5f}°、圧力 {100*e['pressure_relative']:.5f}%、Mach {100*e['mach_relative']:.5f}%。"]
    for label, d in differences.items():
        lines.append(f"- {label}（{pairs[label][0]} → {pairs[label][1]}）の相対差："+"、".join(f"{k} {100*v:.4f}%" for k, v in d.items())+"。")
    lines += ["", "![格子依存性](grid-comparison.png)", ""]
    for name, s in summaries.items():
        failed = [k for k, v in s["checks"].items() if not v]
        lines.append(f"- {name}：密度／エネルギー残差低下 "+" / ".join(f"{v:.2f}" for v in s["residual_reduction_decades"].values())+
                     f" 桁、出口指標の最後500反復の最大変動 {100*max(s['outlet_last500_relative_span'].values()):.5g}%。"+
                     ("検証項目はすべて合格。" if not failed else "未達："+", ".join(failed)+"。"))
        if s.get("stop_request"):
            lines.append(f"  上限{s['requested_iterations']}反復に対して{s['completed_iterations']}反復で停止要求により終了した。停止理由・判定値はsecond/stop-request.jsonに保存。収束は上記の検査項目で判定し、停止要求や正常終了メッセージ自体を収束の根拠にはしない。")
    lines += ["", "### カウル先端の局所的な数値誤差", "",
              "全温は断熱・仕事なしの定常Euler流れで一定になるため、その過大値を追加点検した。以下の局所過大値を物理的な加熱として解釈しない。出口積分指標の合格と、先端の点ごとの精度は別である。", ""]
    for name, s in summaries.items():
        diagnostic = s.get("local_temperature_diagnostic", {})
        if s["settings"]["kind"] == "intake" and diagnostic:
            lines.append(f"- {name}：全温最大 {diagnostic['max_total_temperature']:.2f} K、基準全温を1%以上超える点 {diagnostic['nodes_above_one_percent']} 点、先端からの最大距離 {diagnostic.get('max_distance_from_lip_m', 0)*1000:.3f} mm。全開境界のエネルギー不釣合い {100*s['energy_imbalance']:.5g}%。")
    lines += ["", "鋭いカウル先端の温度・圧力ピークを定量評価するには、先端形状・境界離散化を含む追加検証が必要。今回の格子比較は出口積分指標の確認に限定する。"]
    final = "extra_fine" if "extra_fine" in summaries else "fine"
    lines += ["", "## 流れ場", "", f"![Mach・圧力・密度勾配]({link(paths[final]/'flow.png')})", "",
              f"![壁面圧力と収束]({link(paths[final]/'wall-convergence.png')})", "",
              "破線は各段の逐次斜め衝撃波の推定線で、衝撃波同士の干渉後の位置は表さない。カウル内面とランプ終端から生じる反射・膨張を含めて計算している。", "",
              "## 数値条件と解釈", "",
              "HLLC、一次精度で初期化後にMUSCL二次精度、Venkatakrishnan制限関数を二次精度の1500反復後に固定。陰解法、マルチグリッドなし。CFL・反復数は各runのcase.cfg・run.json・solver.logに保存。", "",
              "残差は全変数の有限性を確認し、密度・エネルギーを初期状態から5桁以上低下させる。最後500反復の出口指標変動0.1%未満、全開境界の質量不釣合い0.1%未満、流出面法線Mach>1、非物理点0を確認する。独立に積分した出口指標とSU2モニターの差も確認する。", "",
              "格子・領域比較の1%基準は上記の出口積分指標に対するもの。局所衝撃波厚さや壁面圧力ピークの格子独立性は主張しない。", "",
              f"境界層、粘性剥離、壁面熱流束、燃焼、高温気体効果、始動限界は対象外。完全気体モデルの全温は{reference['temperature']*(1+(reference['gamma']-1)/2*reference['mach']**2):g} Kで、実在空気に対する比熱一定近似の誤差を検証した結果ではない。", "",
              "実行時間はrun.jsonに観測値として保存し、速度比較のベンチマーク結論には使わない。", "",
              "## 再現・参照", "",
              "生成器・実行器・集計器の使用法はリポジトリのscripts/README.mdを参照。各ケースのsettings.json、mesh.su2、quality.jsonと各結果のrestart_flow.csv、ParaView用flow.vtuを保存。", "",
              "- [SU2 8.5.0のくさび例](https://github.com/su2code/SU2/blob/v8.5.0/TestCases/euler/wedge/inv_wedge_HLLC.cfg)",
              "- [NASAのくさび流れ検証例](https://www.grc.nasa.gov/www/wind/valid/wedgeM2/wedgeM2.html)", ""]
    (out/"report.md").write_text("\n".join(lines))
    print(json.dumps(dict(report=str(out/"report.md"), checks=checks, differences=differences)), flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("settings", type=Path)
    comparison = sub.add_parser("compare")
    comparison.add_argument("manifest", type=Path)
    for name in ("run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("case", type=Path)
        p.add_argument("result", type=Path)
        if name == "run":
            p.add_argument("--ranks", type=int, default=None,
                           help="MPI processes; omitted: MacBook 4, Mac mini 12, capped by available CPUs")
            p.add_argument("--first", type=int, default=500)
            p.add_argument("--iterations", type=int, default=5000)
            p.add_argument("--cfl", type=float, default=3.0)
            p.add_argument("--restart", type=Path)
            p.add_argument("--initial-guess",type=Path,help='Validated same-physics mesh-transfer folder; begins a new residual history')
            p.add_argument("--flow-limiter", choices=('VENKATAKRISHNAN','BARTH_JESPERSEN'))
            p.add_argument("--venkat-coeff", type=float)
            p.add_argument("--max-update-flow", type=float)
            p.add_argument("--cfl-reduction-turb", type=float)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.settings)
    elif args.command == "compare":
        compare(args.manifest)
    elif args.command == "run":
        numerics={}
        if args.flow_limiter is not None:numerics['flow_limiter']=args.flow_limiter
        if args.venkat_coeff is not None:numerics['venkat_coefficient']=args.venkat_coeff
        if args.max_update_flow is not None:numerics['max_update_flow']=args.max_update_flow
        if args.cfl_reduction_turb is not None:numerics['cfl_reduction_turb']=args.cfl_reduction_turb
        run(args.case, args.result, args.ranks, args.first, args.iterations, args.cfl, args.restart, numerics,args.initial_guess)
    else:
        analyze(args.case, args.result)


if __name__ == "__main__":
    main()
