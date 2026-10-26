#!/usr/bin/env python3
"""Extract a diagnostic interior section from an intake solution (SI, per metre span).

Interpolate conservative variables linearly on triangles (split quads), then
integrate with Gaussian quadrature on every mesh/section intersection interval.
This is postprocessing, not the solver's dual-face flux, and does not certify an
inlet boundary profile. Domain, mesh and convergence checks remain necessary.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from intake_euler import is_rans, physics_signature, plot_triangles, primitives, read_csv, read_su2, require


class SectionGeometry:
    """Cache the immutable mesh geometry for several sections or solutions."""

    def __init__(self, points, cells):
        import matplotlib.tri as mtri
        self.points, self.cells = points, cells
        self.triangles = plot_triangles(cells)
        self.tri = mtri.Triangulation(points[:, 0], points[:, 1], self.triangles)
        self.edges = np.unique(np.sort(np.concatenate([
            self.triangles[:, [0, 1]], self.triangles[:, [1, 2]],
            self.triangles[:, [2, 0]]]), axis=1), axis=0)
        self.finder = self.tri.get_trifinder()


def section(points, cells, data, settings, x, lower, upper, order=5, geometry=None):

    require(np.isfinite([x, lower, upper]).all() and upper > lower, "Invalid section limits")
    geometry = geometry or SectionGeometry(points, cells)
    require(geometry.points is points and geometry.cells is cells, "Cached section geometry differs")
    triangles, edges = geometry.triangles, geometry.edges
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    dx = b[:, 0]-a[:, 0]
    crossing = (np.minimum(a[:, 0], b[:, 0]) <= x) & (np.maximum(a[:, 0], b[:, 0]) >= x)
    nonvertical = crossing & (np.abs(dx) > 1e-14)
    y = a[nonvertical, 1]+(x-a[nonvertical, 0])/dx[nonvertical]*(b[nonvertical, 1]-a[nonvertical, 1])
    vertical = crossing & ~nonvertical
    y = np.concatenate(([lower, upper], y, a[vertical, 1], b[vertical, 1]))
    y = np.unique(np.clip(y[(y >= lower-1e-12) & (y <= upper+1e-12)], lower, upper))
    y = y[np.r_[True, np.diff(y) > 1e-12]]
    y[-1] = upper
    mid, half = (y[:-1]+y[1:])/2, np.diff(y)/2
    indices = geometry.finder(np.full_like(mid, x), mid)
    require(np.all(indices >= 0), "Section crosses a solid or leaves the mesh")
    vertices = triangles[indices]
    matrix = np.concatenate((np.ones((*vertices.shape, 1)), points[vertices]), axis=2)
    keys = ["Density", "Momentum_x", "Momentum_y", "Energy"]
    require(not is_rans(settings) or "Nu_Tilde" in data, "Missing SA restart variable Nu_Tilde")
    if "Nu_Tilde" in data:
        keys.append("Nu_Tilde")
    coefficients = {k: np.linalg.solve(matrix, data[k][vertices][..., None])[..., 0] for k in keys}

    def values(at):
        return {k: (c[:, 0, None]+x*c[:, 1, None]+at*c[:, 2, None]).ravel()
                for k, c in coefficients.items()}

    z, w = np.polynomial.legendre.leggauss(order)
    qy = mid[:, None]+half[:, None]*z
    weights = (half[:, None]*w).ravel()
    state = values(qy)
    flow = primitives(state, settings)
    flux = state["Momentum_x"]
    forward, backward = np.maximum(flux, 0)*weights, np.maximum(-flux, 0)*weights
    mass = float(np.sum(flux*weights))
    require(forward.sum() > 0, "No forward mass flow through section")
    averages = {key: float(np.sum(forward*v)/forward.sum()) for key, v in flow.items()}
    mn = flow["u"]/flow["sound_speed"]
    integral = dict(massflow=mass, forward_massflow=float(forward.sum()), reverse_massflow=float(backward.sum()),
                    reverse_massflow_fraction=float(backward.sum()/(forward.sum()+backward.sum())),
                    convective_energyflow=float(np.sum(flux*flow["total_enthalpy"]*weights)),
                    axial_momentum_flux=float(np.sum((flux*flow["u"]+flow["pressure"])*weights)),
                    subsonic_forward_massflow_fraction=float(np.sum(forward*(mn < 1))/forward.sum()),
                    min_normal_mach=float(mn.min()), max_normal_mach=float(mn.max()),
                    forward_massflow_averages=averages)
    end_values = values(np.column_stack((y[:-1], y[1:])))
    # Store duplicate shared endpoints only once; adjacent interpolants agree.
    choose = np.r_[np.arange(0, 2*len(mid), 2), 2*len(mid)-1]
    endpoint_data = {k: v[choose] for k, v in end_values.items()}
    endpoint_flow = primitives(endpoint_data, settings)
    profile = dict(y=y.tolist(), **{k: v.tolist() for k, v in endpoint_data.items()},
                   **{k: v.tolist() for k, v in endpoint_flow.items()})
    return integral, profile


def extract(case, result, output, x):
    s = json.loads((case/"settings.json").read_text())
    require(s["kind"] == "intake" and s["ramp_length"] < x < s["outlet_x"],
            "Choose an interior section downstream of the ramp and upstream of the outlet")
    stage = result/"second"
    points, cells, _ = read_su2(case/"mesh.su2")
    data = read_csv(stage/"restart_flow.csv")
    ids = data["PointID"].astype(int)
    require(np.array_equal(np.sort(ids), np.arange(len(points))), "Point IDs differ from mesh")
    data = {k: v[np.argsort(ids)] for k, v in data.items()}
    require(np.allclose(np.column_stack((data["x"], data["y"])), points, rtol=0, atol=1e-12), "Mesh coordinates differ")
    record = json.loads((stage/"run.json").read_text())
    require(record["mesh_sha256"] == hashlib.sha256((case/"mesh.su2").read_bytes()).hexdigest(), "Mesh provenance differs")
    require(record["physics_sha256"] == physics_signature(s), "Physical conditions differ")
    require(record["settings_sha256"] == hashlib.sha256((case/"settings.json").read_bytes()).hexdigest(), "Settings provenance differs")
    integral, profile = section(points, cells, data, s, x, s["ramp_y"], s["lip_y"])
    coarse, _ = section(points, cells, data, s, x, s["ramp_y"], s["lip_y"], order=3)
    quadrature = {k: abs(integral[k]-coarse[k])/max(abs(integral[k]), 1e-30)
                  for k in ("massflow", "convective_energyflow", "axial_momentum_flux")}
    summary_path = result/"summary.json"
    checks = json.loads(summary_path.read_text())["checks"] if summary_path.exists() else {}
    summary = dict(x=x, lower=s["ramp_y"], upper=s["lip_y"], diagnostics_only=True,
                   section_normal=[1, 0],
                   gas_model=dict(name="calorically_perfect_gas", gamma=s["gamma"], gas_constant=s["gas_constant"]),
                   turbulence_model="SA" if is_rans(s) else None,
                   usable_as_combustor_inlet=False, solver_checks_passed=bool(checks) and all(checks.values()),
                   boundary_sensitivity_verified=False, mesh_sensitivity_verified=False,
                   integral=integral, quadrature_relative_difference=quadrature,
                   units="SI; mass and fluxes per metre span; pressure is absolute static pressure",
                   method="Piecewise linear conservative reconstruction on split-quadrilateral/triangle mesh; 5-point Gauss per interval",
                   limitations="Postprocessed flux, not solver dual-face flux. Energy flux excludes viscous work and heat diffusion. SA variable is not k/omega.",
                   source=dict(case=str(case), result=str(result), mesh_sha256=record["mesh_sha256"],
                               restart_sha256=hashlib.sha256((stage/"restart_flow.csv").read_bytes()).hexdigest()),
                   profile=profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--x", type=float, required=True)
    args = parser.parse_args()
    s = extract(args.case, args.result, args.output, args.x)
    print(json.dumps({k: s[k] for k in ("x", "diagnostics_only", "solver_checks_passed", "integral", "quadrature_relative_difference")}, indent=2))


if __name__ == "__main__":
    main()
