#!/usr/bin/env python3
"""Two-block structured quadrilateral mesh for a sharp multi-ramp Euler intake.

Generalizes the 2026-10-11 local study generator ("fine-quad"). The lower block
spans the ramp/duct floor to the lip height; the upper block spans the lip height
(and, downstream of the lip, the cowl outer surface) to the top boundary. Both
blocks share every node on the artificial pre-lip interface. Wall-normal node
lines are blended near ramp corners. Geometry is exact and the cowl tip radius is
zero. Streamwise columns come from the bottom-boundary nodes of an existing mesh
(lower wall plus the y=0 farfield front), so the same columns can be reused for
several duct heights, or from a spacing rule (rule_columns) that mimics the
2026-10-11 fine-quad columns. length_scale multiplies every absolute length
(spacings, bands, blends) so a geometrically scaled intake keeps the same
resolution relative to its size; length_scale=1 reproduces the original mesh.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from generate_airfoil_mesh import read_su2
from intake_euler import polygon, ramp_vertices, require, settings as read_settings
from refine_airfoil_wake import write_su2

LIP_DX = 0.0005           # cowl-tip streamwise target within +/-10 mm of the lip [m]
LIP_BAND = 0.010
FIRST_SPACING = 0.0004    # lower-block wall-normal first-spacing cap [m]
LOWER_LAYERS, UPPER_LAYERS = 80, 72
NORMAL_BAND = 0.020       # wall-normal alignment band [m]
CORNER_BLEND = 0.010      # half width of the slope blend at ramp corners [m]
# Spacing rule fitted to the 2026-10-11 fine-quad bottom nodes (capture 0.35 m).
RULE_DX, RULE_CORNER_DX, RULE_CORNER_BAND = 0.003, 0.002, 0.05
RULE_INLET_DX, RULE_GROWTH = 0.025, 1.06


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def bottom_columns(points, markers):
    """x of lower-wall nodes plus y=0 farfield nodes upstream of the ramp."""
    ids = np.unique(markers['lower'])
    front = np.unique(markers['farfield']) if 'farfield' in markers else np.empty(0, int)
    front = front[(np.abs(points[front, 1]) <= 1e-12) & (points[front, 0] <= 0)]
    return np.unique(points[np.r_[ids, front], 0])


def rule_columns(s, length_scale=1.):
    """Streamwise columns from a spacing rule instead of an existing mesh.

    About RULE_DX on the ramps and duct, RULE_CORNER_DX near ramp corners, the
    lip and the cowl nose, and spacing growing by RULE_GROWTH per cell upstream
    of the first ramp up to RULE_INLET_DX. Every geometric knot is a column;
    between knots the columns equidistribute the spacing function, so no sliver
    cells appear. columns_for() adds the lip band refinement.
    """
    require(length_scale > 0, 'length_scale must be positive')
    dx, cdx, band, inlet = (v*length_scale for v in (RULE_DX, RULE_CORNER_DX, RULE_CORNER_BAND, RULE_INLET_DX))
    lip = s['ramp_length']
    knots = np.unique(np.r_[-s['upstream_length'], floor_vertices(s)[1:, 0], lip, lip+s['cowl_nose_length']])
    corners = knots[(knots >= 0) & (knots < s['outlet_x'])]

    def spacing(x):
        d = np.min(np.abs(corners[None, :]-x[:, None]), axis=1)
        inside = dx-(dx-cdx)*np.exp(-(d/band)**2)
        return np.where(x < 0, np.minimum(cdx+(RULE_GROWTH-1)*(-x), inlet), inside)
    out = [knots[:1]]
    for a, b in zip(knots[:-1], knots[1:]):
        x = np.linspace(a, b, 2001)
        f = np.r_[0, np.cumsum(np.diff(x)*(1/spacing(x[:-1])+1/spacing(x[1:]))/2)]
        n = max(1, int(round(f[-1])))
        inner = np.interp(np.linspace(0, f[-1], n+1)[1:-1], f, x)
        out += [inner, [b]]
    return np.concatenate(out)


def smooth_slopes(x, vertices, width=CORNER_BLEND):
    slopes = np.diff(vertices[:, 1])/np.diff(vertices[:, 0])
    v = np.full(len(x), slopes[0])
    for i, cx in enumerate(vertices[1:-1, 0]):
        t = np.clip((x-cx+width)/(2*width), 0, 1)
        t = t*t*(3-2*t)
        v += (slopes[i+1]-slopes[i])*t
    return v


def floor_vertices(s):
    return np.array([[-s['upstream_length'], 0], [0, 0], *ramp_vertices(s), [s['outlet_x'], s['ramp_y']]])


def columns_for(s, columns, factor=1, length_scale=1.):
    vs = floor_vertices(s)
    lip, nose = s['ramp_length'], s['ramp_length']+s['cowl_nose_length']
    columns = np.asarray(columns, float)
    require(columns.min() <= -s['upstream_length']+1e-12 and columns.max() >= s['outlet_x']-1e-12,
            'Columns must span inlet to outlet')
    xx = np.unique(np.r_[columns, vs[:, 0], lip, nose])
    # Snap ulp-level copies to the exact geometric knots before uniquing.
    for knot in np.r_[vs[:, 0], lip, nose]:
        xx[np.abs(xx-knot) < 1e-12] = knot
    xx = np.unique(xx)
    xx = xx[(xx >= -s['upstream_length']) & (xx <= s['outlet_x'])]
    refined = []
    target, band = LIP_DX*factor*length_scale, LIP_BAND*length_scale
    for a, b in zip(xx[:-1], xx[1:]):
        near = a < lip+band and b > lip-band
        count = int(np.ceil((b-a)/target-1e-9)) if near else 1
        refined.extend(np.linspace(a, b, max(count, 1)+1)[:-1])
    return np.r_[refined, xx[-1]]


def refine2(a):
    """Index-space bisection: edge midpoints and cell centres (exact 4-split of each quad)."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        out = np.empty(2*len(a)-1); out[::2] = a; out[1::2] = (a[:-1]+a[1:])/2
        return out
    out = np.empty((2*a.shape[0]-1, 2*a.shape[1]-1))
    out[::2, ::2] = a
    out[1::2, ::2] = (a[:-1]+a[1:])/2
    out[::2, 1::2] = (a[:, :-1]+a[:, 1:])/2
    out[1::2, 1::2] = (a[:-1, :-1]+a[1:, :-1]+a[:-1, 1:]+a[1:, 1:])/4
    return out


def mesh(s, columns, factor=1, split=False, return_grids=False, first_spacing=None,
         layers=(LOWER_LAYERS, UPPER_LAYERS), upper_first_spacing=None, length_scale=1.):
    """Return points, quads, markers and metadata. factor=2 halves the resolution.

    split=True divides every quad of the factor mesh into four (edge midpoints and
    cell centres), so the coarser mesh's nodes are a subset; return_grids adds the
    lower/upper node-index arrays used to transfer solutions between the two.
    For viscous walls, first_spacing sets the lower-block first layer at the ramp and
    cowl inner wall (two-sided tanh) and upper_first_spacing the first layer at the
    cowl outer surface and pre-lip interface (exponential, solved per column).
    first_spacing defaults to FIRST_SPACING*length_scale.
    """
    require(s['kind'] == 'intake' and s.get('mode', 'euler') == 'euler', 'Euler intake settings required')
    require(factor in (1, 2), 'factor must be 1 or 2')
    require(length_scale > 0, 'length_scale must be positive')
    first_spacing = FIRST_SPACING*length_scale if first_spacing is None else first_spacing
    normal_band, blend, edge = NORMAL_BAND*length_scale, CORNER_BLEND*length_scale, .05*length_scale
    vs = floor_vertices(s)
    lip, nose = s['ramp_length'], s['ramp_length']+s['cowl_nose_length']
    xx = columns_for(s, columns, factor, length_scale)
    nx, nl, nu = len(xx), layers[0]//factor, layers[1]//factor
    floor = np.interp(xx, vs[:, 0], vs[:, 1])
    gap = s['lip_y']-floor
    require(np.all(gap > 0), 'Lower block must have positive height')
    t = np.linspace(0, 1, nl+1)
    # Symmetric tanh grading with a physical first-layer cap; uniform where the
    # gap is already fine enough (narrow ducts).
    lo, hi = np.zeros(nx), np.full(nx, 6. if first_spacing >= 1e-4 else 30.)
    for _ in range(60):
        b = (lo+hi)/2
        first = .5*(1+np.tanh(b*(2/nl-1))/np.tanh(b))
        small = gap*first < first_spacing*factor
        hi, lo = np.where(small, b, hi), np.where(small, lo, b)
    beta = (lo+hi)/2
    uniform = gap/nl <= first_spacing*factor
    beta[uniform] = 0
    eta = np.tile(t[:, None], (1, nx))
    graded = ~uniform
    eta[:, graded] = .5*(1+np.tanh(beta[graded][None, :]*(2*t[:, None]-1))/np.tanh(beta[graded])[None, :])
    eta[0], eta[-1] = 0, 1
    height = eta*gap
    yy = floor+height
    slopes = smooth_slopes(xx, vs, blend)
    shift = -slopes*height*np.exp(-(height/normal_band)**2)*(1-eta)**2
    taper = np.minimum(np.clip((xx+s['upstream_length'])/edge, 0, 1), np.clip((s['outlet_x']-xx)/edge, 0, 1))
    lx = xx+shift*taper
    lx[0], lx[-1] = xx, xx
    floorup = np.where(xx <= lip, s['lip_y'], np.where(
        xx <= nose, s['lip_y']+(xx-lip)*s['cowl_thickness']/s['cowl_nose_length'], s['lip_y']+s['cowl_thickness']))
    tu = np.linspace(0, 1, nu+1)
    if upper_first_spacing is None:
        E = (np.expm1(4.8*tu)/np.expm1(4.8))[:, None]
    else:  # exponent per column so that the first layer equals upper_first_spacing
        H = s['top_y']-floorup; alo, ahi = np.full(nx, 1e-6), np.full(nx, 80.)
        for _ in range(100):
            a = (alo+ahi)/2; big = np.expm1(a/nu)/np.expm1(a)*H > upper_first_spacing*factor
            alo, ahi = np.where(big, a, alo), np.where(big, ahi, a)
        a = (alo+ahi)/2; E = np.expm1(a[None, :]*tu[:, None])/np.expm1(a)[None, :]
    hu = E*(s['top_y']-floorup)
    uy = floorup+hu
    upvs = np.array([[xx[0], s['lip_y']], [lip, s['lip_y']], [nose, s['lip_y']+s['cowl_thickness']],
                     [xx[-1], s['lip_y']+s['cowl_thickness']]])
    us = smooth_slopes(xx, upvs, blend)
    start = np.clip((xx-lip)/blend, 0, 1)
    us *= start*start*(3-2*start)
    ux = xx-us*hu*np.exp(-(hu/normal_band)**2)*(1-E)**2*taper
    ux[0], ux[-1] = xx, xx
    if split:
        lx, yy, ux, uy, height, hu, xx = (refine2(a) for a in (lx, yy, ux, uy, height, hu, xx))
        nx, nl, nu = len(xx), 2*nl, 2*nu
    lower = np.arange((nl+1)*nx).reshape(nl+1, nx)
    points = np.c_[lx.ravel(), yy.ravel()]
    upper = np.empty((nu+1, nx), int)
    shared = xx <= lip
    upper[0, shared] = lower[-1, shared]
    remaining = np.flatnonzero(~shared)
    upper[0, remaining] = np.arange(len(points), len(points)+len(remaining))
    points = np.vstack([points, np.c_[ux[0, remaining], uy[0, remaining]]])
    upper[1:] = np.arange(len(points), len(points)+nu*nx).reshape(nu, nx)
    points = np.vstack([points, np.c_[ux[1:].ravel(), uy[1:].ravel()]])
    quads = np.vstack([np.c_[g[:-1, :-1].ravel(), g[:-1, 1:].ravel(), g[1:, 1:].ravel(), g[1:, :-1].ravel()]
                       for g in (lower, upper)])

    def edges(row):
        return np.c_[row[:-1], row[1:]]
    bottom = edges(lower[0])
    mid = (xx[:-1]+xx[1:])/2
    outside, cw = mid < 0, mid >= lip
    markers = dict(lower=bottom[~outside],
                   farfield=np.vstack([bottom[outside], edges(upper[-1])[:, ::-1]]),
                   inlet=np.vstack([np.c_[lower[1:, 0], lower[:-1, 0]], np.c_[upper[1:, 0], upper[:-1, 0]]]),
                   outlet=np.c_[lower[:-1, -1], lower[1:, -1]],
                   external_outlet=np.c_[upper[:-1, -1], upper[1:, -1]],
                   cowl=np.vstack([edges(lower[-1])[cw][:, ::-1], edges(upper[0])[cw]]))
    meta = dict(nx=nx-1, lower_layers=nl, upper_layers=nu, factor=factor, wall_normal_band_m=normal_band,
                corner_blend_half_width_m=blend, cowl_tip_dx_target_m=LIP_DX*factor*length_scale,
                length_scale=length_scale,
                lower_first_spacing_min_m=float(height[1].min()), lower_first_spacing_max_m=float(height[1].max()),
                upper_first_spacing_min_m=float(hu[1].min()), upper_first_spacing_max_m=float(hu[1].max()),
                geometry_exact=True, cowl_radius_m=0., split_into_four=bool(split))
    if return_grids:
        return points, quads, markers, meta, dict(lower=lower, upper=upper)
    return points, quads, markers, meta


def check(points, quads, markers, s):
    """Topology, positive cells, boundary closure, area and fluid-domain containment."""
    poly = points[quads]
    area = .5*np.sum(cross(poly, np.roll(poly, -1, axis=1)), axis=1)
    require(np.all(area > 0), 'Nonpositive quad')
    u, v = np.roll(poly, -1, axis=1)-poly, np.roll(poly, 1, axis=1)-poly
    jac = cross(u, v)/(np.linalg.norm(u, axis=2)*np.linalg.norm(v, axis=2))
    require(np.all(jac > 0), 'Nonconvex quad corner')
    edges = np.sort(np.concatenate([quads[:, [i, (i+1) % 4]] for i in range(4)]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    require(set(counts.tolist()) <= {1, 2}, 'Hanging or overlapping edges')
    boundary = {tuple(e) for e in np.sort(np.concatenate(list(markers.values())), axis=1).tolist()}
    require({tuple(e) for e in unique[counts == 1].tolist()} == boundary, 'Boundary markers incomplete')
    vertices, _ = polygon(s)
    expected = .5*abs(np.sum(cross(vertices, np.roll(vertices, -1, axis=0))))
    require(abs(area.sum()-expected) < 1e-10, 'Mesh area differs from fluid polygon')
    vs = floor_vertices(s)
    require(np.min(points[:, 1]-np.interp(points[:, 0], vs[:, 0], vs[:, 1])) > -1e-12, 'Node below floor')
    lip = s['ramp_length']
    cowlupper = np.interp(points[:, 0], [lip, lip+s['cowl_nose_length'], s['outlet_x']],
                          [s['lip_y'], s['lip_y']+s['cowl_thickness'], s['lip_y']+s['cowl_thickness']])
    solid = (points[:, 0] > lip+1e-12) & (points[:, 1] > s['lip_y']+1e-12) & (points[:, 1] < cowlupper-1e-12)
    require(not solid.any(), 'Node inside cowl')
    return dict(nodes=len(points), quads=len(quads), minimum_scaled_jacobian=float(jac.min()),
                domain_area_m2=float(area.sum()), boundary_complete=True, no_hanging_edges=True)


def generate(settings_path, columns_mesh, output, factor=1, split=False):
    s = read_settings(Path(settings_path)) if 'lip_y' not in json.loads(Path(settings_path).read_text()) \
        else json.loads(Path(settings_path).read_text())
    p0, _, m0 = read_su2(Path(columns_mesh))
    points, quads, markers, meta = mesh(s, bottom_columns(p0, m0), factor, split)
    quality = check(points, quads, markers, s)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_su2(output/'mesh.su2', points, {4: quads}, markers)
    rp, _, rm = read_su2(output/'mesh.su2')
    require(np.array_equal(points, rp) and all(np.array_equal(markers[k], rm[k]) for k in markers),
            'Mesh round trip differs')
    s = dict(s, output=str(output.resolve()), mesh_policy='structured_two_block_quad')
    (output/'settings.json').write_text(json.dumps(s, indent=2)+'\n')
    (output/'quality.json').write_text(json.dumps(dict(**quality, meta=meta), indent=2)+'\n')
    return quality, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('settings', help='resolved settings.json of a generated intake (or an input JSON)')
    ap.add_argument('--columns-from', required=True, help='existing SU2 mesh whose bottom nodes give the columns')
    ap.add_argument('--output', required=True, help='new directory; refused if it exists')
    ap.add_argument('--coarse', action='store_true', help='half resolution (factor 2)')
    ap.add_argument('--split', action='store_true', help='split every quad into four (nested refinement)')
    a = ap.parse_args()
    quality, meta = generate(a.settings, a.columns_from, a.output, 2 if a.coarse else 1, a.split)
    print(json.dumps(dict(**quality, meta=meta)))


if __name__ == '__main__':
    main()
