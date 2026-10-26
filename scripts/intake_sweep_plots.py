#!/usr/bin/env python3
"""Figures for the intake Mach x AoA sweep (Matplotlib, Agg backend).

Encoding used throughout: AoA lines use a diverging blue (negative) / gray (0) /
red (positive) scale; Mach lines a single-hue blue ramp, light to dark. Filled
markers are performance values (qualified and started), hollow markers are
diagnostic values (residual gate not met or start not confirmed); cases without a
usable solution have no marker. Squares mark conditions where the straight ramp
waves cross ahead of the lip (sequential theory not applicable).
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUES = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
NEG, ZERO, POS = ('#5598e7', '#184f95'), '#52514e', ('#e86a69', '#a8201f')
STATUS = dict(performance='#0ca30c', retry='#6fbf6f', diagnostic='#fab219', unstarted='#ec835a', none='#d03b3b',
              unknown='#b5b4ae', not_started='#e4e3df')
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': .8, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'axes.titlelocation': 'left', 'axes.titlesize': 9.5})
LIP_PANELS = [('p0_recovery_massavg', 'p0 recovery (mass-flux avg)'), ('p0_recovery_mixedout', 'p0 recovery (mixed-out)'),
              ('T_ratio_massavg', 'static T / T∞'), ('p_ratio_massavg', 'static p / p∞'),
              ('mach_massavg', 'Mach (mass-flux avg)'), ('massflow_over_rhoV_capture', 'mass capture ratio')]
ISO_PANELS = [('p0_recovery_massavg', 'p0 recovery'), ('T_ratio_massavg', 'static T / T∞'),
              ('p_ratio_massavg', 'static p / p∞'), ('mach_massavg', 'Mach'), ('reverse_massflow_fraction', 'reverse mass-flow fraction'),
              ('subsonic_forward_massflow_fraction', 'subsonic forward mass fraction')]
STREAM_PANELS = [('mach_massavg', 'Mach'), ('p_ratio_massavg', 'static p / p∞'), ('T_ratio_massavg', 'static T / T∞'),
                 ('rho_ratio_massavg', 'ρ / ρ∞'), ('p0_recovery_massavg', 'p0 recovery'), ('massflow_over_rhoV_capture', 'mass flow / (ρ∞V∞ h_capture)')]
PROFILE_PANELS = [('mach', 'Mach'), ('p_ratio', 'p / p∞'), ('T_ratio', 'T / T∞'), ('rho_ratio', 'ρ / ρ∞'),
                  ('u_over_V', 'u / V∞'), ('v_over_V', 'v / V∞'), ('flow_angle_deg', 'flow angle [deg]'),
                  ('p0_recovery', 'p0 / p0∞'), ('T0_ratio', 'T0 / T0∞')]


def aoa_colors(values):
    vals = sorted(set(values))
    neg, pos = [v for v in vals if v < 0], [v for v in vals if v > 0]
    out = {}
    for group, (light, dark) in ((neg, NEG), (pos, POS)):
        mags = sorted({abs(v) for v in group})
        for v in group:
            t = mags.index(abs(v))/max(len(mags)-1, 1)
            out[v] = _mix(light, dark, t)
    for v in vals:
        out.setdefault(v, ZERO)
    return out


def mach_colors(values):
    vals = sorted(set(values))
    lo, hi = 0, len(BLUES)-1
    idx = np.linspace(lo+1, hi-1, len(vals)) if len(vals) > 1 else [5]
    return {v: BLUES[int(round(i))] for v, i in zip(vals, idx)}


def _mix(a, b, t):
    ca, cb = (np.array([int(c[i:i+2], 16) for i in (1, 3, 5)]) for c in (a, b))
    return '#'+''.join(f'{int(round(x)):02x}' for x in ca+(cb-ca)*t)


def num(v):
    return v is not None and not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def fmt_aoa(v):
    return f'AoA {v:+g}°'


def fmt_mach(v):
    return f'M{v:g}'


def lines(ax, rows, key, xkey, gkey, colors, square_key=None):
    for g, c in colors.items():
        pts = sorted((r[xkey], r[key], r['use'], bool(r.get(square_key))) for r in rows
                     if r[gkey] == g and num(r.get(key)) and r['use'] in ('performance', 'diagnostic'))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2, zorder=2)
        for x, y, use, sq in pts:
            ax.plot(x, y, 's' if sq else 'o', ms=6, mfc=c if use == 'performance' else SURF, mec=c, mew=1.6, zorder=3)


def legend(fig, colors, fmt, title, extra=True, square=False):
    h = [Line2D([], [], color=c, lw=2, marker='o', mfc=c, mec=c) for c in colors.values()]
    labels = [fmt(g) for g in colors]
    if extra:
        h.append(Line2D([], [], ls='', marker='o', mfc=SURF, mec=MUTED, mew=1.6))
        labels.append('hollow: diagnostic only')
    if square:
        h.append(Line2D([], [], ls='', marker='s', mfc=MUTED, mec=MUTED))
        labels.append('square: ramp waves cross ahead of lip')
    fig.legend(h, labels, loc='lower center', ncol=min(len(h), 8), frameon=False, title=title)


def note(fig, text):
    fig.suptitle(text, x=.01, ha='left', color=MUTED, fontsize=8.5)


def grid_sheet(rows, panels, xkey, gkey, colors, xlabel, fmt, gtitle, title, path, square_key=None, ncols=3):
    n = len(panels)
    nr = int(math.ceil(n/ncols))
    fig, axs = plt.subplots(nr, ncols, figsize=(4.1*ncols, 3.2*nr+0.9), squeeze=False)
    xs = sorted({r[xkey] for r in rows})
    for ax, (k, lab) in zip(axs.flat, panels):
        lines(ax, rows, k, xkey, gkey, colors, square_key)
        ax.set_title(lab)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xs)
    for ax in list(axs.flat)[n:]:
        ax.set_visible(False)
    legend(fig, colors, fmt, gtitle, square=bool(square_key))
    note(fig, title)
    fig.tight_layout(rect=(0, .1 if nr > 1 else .16, 1, .96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- sweep figures

def sweep_figures(run, plan, lips, streams, cases):
    out = Path(run)/'analysis'/'figures'
    out.mkdir(parents=True, exist_ok=True)
    g = plan['geometry']
    tag = (f"{plan['name']}: {g['stages']}-ramp, design M{g['design_mach']:g}, T ratio {g['temperature_ratio']:g}, "
           f"CR {g['contraction_ratio']:.2f}")
    machs, aoas = sorted({r['mach'] for r in lips}), sorted({r['aoa_deg'] for r in lips})
    ac, mc = aoa_colors(aoas), mach_colors(machs)
    figs = []

    def save(name):
        figs.append(name)
        return out/name
    grid_sheet(lips, LIP_PANELS, 'mach', 'aoa_deg', ac, 'freestream Mach', fmt_aoa, 'AoA',
               f'Lip section vs Mach  —  {tag}', save('lip-vs-mach.png'))
    grid_sheet(lips, LIP_PANELS, 'aoa_deg', 'mach', mc, 'AoA [deg]', fmt_mach, 'Mach',
               f'Lip section vs AoA  —  {tag}', save('lip-vs-aoa.png'))
    theory_diff(lips, ac, mc, tag, save('theory-diff.png'))
    iso = [dict(r, use=r.get('use') or 'none') for r in streams if r.get('station') == 'OUT']
    if iso:
        grid_sheet(iso, ISO_PANELS, 'mach', 'aoa_deg', ac, 'freestream Mach', fmt_aoa, 'AoA',
                   f'Isolator exit (diagnostic)  —  {tag}', save('isolator-vs-mach.png'))
        grid_sheet(iso, ISO_PANELS, 'aoa_deg', 'mach', mc, 'AoA [deg]', fmt_mach, 'Mach',
                   f'Isolator exit (diagnostic)  —  {tag}', save('isolator-vs-aoa.png'))
    for M in machs:
        lip_profiles(run, plan, [c for c in cases if c['row']['mach'] == M], 'aoa_deg', ac, fmt_aoa,
                     f'Lip-plane profiles, Mach {M:g}  —  {tag}', save(f'lip-profiles-m{M:g}.png'))
        streamwise(run, plan, streams, 'mach', M, 'aoa_deg', ac, fmt_aoa,
                   f'Streamwise averages, Mach {M:g}  —  {tag}', save(f'streamwise-m{M:g}.png'))
    for a in aoas:
        lip_profiles(run, plan, [c for c in cases if c['row']['aoa_deg'] == a], 'mach', mc, fmt_mach,
                     f'Lip-plane profiles, AoA {a:+g}°  —  {tag}', save(f'lip-profiles-a{a:+g}.png'))
        streamwise(run, plan, streams, 'aoa_deg', a, 'mach', mc, fmt_mach,
                   f'Streamwise averages, AoA {a:+g}°  —  {tag}', save(f'streamwise-a{a:+g}.png'))
    geometry(run, plan, save('geometry.png'))
    setup_figures(run, plan, out)
    figs += ['boundary-conditions.png', 'mesh.png']
    outcome_map(lips, machs, aoas, tag, save('outcome-map.png'))
    kantrowitz(lips, machs, aoas, tag, save('kantrowitz-margin.png'))
    residual_grid(run, plan, cases, machs, aoas, tag, save('residuals.png'))
    return figs


def theory_diff(lips, ac, mc, tag, path):
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6))
    for j, (xkey, gkey, colors, xl, fmt) in enumerate([('mach', 'aoa_deg', ac, 'freestream Mach', fmt_aoa),
                                                       ('aoa_deg', 'mach', mc, 'AoA [deg]', fmt_mach)]):
        for i, (k, lab) in enumerate([('d_p0_recovery_vs_theory', 'Δ p0 recovery (CFD − theory)'),
                                      ('d_T_ratio_vs_theory', 'Δ T ratio (CFD − theory)')]):
            ax = axs[i, j]
            lines(ax, lips, k, xkey, gkey, colors, 'theory_waves_cross_ahead_of_lip')
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(lab)
            ax.set_xlabel(xl)
            ax.set_xticks(sorted({r[xkey] for r in lips}))
            if i == 0:  # above the panel so it never covers data
                ax.legend([Line2D([], [], color=c, lw=2) for c in colors.values()], [fmt(g) for g in colors],
                          ncol=min(5, len(colors)), fontsize=7.5, frameon=False, loc='lower left',
                          bbox_to_anchor=(0, 1.08), borderaxespad=0)
                ax.set_title(lab, pad=24)
    fig.legend([Line2D([], [], ls='', marker='o', mec=MUTED, mfc=MUTED), Line2D([], [], ls='', marker='s', mec=MUTED, mfc=MUTED),
                Line2D([], [], ls='', marker='o', mec=MUTED, mfc=SURF, mew=1.6)],
               ['waves do not cross ahead of lip', 'waves cross ahead of lip (theory not applicable)', 'hollow: diagnostic only'],
               loc='lower center', ncol=3, frameon=False)
    note(fig, f'CFD lip mass-flux average minus sequential oblique-shock theory  —  {tag}')
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _profiles(run, cid):
    p = Path(run)/'cases'/cid/'profiles.json'
    return json.loads(p.read_text()) if p.exists() else {}


def lip_profiles(run, plan, cases, gkey, colors, fmt, title, path):
    ls = plan['mesh']['length_scale']
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    opening = plan['geometry']['lip_opening_m']*1000
    for c in cases:
        r = c['row']
        prof = _profiles(run, r['case']).get('C-lip')
        if not prof or r['use'] not in ('performance', 'diagnostic'):
            continue
        h = np.asarray(prof['height_above_wall_mm'])
        keep = (h > .5*ls) & (h < h.max()-1.*ls)  # singular wall and lip-tip nodes are kept in the CSV only
        col = colors[r[gkey]]
        style = '-' if r['use'] == 'performance' else '--'
        for ax, k in zip(axs, ('p0_recovery', 'T_ratio', 'mach')):
            ax.plot(np.asarray(prof[k])[keep], h[keep], color=col, lw=1.8, ls=style)
    for ax, lab in zip(axs, ('p0 recovery', 'static T / T∞', 'Mach')):
        ax.set_xlabel(lab)
    axs[0].set_ylabel(f'height above ramp wall at lip plane [mm] (opening {opening:.1f})')
    h = [Line2D([], [], color=c, lw=2) for c in colors.values()]+[Line2D([], [], color=MUTED, lw=2, ls='--')]
    fig.legend(h, [fmt(g) for g in colors]+['dashed: diagnostic only'], loc='lower center', ncol=len(h), frameon=False)
    note(fig, title)
    fig.tight_layout(rect=(0, .1, 1, .95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def streamwise(run, plan, streams, fkey, fval, gkey, colors, fmt, title, path):
    rows = [r for r in streams if r[fkey] == fval and r.get('use') in ('performance', 'diagnostic') and num(r.get('mach'))]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6))
    lip, ramp_end = plan['geometry']['lip'][0], (plan['geometry'].get('ramp_end_point') or plan['geometry']['lip'])[0]
    for ax, (k, lab) in zip(axs.flat, STREAM_PANELS):
        for gval, col in colors.items():
            pts = sorted((r['x_m'], r[k], r['use']) for r in rows if r[gkey] == gval and num(r.get(k)))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.8, marker='o', ms=3.5,
                        ls='-' if pts[0][2] == 'performance' else '--')
        ax.axvline(lip, color=MUTED, lw=.8, ls=':')
        ax.axvline(ramp_end, color=MUTED, lw=.8, ls='--')
        ax.set_title(lab)
        ax.set_xlabel('section x [m]')
    h = [Line2D([], [], color=c, lw=2) for c in colors.values()]
    fig.legend(h, [fmt(g) for g in colors], loc='lower center', ncol=len(h), frameon=False)
    note(fig, title+'\ndotted: cowl lip, dashed: duct start; ramp stations average the captured streamtube; dashed lines: diagnostic')
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


MARKERS = dict(inlet=('#2a78d6', 'inlet: supersonic inlet (M∞, AoA, p∞, T∞ fixed)'),
               farfield=('#1baf7a', 'farfield: characteristic farfield'),
               lower=(INK, 'lower: Euler slip wall (ramps, duct floor)'),
               cowl=('#e87ba4', 'cowl: Euler slip wall (sharp tip)'),
               outlet=('#eb6834', 'outlet: supersonic outlet (extrapolation)'),
               external_outlet=('#eda100', 'external_outlet: supersonic outlet (extrapolation)'))


def _mesh(run):
    """Mesh, markers and, when the generator reproduces it, the structured index grids."""
    import sys
    sys.path.insert(0, str(Path(run)/'scripts'))
    from generate_airfoil_mesh import read_su2
    import intake_structured_mesh as ism
    s = json.loads((Path(run)/'mesh'/'settings.json').read_text())
    points, cells, markers = read_su2(Path(run)/'mesh'/'mesh.su2')
    grids = None
    try:
        ls = s.get('length_scale', 1.)
        pts, _, _, _, g = ism.mesh(s, ism.rule_columns(s, ls), length_scale=ls, return_grids=True)
        if pts.shape == points.shape and np.allclose(pts, points, rtol=0, atol=1e-12):
            grids = g
    except Exception:
        pass
    return s, points, cells, markers, grids


def _draw_markers(ax, points, markers, lw=2.2):
    for name, (col, _) in MARKERS.items():
        if name in markers:
            seg = points[markers[name]]
            ax.add_collection(LineCollection(seg, colors=col, linewidths=lw, zorder=3))


def _draw_grid(ax, points, cells, grids, box, stride=1):
    """Grid lines inside box (x0, x1, y0, y1); every stride-th line of each family."""
    x0, x1, y0, y1 = box
    segs = []
    if grids is not None:
        for g in grids.values():
            for line in list(g[::stride])+list(g.T[::stride])+[g[-1], g.T[-1]]:
                xy = points[line]
                inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
                keep = inside | np.r_[inside[1:], False] | np.r_[False, inside[:-1]]
                if keep.sum() >= 2:
                    segs.append(xy[keep])
    else:
        quads = list(cells.values())[0] if isinstance(cells, dict) else cells
        c = points[np.asarray(quads)].mean(axis=1)
        sel = np.asarray(quads)[(c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)]
        segs = [points[np.r_[q, q[0]]] for q in sel]
    ax.add_collection(LineCollection(segs, colors='#8a8984', linewidths=.35, zorder=1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')


def setup_figures(run, plan, out):
    """boundary-conditions.png and mesh.png from the frozen mesh (no solution needed)."""
    s, points, cells, markers, grids = _mesh(run)
    g = plan['geometry']
    H, xl, yl = s['lip_y'], s['ramp_length'], s['lip_y']
    x_min, x_max, y_top = -s['upstream_length'], s['outlet_x'], s['top_y']
    # boundary conditions: whole domain and the lip region
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9.2), gridspec_kw=dict(height_ratios=[1, 1]))
    for ax in (a1, a2):
        _draw_markers(ax, points, markers)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(False)
    a1.set_xlim(x_min-.02*x_max, x_max*1.02)
    a1.set_ylim(-.05*y_top, y_top*1.05)
    a1.annotate('', (0.55*x_min, 0.75*y_top), (0.95*x_min, 0.75*y_top-0.4*math.tan(math.radians(4))),
                arrowprops=dict(arrowstyle='->', color='#2a78d6', lw=1.5))
    a1.text(0.95*x_min, 0.82*y_top, 'freestream\nM∞, AoA', color=INK, fontsize=8)
    a1.set_title(f"{plan['name']}: boundary markers of the computational domain (capture height {H:g} m)")
    a1.legend([Line2D([], [], color=c, lw=2.5) for c, _ in MARKERS.values()], [t for _, t in MARKERS.values()],
              loc='upper center', bbox_to_anchor=(.5, .9), ncol=2, frameon=False, fontsize=8.5)
    w = .35*H
    a2.set_xlim(xl-w, xl+1.6*w)
    a2.set_ylim(yl-1.6*g['lip_opening_m'], yl+.8*g['lip_opening_m'])
    a2.plot([xl, xl], [yl-g['lip_opening_m'], yl], color='#6250d6', lw=1.4, ls='--', zorder=4)
    a2.text(xl, yl+.05*g['lip_opening_m'], 'lip plane (evaluation)', color='#6250d6', fontsize=8, ha='right')
    a2.set_title(f"lip region: opening {g['lip_opening_m']*1000:.1f} mm, duct {g['duct_gap_height_m']*1000:.1f} mm "
                 f"(CR {g['contraction_ratio']:.2f}); interior lines are not boundaries")
    note(fig, 'SU2 markers: MARKER_SUPERSONIC_INLET inlet, MARKER_FAR farfield, MARKER_EULER lower and cowl, '
              'MARKER_SUPERSONIC_OUTLET outlet and external_outlet')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out/'boundary-conditions.png', dpi=140)
    plt.close(fig)
    # mesh: overview (every 8th line) and three full-resolution windows
    verts = [(0., 0.)]+[tuple(v) for v in _ramp_vertices(s)]
    corner = verts[-2] if len(verts) > 2 else verts[0]
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
    top = fig.add_subplot(gs[0, :])
    axs = np.array([top]+[fig.add_subplot(gs[1, k]) for k in range(3)])
    _draw_grid(top, points, cells, grids, (x_min, x_max, 0, y_top), stride=8)
    top.set_title(f'whole domain (every 8th grid line), {len(points):,} nodes')
    wins = [(axs[1], (-.06*H, .1*H, -.03*H, .08*H), 'first ramp leading edge (all lines)'),
            (axs[2], (corner[0]-.08*H, corner[0]+.08*H, corner[1]-.03*H, corner[1]+.09*H), 'last ramp corner (all lines)'),
            (axs[3], (xl-.12*H, xl+.12*H, yl-.14*H, yl+.06*H), 'cowl lip and duct entrance (all lines)')]
    for ax, box, title in wins:
        _draw_grid(ax, points, cells, grids, box)
        ax.set_title(title)
    for ax in axs.flat:
        _draw_markers(ax, points, markers, lw=1.6)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(False)
    q = json.loads((Path(run)/'mesh'/'quality.json').read_text())
    meta = q.get('meta', {})
    note(fig, f"two-block structured quad mesh: {q['nodes']:,} nodes, {q['quads']:,} quads, min scaled Jacobian "
              f"{q['minimum_scaled_jacobian']:.3f}\nstreamwise columns {meta.get('nx', '?')}, layers "
              f"{meta.get('lower_layers', '?')} below / {meta.get('upper_layers', '?')} above the lip height; "
              f"first wall spacing {meta.get('lower_first_spacing_max_m', float('nan'))*1000:.2f} mm, "
              f"cowl-tip streamwise {meta.get('cowl_tip_dx_target_m', float('nan'))*1000:.2f} mm")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out/'mesh.png', dpi=150)
    plt.close(fig)


def _ramp_vertices(s):
    x = y = 0.
    out = []
    for r in s['ramps']:
        x += r['length']
        y += r['length']*math.tan(math.radians(r['angle_deg']))
        out.append((x, y))
    if s.get('ramp_end_point'):
        out[-1] = tuple(s['ramp_end_point'])
    return out


def geometry(run, plan, path):
    d = json.loads((Path(run)/'design'/'design.json').read_text())
    s = json.loads((Path(run)/'mesh'/'settings.json').read_text())
    w = d['geometry']['walls_m']
    g = plan['geometry']
    xl, yl = g['lip']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.6), gridspec_kw=dict(height_ratios=[1, 1.15]))
    seen = {}
    rp = Path(run)/'analysis'/'streamwise-sections.csv'
    if rp.exists():
        import csv
        with rp.open() as f:
            for r in csv.DictReader(f):
                if r.get('x_m') and r.get('upper_y_m'):
                    seen.setdefault(r['station'], (float(r['x_m']), float(r['lower_y_m']), float(r['upper_y_m'])))
    for ax in (a1, a2):
        for k in ('lower', 'cowl_inner', 'cowl_outer'):
            if k in w:
                ax.plot([p[0] for p in w[k]], [p[1] for p in w[k]], color=INK, lw=2)
        for st in d['stage_states']:
            ax.plot([st['origin_m'][0], st['focus_m'][0]], [st['origin_m'][1], st['focus_m'][1]], color=MUTED, lw=1, ls=':')
        for sid, (x, lo, hi) in seen.items():
            ax.plot([x, x], [lo, hi], color='#2a78d6', lw=1)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    a1.set_aspect('equal')
    a1.set_title(f"{g['stages']}-ramp, ramp angles {', '.join(f'{v:.2f}' for v in g['ramp_angles_deg'])}°; "
                 "dotted: design shocks; blue: stations (ramp stations: captured streamtube)")
    for sid, (x, lo, hi) in seen.items():
        if sid.startswith('R'):
            a1.text(x, hi, sid, fontsize=7, color=MUTED, ha='center', va='bottom')
    x0, x1 = xl-.25*s['lip_y'], s['outlet_x']
    a2.set_xlim(x0, x1)
    a2.set_ylim(yl-2.2*g['lip_opening_m'], yl+1.2*g['lip_opening_m'])
    for sid, (x, lo, hi) in seen.items():
        if x0 <= x <= x1 and not sid.startswith('R'):
            a2.text(x, yl+.08*g['lip_opening_m'], sid, rotation=90, fontsize=7, color=MUTED, ha='center', va='bottom', clip_on=True)
    a2.set_title(f"lip and duct (stretched vertically): opening {g['lip_opening_m']*1000:.1f} mm, "
                 f"duct {g['duct_gap_height_m']*1000:.1f} mm, CR {g['contraction_ratio']:.3f}")
    note(fig, f"{plan['name']}: capture {g['capture_height_m']:g} m, mesh {plan['mesh']['nodes']:,} nodes")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _category(r):
    if r['outcome'] in ('unknown', 'not_started', None):
        return 'unknown' if r['outcome'] == 'unknown' else 'not_started'
    if r['use'] == 'performance':
        return 'retry' if (r.get('attempts') or 0) > 1 else 'performance'
    if r['use'] == 'diagnostic':
        return 'unstarted' if r.get('started') is False else 'diagnostic'
    return 'none'


LABELS = dict(performance='qualified', retry='qualified after retry', diagnostic='diagnostic (gate not met)',
              unstarted='start not confirmed', none='failed / no solution', unknown='outcome unknown', not_started='not started')


def outcome_map(lips, machs, aoas, tag, path):
    fig, ax = plt.subplots(figsize=(1.3*len(machs)+3.2, .8*len(aoas)+1.8))
    ax.grid(False)
    for r in lips:
        i, j = machs.index(r['mach']), aoas.index(r['aoa_deg'])
        cat = _category(r)
        ax.add_patch(plt.Rectangle((i-.47, j-.45), .94, .9, color=STATUS[cat], lw=0))
        v = r.get('p0_recovery_massavg')
        txt = LABELS[cat].split(' ')[0]+(f'\np0 {v:.3f}' if num(v) else '')
        ax.text(i, j, txt, ha='center', va='center', fontsize=7.5, color=INK)
    ax.set_xlim(-.5, len(machs)-.5)
    ax.set_ylim(-.5, len(aoas)-.5)
    ax.set_xticks(range(len(machs)))
    ax.set_xticklabels([f'{m:g}' for m in machs])
    ax.set_yticks(range(len(aoas)))
    ax.set_yticklabels([f'{a:+g}' for a in aoas])
    ax.set_xlabel('freestream Mach')
    ax.set_ylabel('AoA [deg]')
    used = sorted({_category(r) for r in lips}, key=list(STATUS).index)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=STATUS[c]) for c in used], [LABELS[c] for c in used],
              loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    note(fig, f'Outcome per case  —  {tag}')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def kantrowitz(lips, machs, aoas, tag, path):
    fig, ax = plt.subplots(figsize=(1.3*len(machs)+2.4, .8*len(aoas)+1.8))
    ax.grid(False)
    vals = [r.get('kantrowitz_margin') for r in lips if num(r.get('kantrowitz_margin'))]
    lo, hi = (min(vals), max(vals)) if vals else (1, 2)
    for r in lips:
        v = r.get('kantrowitz_margin')
        if not num(v):
            continue
        i, j = machs.index(r['mach']), aoas.index(r['aoa_deg'])
        t = (v-lo)/max(hi-lo, 1e-9)
        col = '#d03b3b' if v < 1 else _mix('#cde2fb', '#256abf', t)
        ax.add_patch(plt.Rectangle((i-.47, j-.45), .94, .9, color=col, lw=0))
        ax.text(i, j, f"{v:.2f}\n(max CR {r['kantrowitz_max_contraction']:.2f})", ha='center', va='center', fontsize=7.5,
                color='#fcfcfb' if t > .6 or v < 1 else INK)
    ax.set_xlim(-.5, len(machs)-.5)
    ax.set_ylim(-.5, len(aoas)-.5)
    ax.set_xticks(range(len(machs)))
    ax.set_xticklabels([f'{m:g}' for m in machs])
    ax.set_yticks(range(len(aoas)))
    ax.set_yticklabels([f'{a:+g}' for a in aoas])
    ax.set_xlabel('freestream Mach')
    ax.set_ylabel('AoA [deg]')
    note(fig, f'Kantrowitz margin = allowed / geometric contraction (theory lip Mach; <1 red)  —  {tag}')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _history(stage):
    import csv
    import gzip
    p = Path(stage)/'history.csv'
    if not p.exists():
        p = p.with_name('history.csv.gz')  # compressed by scripts/sweep_retention.py after review
        if not p.exists():
            return None
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        rd = csv.reader(f)
        head = [h.strip().strip('"') for h in next(rd)]
        rows = [r for r in rd if len(r) == len(head)]
    if not rows:
        return None
    a = np.array([[float(v) for v in r] for r in rows])
    return {k: a[:, i] for i, k in enumerate(head)}


def residual_series(run, cid, state):
    """Concatenated rms[Rho] over all stages in execution order."""
    out, offset = [], 0
    for a in state.get('attempts', []):
        h = _history(Path(run)/'cases'/cid/a['name'])
        if h is None:
            continue
        it = h['Inner_Iter']+offset
        out.append((a['name'], a['status'], it, h['rms[Rho]']))
        offset = it[-1]+1
    return out


def residual_grid(run, plan, cases, machs, aoas, tag, path):
    fig, axs = plt.subplots(len(aoas), len(machs), figsize=(2.6*len(machs), 1.9*len(aoas)+.8), squeeze=False,
                            sharex=False, sharey=True)
    colors = dict(completed=MUTED, qualified='#0ca30c', unconverged='#eda100', failed='#d03b3b', error='#d03b3b')
    for c in cases:
        r = c['row']
        ax = axs[len(aoas)-1-aoas.index(r['aoa_deg']), machs.index(r['mach'])]
        for name, st, it, res in residual_series(run, r['case'], c['state']):
            ax.plot(it, res, color=colors.get(st, MUTED), lw=.8)
        ax.set_title(f"{r['case']}: {_category(r)}", fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel('iteration (all stages)', fontsize=7.5)
    for ax in axs[:, 0]:
        ax.set_ylabel('log10 rms[ρ]', fontsize=7.5)
    fig.legend([Line2D([], [], color=v, lw=2) for v in colors.values()][:4], list(colors)[:4], loc='lower center',
               ncol=4, frameon=False)
    note(fig, f'Residual histories (rows: AoA, top = largest; columns: Mach)  —  {tag}')
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- per-case figures

def case_figures(run, cid, state, result, profiles):
    out = Path(run)/'cases'/cid/'plots'
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = dict(completed=MUTED, qualified='#0ca30c', unconverged='#eda100', failed='#d03b3b', error='#d03b3b')
    for name, st, it, res in residual_series(run, cid, state):
        ax.plot(it, res, color=colors.get(st, MUTED), lw=1, label=f'{name}: {st}')
    ax.set_xlabel('iteration (all stages, concatenated)')
    ax.set_ylabel('log10 rms[ρ]')
    ax.legend(frameon=False, fontsize=8)
    note(fig, f"{cid}: outcome {state.get('outcome')}, use {(result or {}).get('use', 'none')}")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out/'residuals.png', dpi=130)
    plt.close(fig)
    if not profiles:
        return
    fig, axs = plt.subplots(3, 3, figsize=(11.5, 9.5))
    ls = json.loads((Path(run)/'plan.json').read_text())['mesh']['length_scale']
    for i, (sid, p) in enumerate(profiles.items()):
        h = np.asarray(p['height_above_wall_mm'])
        eta = h/h.max()
        # the cowl-tip node on the lip plane is singular; it stays in the CSV only
        keep = h < h.max()-1.*ls if sid == 'C-lip' else np.ones(len(h), bool)
        for ax, (k, lab) in zip(axs.flat, PROFILE_PANELS):
            ax.plot(np.asarray(p[k])[keep], eta[keep], color=CAT[i % len(CAT)], lw=1.4, label=sid)
            ax.set_xlabel(lab)
            ax.set_ylim(0, 1)
    for ax in axs[:, 0]:
        ax.set_ylabel('height / section height')
    axs[0, 0].legend(frameon=False, fontsize=8)
    note(fig, f"{cid}: transverse profiles at the lip plane (without the 1 mm-scaled cowl-tip band) and isolator stations "
              f"({(result or {}).get('use')})")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out/'profiles.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- comparison of runs

def compare_figures(runs, out):
    import csv
    out.mkdir(parents=True)
    data = []
    for r in runs:
        plan = json.loads((r/'plan.json').read_text())
        rows = []
        for x in csv.DictReader((r/'analysis'/'lip-results.csv').open()):
            rows.append({k: _val(v) for k, v in x.items()})
        data.append((plan['name'], rows))
    merged = []
    for name, rows in data:
        merged += [dict(run=name, **x) for x in rows]
    with (out/'compare-lip-results.csv').open('w', newline='') as f:
        keys = list(dict.fromkeys(k for x in merged for k in x))
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(merged)
    quantities = [('p0_recovery_massavg', 'p0 recovery'), ('T_ratio_massavg', 'static T / T∞'),
                  ('p_ratio_massavg', 'static p / p∞'), ('mach_massavg', 'Mach')]
    for xkey, fkey, xlab, ffmt, name in (('mach', 'aoa_deg', 'freestream Mach', fmt_aoa, 'compare-lip-vs-mach.png'),
                                         ('aoa_deg', 'mach', 'AoA [deg]', fmt_mach, 'compare-lip-vs-aoa.png')):
        facets = sorted({x[fkey] for _, rows in data for x in rows})
        fig, axs = plt.subplots(len(quantities), len(facets), figsize=(2.9*len(facets), 2.5*len(quantities)+1),
                                squeeze=False, sharey='row')
        for j, fv in enumerate(facets):
            for i, (k, lab) in enumerate(quantities):
                ax = axs[i, j]
                for n, (rname, rows) in enumerate(data):
                    pts = sorted((x[xkey], x[k], x['use']) for x in rows if x[fkey] == fv and num(x.get(k))
                                 and x['use'] in ('performance', 'diagnostic'))
                    if pts:
                        col = CAT[n % len(CAT)]
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2)
                        for xx, yy, u in pts:
                            ax.plot(xx, yy, 'o', ms=5, mfc=col if u == 'performance' else SURF, mec=col, mew=1.5)
                if i == 0:
                    ax.set_title(ffmt(fv))
                if j == 0:
                    ax.set_ylabel(lab)
                if i == len(quantities)-1:
                    ax.set_xlabel(xlab)
        fig.legend([Line2D([], [], color=CAT[n % len(CAT)], lw=2) for n in range(len(data))]+
                   [Line2D([], [], ls='', marker='o', mfc=SURF, mec=MUTED, mew=1.5)],
                   [d[0] for d in data]+['hollow: diagnostic only'], loc='lower center', ncol=len(data)+1, frameon=False)
        note(fig, 'Lip section (mass-flux average), runs overlaid')
        fig.tight_layout(rect=(0, .05, 1, .96))
        fig.savefig(out/name, dpi=140)
        plt.close(fig)


def _val(v):
    if v == '':
        return None
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return float(v)
    except ValueError:
        return v
