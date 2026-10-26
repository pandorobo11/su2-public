#!/usr/bin/env python3
"""Section metrics, field balance and the lip qualification gate for the intake sweep.

Sections are vertical lines integrated with scripts/intake_section.py (conservative
variables linear in each triangle, 5-point Gauss per interval). Averages weight by
the forward axial mass flux. The mixed-out state conserves the section's mass,
x/y momentum and energy fluxes (supersonic root). Percentiles are forward-mass-flux
weighted. All ratios use the freestream static/total state; values are
nondimensional and do not depend on the dimensional freestream for a perfect gas.
"""
import json
import math
from pathlib import Path

import numpy as np

from generate_airfoil_mesh import read_su2
from intake_euler import boundary_integrals, primitives, read_restart, require
from intake_profiles import boundary_y, capture_upper
from intake_section import SectionGeometry, section

trapezoid = getattr(np, 'trapezoid', None) or np.trapz
GATE_KEYS = ('p0_recovery', 'T_ratio', 'p_ratio', 'mach', 'massflow')


def freestream(phys):
    g, R, M = phys['gamma'], phys['gas_constant'], phys['mach']
    f = 1+(g-1)/2*M*M
    rho = phys['pressure']/(R*phys['temperature'])
    V = M*math.sqrt(g*R*phys['temperature'])
    return dict(p=phys['pressure'], T=phys['temperature'], rho=rho, V=V, rhoV=rho*V,
                p0=phys['pressure']*f**(g/(g-1)), T0=phys['temperature']*f)


class Mesh:
    """Mesh, markers, resolved settings and cached section geometry."""

    def __init__(self, mesh_dir):
        mesh_dir = Path(mesh_dir)
        self.points, self.cells, self.markers = read_su2(mesh_dir/'mesh.su2')
        self.settings = json.loads((mesh_dir/'settings.json').read_text())
        self.geom = SectionGeometry(self.points, self.cells)

    def state(self, restart):
        d = read_restart(Path(restart))
        if 'PointID' in d:  # ASCII restart; the binary one is written in global point order
            ids = d['PointID'].astype(int)
            require(np.array_equal(np.sort(ids), np.arange(len(self.points))), 'Restart PointID mismatch')
            order = np.argsort(ids)
            d = {k: v[order] for k, v in d.items()}
        require(len(d['x']) == len(self.points), 'Restart point count differs from mesh')
        require(np.allclose(np.c_[d['x'], d['y']], self.points, rtol=0, atol=1e-9), 'Restart coordinates differ from mesh')
        return d

    def flow_settings(self, phys):
        return dict(self.settings, gamma=phys['gamma'], gas_constant=phys['gas_constant'])


def mixed_out(yy, q, g, R):
    rho = q['Density']
    u, v = q['Momentum_x']/rho, q['Momentum_y']/rho
    P = (g-1)*(q['Energy']-0.5*rho*(u*u+v*v))
    H = (q['Energy']+P)/rho
    h = yy[-1]-yy[0]
    m, Fx, Fy, E = (trapezoid(f, yy) for f in (rho*u, rho*u*u+P, rho*u*v, rho*u*H))
    k = g/(g-1)
    vm = Fy/m
    a, b, c = .5-k, k*Fx/m, vm*vm/2-E/m
    disc = b*b-4*a*c
    if m <= 0 or disc < 0:
        return None
    um = (-b-math.sqrt(disc))/(2*a)
    pm = Fx/h-m*um/h
    rm = m/(um*h)
    if pm <= 0 or rm <= 0:
        return None
    Mm = math.hypot(um, vm)/math.sqrt(g*pm/rm)
    return dict(p0=pm*(1+(g-1)/2*Mm*Mm)**k, T=pm/(rm*R), p=pm, mach=Mm, angle_deg=math.degrees(math.atan2(vm, um)))


def weighted_percentiles(x, w, qs):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    cw = cw/cw[-1]
    return [float(np.interp(q, cw, x[order])) for q in qs]


def section_metrics(mesh, data, phys, x, lower, upper, full=True):
    """Averages on the vertical section x from y=lower to y=upper."""
    s = mesh.flow_settings(phys)
    ref = freestream(phys)
    integral, profile = section(mesh.points, mesh.cells, data, s, x, lower, upper, geometry=mesh.geom)
    av = integral['forward_massflow_averages']
    out = dict(x_m=x, lower_y_m=lower, upper_y_m=upper, height_m=upper-lower,
               p0_recovery=av['total_pressure']/ref['p0'], T_ratio=av['temperature']/ref['T'],
               p_ratio=av['pressure']/ref['p'], rho_ratio=av['density']/ref['rho'], mach=av['mach'],
               flow_angle_deg=math.degrees(math.atan2(av['v'], av['u'])), massflow=integral['massflow'],
               massflow_over_rhoV_capture=integral['massflow']/(ref['rhoV']*mesh.settings['lip_y']),
               T0_error=av['total_temperature']/ref['T0']-1,
               reverse_massflow_fraction=integral['reverse_massflow_fraction'],
               subsonic_forward_massflow_fraction=integral['subsonic_forward_massflow_fraction'],
               min_normal_mach=integral['min_normal_mach'], max_normal_mach=integral['max_normal_mach'])
    if not full:
        return out, None
    g, R = phys['gamma'], phys['gas_constant']
    y = np.asarray(profile['y'])
    yy = np.linspace(lower, upper, 20001)
    q = {k: np.interp(yy, y, np.asarray(profile[k])) for k in ('Density', 'Momentum_x', 'Momentum_y', 'Energy')}
    mo = mixed_out(yy, q, g, R)
    out.update(p0_recovery_mixedout=mo and mo['p0']/ref['p0'], T_ratio_mixedout=mo and mo['T']/ref['T'],
               p_ratio_mixedout=mo and mo['p']/ref['p'], mach_mixedout=mo and mo['mach'])
    rho = q['Density']
    u, v = q['Momentum_x']/rho, q['Momentum_y']/rho
    P = (g-1)*(q['Energy']-0.5*rho*(u*u+v*v))
    T = P/(rho*R)
    M = np.hypot(u, v)/np.sqrt(g*P/rho)
    p0 = P*(1+(g-1)/2*M*M)**(g/(g-1))/ref['p0']
    w = np.maximum(rho*u, 0)
    if w.sum() > 0:
        for tag, arr in (('p0_recovery', p0), ('T_ratio', T/ref['T'])):
            for qq, val in zip((5, 50, 95), weighted_percentiles(arr, w, (.05, .5, .95))):
                out[f'{tag}_p{qq:02d}'] = val
        fs = (np.abs(p0-1) < .005) & (np.abs(T/ref['T']-1) < .005)
        out['freestream_state_mass_fraction'] = float(w[fs].sum()/w.sum())
    prof = dict(height_above_wall_mm=1000*(y-lower), p0_recovery=np.asarray(profile['total_pressure'])/ref['p0'],
                T_ratio=np.asarray(profile['temperature'])/ref['T'], p_ratio=np.asarray(profile['pressure'])/ref['p'],
                rho_ratio=np.asarray(profile['density'])/ref['rho'], mach=np.asarray(profile['mach']),
                u_over_V=np.asarray(profile['u'])/ref['V'], v_over_V=np.asarray(profile['v'])/ref['V'],
                flow_angle_deg=np.degrees(np.arctan2(profile['v'], profile['u'])),
                T0_ratio=np.asarray(profile['total_temperature'])/ref['T0'])
    return out, prof


def lip_sections(mesh, data, phys, pre_offset, full=False):
    """Cowl-lip plane (physical aperture) and the plane pre_offset upstream of it."""
    s = mesh.settings
    res = {}
    for name, x in (('lip', s['ramp_length']), ('pre', s['ramp_length']-pre_offset)):
        lower = boundary_y(mesh.points, mesh.markers, 'lower', x)
        res[name] = section_metrics(mesh, data, phys, x, lower, s['lip_y'], full=full)
    return res


def stations(settings, analysis):
    """Streamwise stations: ramps (captured streamtube), pre-lip, lip, isolator, outlet."""
    from intake_euler import ramp_vertices
    verts = [(0., 0.), *ramp_vertices(settings)]
    lip = settings['ramp_length']
    out = []
    for k, (a, b) in enumerate(zip(verts, verts[1:]), 1):
        for j, f in enumerate((.5, .95), 1):
            x = a[0]+f*(b[0]-a[0])
            if x < lip-analysis['pre_lip_offset_m']:
                out.append(dict(id=f'R{k}-{j}', kind='ramp', x=x))
    out += [dict(id='C-pre', kind='pre_lip', x=lip-analysis['pre_lip_offset_m']), dict(id='C-lip', kind='lip', x=lip)]
    start, end = verts[-1][0], settings['outlet_x']
    x, i = start+analysis['isolator_offset_m'], 1
    while x < end-analysis['outlet_inset_m']-1e-9:
        out.append(dict(id=f'I{i:02d}', kind='isolator', x=x, isolator_distance_m=x-start))
        x, i = x+analysis['isolator_spacing_m'], i+1
    out.append(dict(id='OUT', kind='outlet', x=end-analysis['outlet_inset_m'], isolator_distance_m=end-analysis['outlet_inset_m']-start))
    return out


def station_metrics(mesh, data, phys, station, target_massflow=None):
    """Integrate one station. Ramp stations use the wall-anchored streamtube carrying target_massflow."""
    s = mesh.settings
    x = station['x']
    lower = boundary_y(mesh.points, mesh.markers, 'lower', x)
    if station['kind'] == 'ramp':
        require(target_massflow is not None and target_massflow > 0, 'Ramp station needs the captured mass flow')
        upper = capture_upper(_profile(mesh, data, phys, x, lower), target_massflow)
    elif station['kind'] in ('pre_lip', 'lip'):
        upper = s['lip_y']
    else:
        upper = boundary_y(mesh.points, mesh.markers, 'cowl', x)
    return section_metrics(mesh, data, phys, x, lower, upper)


def _profile(mesh, data, phys, x, lower):
    top = boundary_y(mesh.points, mesh.markers, 'farfield', x, highest=True)
    _, profile = section(mesh.points, mesh.cells, data, mesh.flow_settings(phys), x, lower, top, geometry=mesh.geom)
    return profile


def balance(mesh, data, phys):
    """Open-boundary mass/energy imbalance and field health."""
    flow = primitives(data, mesh.flow_settings(phys))
    b = boundary_integrals(mesh.points, mesh.cells, mesh.markers, flow)
    opn = [n for n in mesh.markers if n not in ('lower', 'cowl')]
    mi = -sum(min(0, b[n]['massflow_outward']) for n in opn)
    ei = -sum(min(0, b[n]['energyflow_outward']) for n in opn)
    ref = freestream(phys)
    finite = all(np.isfinite(flow[k]).all() for k in ('density', 'pressure', 'temperature'))
    return dict(mass_imbalance=abs(sum(b[n]['massflow_outward'] for n in opn))/mi,
                energy_imbalance=abs(sum(b[n]['energyflow_outward'] for n in opn))/ei,
                finite=bool(finite), density_min=float(flow['density'].min()), pressure_min=float(flow['pressure'].min()),
                temperature_min=float(flow['temperature'].min()),
                T0_max_error=float(np.max(np.abs(flow['total_temperature']/ref['T0']-1))),
                outlet_min_normal_mach=b['outlet']['min_normal_mach'],
                external_outlet_min_normal_mach=b['external_outlet']['min_normal_mach'])


def started(pre):
    """Supersonic, attached-to-lip flow just upstream of the lip (recorded with the gate)."""
    return bool(pre['reverse_massflow_fraction'] == 0 and pre['subsonic_forward_massflow_fraction'] < 0.01)


def gate(cur, prev, history, iteration, first_reference, nonphysical_max, rules):
    """Qualification of one second-order checkpoint.

    cur/prev: dict(lip=..., pre=..., balance=...) for this and the previous
    checkpoint of the same attempt; history: dict of lists incl. Inner_Iter,
    rms[Rho], rms[RhoE] for this attempt; iteration counts within the attempt.
    """
    change = None
    if prev is not None:
        change = max(abs(cur[s][k]-prev[s][k])/max(abs(cur[s][k]), 1e-300) for s in ('lip', 'pre') for k in GATE_KEYS)
    it = np.asarray(history['Inner_Iter'])
    window = (it > iteration-500) & (it <= iteration)
    tail = {k: float(np.max(np.asarray(history[k])[window])) if window.sum() >= 450 else None
            for k in ('rms[Rho]', 'rms[RhoE]')}
    b = cur['balance']
    decades = {k: (first_reference[k]-tail[k]) if tail[k] is not None else None for k in tail}
    gates = dict(
        min_iterations=iteration >= rules['min_second_order_iterations'],
        health=bool(b['finite'] and b['density_min'] > 0 and b['pressure_min'] > 0 and b['temperature_min'] > 0
                    and nonphysical_max == 0),
        residual=all(d is not None and d >= rules['residual_decades'] for d in decades.values()),
        balance=b['mass_imbalance'] < rules['imbalance'] and b['energy_imbalance'] < rules['imbalance'],
        lip_stability=change is not None and change < rules['lip_relative_change'])
    return dict(iteration=iteration, qualified=all(gates.values()), gates=gates, lip_relative_change=change,
                residual_tail_max=tail, residual_decades=decades, started=started(cur['pre']))
