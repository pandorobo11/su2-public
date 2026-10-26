#!/usr/bin/env python3
"""Record ramp-wall and transverse intake profiles from saved steady solutions.

Sections are vertical (normal +x). External sections retain the full domain and
a mass-flow-matched captured streamtube, anchored on the solid lower wall.
Internal sections cover the complete channel. Results remain diagnostic data.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from benchmark_su2 import read_config
from intake_euler import (is_rans, physics_signature, primitives, ramp_vertices, ramps,
                          read_csv, read_su2, require, shock)
from intake_section import SectionGeometry, section


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    require(bool(rows), 'No rows to record')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def references(settings):
    gamma, gas = settings['gamma'], settings['gas_constant']
    rho = settings['pressure']/(gas*settings['temperature'])
    speed = settings['mach']*math.sqrt(gamma*gas*settings['temperature'])
    factor = 1+(gamma-1)/2*settings['mach']**2
    return dict(pressure=settings['pressure'], temperature=settings['temperature'],
                density=rho, speed=speed, total_temperature=settings['temperature']*factor,
                total_pressure=settings['pressure']*factor**(gamma/(gamma-1)))


def derived_profile(profile, settings):
    result = {key: np.asarray(value) for key, value in profile.items()}
    ref = references(settings)
    for key in ('pressure', 'temperature', 'density'):
        result[key+'_ratio'] = result[key]/ref[key]
    result['total_pressure_recovery'] = result['total_pressure']/ref['total_pressure']
    result['total_temperature_ratio'] = result['total_temperature']/ref['total_temperature']
    result['u_over_uinf'], result['v_over_uinf'] = result['u']/ref['speed'], result['v']/ref['speed']
    result['flow_angle_deg'] = np.degrees(np.arctan2(result['v'], result['u']))
    result['normal_mach'] = result['u']/result['sound_speed']
    gamma = settings['gamma']
    result['entropy_increase_J_kg_K'] = settings['gas_constant']/(gamma-1)*(
        np.log(result['pressure_ratio'])-gamma*np.log(result['density_ratio']))
    y = result['y']
    result['eta'] = (y-y[0])/(y[-1]-y[0])
    result['height_above_lower_mm'] = 1000*(y-y[0])
    require(all(np.isfinite(values).all() for values in result.values()), 'Nonfinite profile')
    return {key: values.tolist() for key, values in result.items()}


def capture_upper(profile, target):
    """Invert the exact integral of piecewise-linear axial momentum density."""
    y, flux = np.asarray(profile['y']), np.asarray(profile['Momentum_x'])
    require(np.isfinite(target) and target > 0, 'Invalid captured mass flow')
    require(np.isfinite(y).all() and np.isfinite(flux).all() and np.all(np.diff(y) > 0),
            'Invalid streamfunction profile')
    require(np.all(flux >= 0), 'Reverse flow prevents a unique wall-anchored captured streamtube')
    dy = np.diff(y)
    increments = (flux[:-1]+flux[1:])*dy/2
    cumulative = np.r_[0., np.cumsum(increments)]
    require(target < cumulative[-1], 'Captured mass flow exceeds external section capacity')
    index = int(np.searchsorted(cumulative, target, side='right')-1)
    mass = target-cumulative[index]
    f0, slope = flux[index], (flux[index+1]-flux[index])/dy[index]
    if mass == 0:
        distance = 0.
    elif slope == 0:
        require(f0 > 0, 'Zero-flow interval')
        distance = mass/f0
    else:
        discriminant = f0*f0+2*slope*mass
        require(discriminant >= 0, 'Invalid streamfunction discriminant')
        distance = 2*mass/(f0+math.sqrt(discriminant))
    require(-1e-12 <= distance <= dy[index]+1e-12, 'Streamfunction inversion leaves interval')
    return float(y[index]+np.clip(distance, 0, dy[index]))


def boundary_y(points, markers, name, x, highest=False):
    edges = markers.get(name, np.empty((0, 2), dtype=int))
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    dx = b[:, 0]-a[:, 0]
    keep = (np.minimum(a[:, 0], b[:, 0])-1e-12 <= x) & (np.maximum(a[:, 0], b[:, 0])+1e-12 >= x) & (dx != 0)
    ys = a[keep, 1]+(x-a[keep, 0])*(b[keep, 1]-a[keep, 1])/dx[keep]
    require(len(ys) > 0, f'No {name} boundary at x={x}')
    return float(ys.max() if highest else ys.min())


def stations(settings, fractions=(0.5, 0.95), spacing=0.1, offset=0.05, lip_offset=0.005):
    require(bool(fractions) and all(np.isfinite(f) and 0 < f < 1 for f in fractions), 'Invalid ramp fractions')
    require(len(set(fractions)) == len(fractions), 'Duplicate ramp fractions')
    require(np.isfinite([spacing, offset, lip_offset]).all() and spacing > 0 and offset > 0 and lip_offset > 0,
            'Invalid section spacing or offset')
    vertices = [(0., 0.), *ramp_vertices(settings)]
    require(all(b[0] > a[0] for a, b in zip(vertices, vertices[1:])), 'Unordered ramp vertices')
    result = []
    for stage, (a, b) in enumerate(zip(vertices, vertices[1:]), 1):
        for index, fraction in enumerate(sorted(fractions), 1):
            result.append(dict(id=f'R{stage}-{index}', kind='ramp', ramp_stage=stage,
                               fraction=fraction, x=a[0]+fraction*(b[0]-a[0])))
    lip = settings['ramp_length']
    require(0 < lip-lip_offset and lip+lip_offset < settings['outlet_x'], 'Cowl sections outside domain')
    result += [dict(id='C-pre', kind='cowl', ramp_stage=len(vertices)-1, x=lip-lip_offset),
               dict(id='C-lip', kind='cowl', ramp_stage=len(vertices)-1, x=lip),
               dict(id='C-post', kind='cowl', ramp_stage=len(vertices)-1, x=lip+lip_offset)]
    start, end = vertices[-1][0], settings['outlet_x']
    require(start+offset < end, 'Isolator too short for first station')
    count = int(math.ceil((end-start-offset)/spacing))
    for index in range(count):
        x = start+offset+index*spacing
        if x < end-1e-12:
            result.append(dict(id=f'I{index+1:02d}', kind='isolator', ramp_stage=None,
                               x=x, isolator_distance=x-start))
    result.sort(key=lambda row: row['x'])
    return result


def section_bounds(points, markers, settings, x):
    """At the lip, use the physical aperture rather than a mass-matched tube."""
    lower = boundary_y(points, markers, 'lower', x)
    internal = x >= settings['ramp_length']
    upper = (boundary_y(points, markers, 'cowl', x) if internal else
             boundary_y(points, markers, 'farfield', x, highest=True))
    return lower, upper, internal


def aperture_contours(raw, target, upper):
    """Retain signed tip flux; invert only the monotone interior prefix."""
    y, flux = np.asarray(raw['y']), np.asarray(raw['Momentum_x'])
    negative = np.flatnonzero(flux < 0)
    end = int(negative[0]) if len(negative) else len(y)
    prefix = dict(y=y[:end], Momentum_x=flux[:end])
    inside = [capture_upper(prefix, target*f) for f in (.25, .5, .75)]
    require(y[0] < inside[0] < inside[1] < inside[2] < upper, 'Lip interior contours leave aperture')
    return inside+[upper]


def record_streamtubes(all_cases, output, samples=161):
    """Dense external capture boundaries, anchored by actual lip-aperture flux."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    require(isinstance(samples, int) and samples >= 2, 'At least two streamtube samples required')
    fig, axes = plt.subplots(len(all_cases), 1, figsize=(11.6, 3.4*len(all_cases)),
                             squeeze=False, layout='constrained')
    colors = plt.get_cmap('tab10').colors
    records = []
    for index, (ax, (item, points, markers, geometry)) in enumerate(zip(axes[:, 0], all_cases)):
        color = colors[index % len(colors)]
        settings, label = item['settings'], item['case']
        saved = read_csv(Path(item['source']['result'])/'second/restart_flow.csv')
        order = np.argsort(saved['PointID'])
        data = {key: values[order] for key, values in saved.items()}
        lip = settings['ramp_length']
        lower, upper, _ = section_bounds(points, markers, settings, lip)
        aperture, _ = section(points, geometry.cells, data, settings, lip, lower, upper, geometry=geometry)
        target = aperture['massflow']
        xs = np.unique(np.r_[np.linspace(0, lip, samples),
            [s['station']['x'] for s in item['stations'] if s['station']['x'] < lip]])
        rows, max_error = [], 0.
        try:
            for x in xs:
                lo = boundary_y(points, markers, 'lower', float(x))
                _, raw = section(points, geometry.cells, data, settings, float(x), lo, settings['top_y'], geometry=geometry)
                heights = (aperture_contours(raw, target, upper) if x == lip else
                           [capture_upper(raw, target*f) for f in (.25, .5, .75, 1.)])
                cap = heights[-1]
                yy, flux = np.asarray(raw['y']), np.asarray(raw['Momentum_x'])
                take = yy < cap
                yy2 = np.r_[yy[take], cap]
                ff2 = np.r_[flux[take], np.interp(cap, yy, flux)]
                mass = np.sum((ff2[:-1]+ff2[1:])*np.diff(yy2)/2)
                max_error = max(max_error, abs(mass/target-1))
                rows.append(dict(case=label, x_m=float(x), lower_y_m=lo, upper_y_m=cap,
                    quarter_mass_y_m=heights[0], half_mass_y_m=heights[1],
                    three_quarter_mass_y_m=heights[2], target_massflow_kg_s_m=target))
            require(max_error < 1e-8, 'Streamtube boundary flux differs from aperture reference')
            lows = np.array([r['lower_y_m'] for r in rows])
            tops = np.array([r['upper_y_m'] for r in rows])
            ax.fill_between(xs, lows*1000, tops*1000, color=color, alpha=.25)
            ax.plot(xs, tops*1000, color=color, lw=1.8)
            for key in ('quarter_mass_y_m', 'half_mass_y_m', 'three_quarter_mass_y_m'):
                ax.plot(xs, np.array([r[key] for r in rows])*1000, color=color, lw=.7, alpha=.6)
            downstream = np.linspace(lip, settings['outlet_x']-.01, 80)
            bounds = np.array([section_bounds(points, markers, settings, float(x))[:2] for x in downstream])
            ax.fill_between(downstream, bounds[:, 0]*1000, bounds[:, 1]*1000, color=color, alpha=.25)
            write_csv(output/label/'capture-boundary.csv', rows)
            record = dict(case=label, status='available', samples=len(rows), reference='full physical aperture at exact lip x',
                lip_massflow_kg_s_m=target, isolator_massflow_kg_s_m=item['captured_reference']['integral']['massflow'],
                lip_reverse_massflow_fraction=aperture['reverse_massflow_fraction'],
                max_inversion_relative_mass_error=float(max_error), boundary=rows)
        except ValueError as error:
            record = dict(case=label, status='unavailable', reason=str(error))
            ax.text(.5, .5, 'Capture reconstruction unavailable: '+str(error), transform=ax.transAxes,
                    ha='center', va='center', wrap=True)
        for name in ('lower', 'cowl'):
            edges = points[markers[name]].copy()
            edges[:, :, 1] *= 1000
            ax.add_collection(LineCollection(edges, colors='#25323d', linewidths=1.4))
        ax.plot(lip, upper*1000, 'o', color='#25323d', ms=4)
        ax.annotate('Cowl lip', xy=(lip, upper*1000), xytext=(lip-.2, upper*1250),
                    ha='center', arrowprops=dict(arrowstyle='->', color='#25323d'))
        ax.set(xlim=(-.04, settings['outlet_x']), ylim=(-15, settings['lip_y']*1450),
               xlabel='x [m]', ylabel='y [mm]',
               title=f"{label} | AOA {settings.get('aoa_deg', 0):+g} deg | lip flow {target:.2f} kg/s per m")
        ax.grid(alpha=.12)
        record['limitations'] = ['Steady discrete-field reconstruction, not particle trajectories; vertical scale enlarged.',
            'Captured streamtube includes both shocked and unshocked flow.',
            'Lip flux is an independent aperture integral; station averages upstream use the isolator reference.',
            'At the lip, the physical upper endpoint is used while the signed tip flux is retained.',
            'Interior curves mark cumulative mass fractions 25/50/75 percent upstream; downstream fill is the full channel.',
            'A reverse-flow or insufficient-capacity upstream section makes this reconstruction unavailable.']
        write_json(output/label/'capture-boundary.json', record)
        records.append({k: v for k, v in record.items() if k != 'boundary'})
    fig.suptitle('Captured streamtube | colored: lip-aperture flow | thin curves: cumulative mass fractions', fontsize=12)
    fig.savefig(output/'captured-streamtube.png', dpi=165)
    fig.savefig(output/'captured-streamtube.svg')
    plt.close(fig)
    return records


def ideal_stages(settings):
    ref = references(settings)
    state = dict(mach=settings['mach'], pressure=ref['pressure'], temperature=ref['temperature'],
                 density=ref['density'], total_pressure_recovery=1.)
    angle, result = settings.get('aoa_deg', 0.), []
    for number, ramp in enumerate(ramps(settings), 1):
        try:
            estimate = shock(state['mach'], ramp['angle_deg']-angle, settings['gamma'])
        except ValueError as error:
            result.append(dict(stage=number, status='unavailable', reason=str(error)))
            break
        state = dict(mach=estimate['mach_downstream'], pressure=state['pressure']*estimate['pressure_ratio'],
                     temperature=state['temperature']*estimate['pressure_ratio']/estimate['density_ratio'],
                     density=state['density']*estimate['density_ratio'],
                     total_pressure_recovery=state['total_pressure_recovery']*estimate['total_pressure_ratio'])
        result.append(dict(stage=number, status='isolated_weak_shock_estimate', **state,
                           pressure_ratio=state['pressure']/ref['pressure'], temperature_ratio=state['temperature']/ref['temperature'],
                           density_ratio=state['density']/ref['density'], flow_angle_deg=ramp['angle_deg']))
        angle = ramp['angle_deg']
    return result


def validate_source(case, result, settings):
    stage = result/'second'
    record = json.loads((stage/'run.json').read_text())
    summary = json.loads((result/'summary.json').read_text()) if (result/'summary.json').exists() else {}
    require(record['returncode'] == 0, 'Source stage did not exit successfully')
    for field, path in [('mesh_sha256', case/'mesh.su2'), ('settings_sha256', case/'settings.json'),
                        ('config_sha256', stage/'case.cfg')]:
        require(record[field] == digest(path), f'Source {field} differs')
    provenance = 'standard_physics_signature'
    audit_path = None
    if record['physics_sha256'] != physics_signature(settings):
        reference = summary.get('provenance_audit', {})
        require(bool(reference), 'Source physics signature differs without an independent audit')
        audit_path = (result/reference['path']).resolve()
        require(audit_path.parent == result.resolve() and digest(audit_path) == reference['sha256'], 'Source audit differs')
        audit = json.loads(audit_path.read_text())
        matched = [item for item in audit.get('stages', []) if item['stage'] == 'second']
        require(audit.get('passed') is True and len(matched) == 1
                and matched[0]['original_receipt_sha256'] == digest(stage/'run.json')
                and matched[0]['stored_signature'] == record['physics_sha256']
                and all(matched[0]['checks'].values()), 'Independent source audit does not cover this receipt')
        provenance = 'independently_audited_physics_signature'
    config = read_config(stage/'case.cfg')
    require(config['SOLVER'] in ('EULER', 'RANS') and config['FLUID_MODEL'] == 'IDEAL_GAS',
            'Only constant-gamma ideal-gas Euler/RANS fields are supported')
    require(config['SOLVER'] == ('RANS' if is_rans(settings) else 'EULER'), 'Source solver model differs')
    for field, key in [('MACH_NUMBER', 'mach'), ('GAMMA_VALUE', 'gamma'), ('GAS_CONSTANT', 'gas_constant'),
                       ('FREESTREAM_PRESSURE', 'pressure'), ('FREESTREAM_TEMPERATURE', 'temperature')]:
        require(float(config[field]) == settings[key], f'Source {field} differs')
    angle = settings.get('aoa_deg', 0.)
    require(float(config['AOA']) == angle, 'Source incidence differs')
    inlet = [v.strip() for v in config['MARKER_SUPERSONIC_INLET'].strip('( )').split(',')]
    speed = references(settings)['speed']
    require(inlet[0] == 'inlet' and np.allclose([float(v) for v in inlet[1:]],
        [settings['temperature'], settings['pressure'], speed*math.cos(math.radians(angle)),
         speed*math.sin(math.radians(angle)), 0.], rtol=1e-12, atol=1e-12), 'Source inlet state differs')
    paths = [case/'mesh.su2', case/'settings.json', stage/'case.cfg', stage/'run.json', stage/'restart_flow.csv']
    if audit_path:
        paths.append(audit_path)
    if (result/'summary.json').exists():
        paths.append(result/'summary.json')
    return dict(case=str(case.resolve()), result=str(result.resolve()), provenance=provenance,
                sha256={str(path.resolve()): digest(path) for path in paths},
                source_solver_checks=summary.get('checks', {}),
                source_solver_checks_passed=bool(summary.get('checks')) and all(summary['checks'].values()),
                solver_binary_sha256=record['binary_sha256'],
                requested_iterations=int(config['ITER']), physical_validation=False, grid_independence=False)


def profile_rows(case_id, station, region, profile):
    meta = dict(case=case_id, station=station['id'], kind=station['kind'], ramp_stage=station.get('ramp_stage'),
                region=region, x_m=station['x'], lower_m=profile['y'][0], upper_m=profile['y'][-1])
    return [meta | {key: values[i] for key, values in profile.items()} for i in range(len(profile['y']))]


def integral_row(case_id, station, region, integral, profile, settings):
    row = dict(case=case_id, station=station['id'], kind=station['kind'], ramp_stage=station.get('ramp_stage'),
               region=region, x_m=station['x'], lower_m=profile['y'][0], upper_m=profile['y'][-1],
               profile_intervals=len(profile['y'])-1, includes_cowl_shock=station['x'] > settings['ramp_length'])
    row.update({key: value for key, value in integral.items() if key != 'forward_massflow_averages'})
    average, ref = integral['forward_massflow_averages'], references(settings)
    row.update({'average_'+key: value for key, value in average.items()})
    row.update(average_pressure_ratio=average['pressure']/ref['pressure'],
               average_temperature_ratio=average['temperature']/ref['temperature'],
               average_density_ratio=average['density']/ref['density'],
               average_total_pressure_recovery=average['total_pressure']/ref['total_pressure'],
               mean_velocity_angle_deg=math.degrees(math.atan2(average['v'], average['u'])))
    return row


def wall_profiles(points, markers, data, settings):
    flow = primitives(data, settings)
    wall_nodes = np.unique(markers['lower'])
    vertices = [(0., 0.), *ramp_vertices(settings)]
    result = []
    for number, (a, b) in enumerate(zip(vertices, vertices[1:]), 1):
        theta = math.atan2(b[1]-a[1], b[0]-a[0])
        nodes = wall_nodes[(points[wall_nodes, 0] >= a[0]-1e-12) & (points[wall_nodes, 0] <= b[0]+1e-12)]
        nodes = nodes[np.argsort(points[nodes, 0])]
        require(len(nodes) >= 2, 'Ramp wall has too few nodes')
        expected = a[1]+(points[nodes, 0]-a[0])*math.tan(theta)
        require(np.allclose(points[nodes, 1], expected, rtol=0, atol=1e-10), 'Wall nodes do not follow ramp geometry')
        profile = dict(y=points[nodes, 1].tolist(), **{key: values[nodes].tolist() for key, values in flow.items()})
        # Wall samples use along-ramp position; transverse eta is not meaningful here.
        ref = references(settings)
        profile.update(x=points[nodes, 0].tolist(), PointID=nodes.tolist(),
                       ramp_fraction=((points[nodes, 0]-a[0])/(b[0]-a[0])).tolist(),
                       distance_along_ramp_m=((points[nodes, 0]-a[0])/math.cos(theta)).tolist(),
                       pressure_ratio=(flow['pressure'][nodes]/ref['pressure']).tolist(),
                       temperature_ratio=(flow['temperature'][nodes]/ref['temperature']).tolist(),
                       density_ratio=(flow['density'][nodes]/ref['density']).tolist(),
                       total_pressure_recovery=(flow['total_pressure'][nodes]/ref['total_pressure']).tolist(),
                       total_temperature_ratio=(flow['total_temperature'][nodes]/ref['total_temperature']).tolist(),
                       flow_angle_deg=np.degrees(np.arctan2(flow['v'][nodes], flow['u'][nodes])).tolist(),
                       wall_tangent_velocity_m_s=(flow['u'][nodes]*math.cos(theta)+flow['v'][nodes]*math.sin(theta)).tolist(),
                       wall_normal_velocity_m_s=(-flow['u'][nodes]*math.sin(theta)+flow['v'][nodes]*math.cos(theta)).tolist())
        result.append(dict(stage=number, start=list(a), end=list(b), profile=profile))
    return result


def extract_case(case, result, case_id, output, fractions, spacing, offset, lip_offset):
    settings = json.loads((case/'settings.json').read_text())
    require(settings['kind'] == 'intake', 'Profiles require an intake case')
    source = validate_source(case, result, settings)
    points, cells, markers = read_su2(case/'mesh.su2')
    data = read_csv(result/'second/restart_flow.csv')
    ids = data['PointID'].astype(int)
    require(np.array_equal(np.sort(ids), np.arange(len(points))), 'Restart point IDs differ')
    data = {key: values[np.argsort(ids)] for key, values in data.items()}
    require(np.allclose(np.column_stack((data['x'], data['y'])), points, rtol=0, atol=1e-12), 'Restart coordinates differ')
    primitives(data, settings)
    geometry = SectionGeometry(points, cells)
    plan = stations(settings, fractions, spacing, offset, lip_offset)
    reference_x = (ramp_vertices(settings)[-1][0]+settings['outlet_x'])/2
    reference_lower = boundary_y(points, markers, 'lower', reference_x)
    reference_upper = boundary_y(points, markers, 'cowl', reference_x)
    reference_integral, _ = section(points, cells, data, settings, reference_x, reference_lower, reference_upper, geometry=geometry)
    captured_mass = reference_integral['massflow']
    require(captured_mass > 0, 'No net forward isolator flow')
    records, profiles_csv, integrals_csv, stations_csv = [], [], [], []
    for station in plan:
        x = station['x']
        lower, upper, internal = section_bounds(points, markers, settings, x)
        integral, raw = section(points, cells, data, settings, x, lower, upper, geometry=geometry)
        region = 'internal' if internal else 'external_full'
        profiles = {region: derived_profile(raw, settings)}
        integrals = {region: integral}
        primary = region
        capture_status = 'internal_channel' if internal else 'pending'
        if not internal:
            try:
                captured_upper = capture_upper(raw, captured_mass)
                captured_integral, captured_raw = section(points, cells, data, settings, x, lower, captured_upper, geometry=geometry)
                profiles['captured'] = derived_profile(captured_raw, settings)
                integrals['captured'] = captured_integral
                primary = 'captured'
                capture_error = abs(captured_integral['massflow']/captured_mass-1)
                require(capture_error < 1e-8, 'Captured streamtube mass does not match reference')
                capture_status = 'mass_matched_to_isolator_reference'
            except ValueError as error:
                primary = region
                profiles.pop('captured', None)
                integrals.pop('captured', None)
                capture_status = 'unavailable: '+str(error)
        chosen = profiles[primary]
        coarse, _ = section(points, cells, data, settings, x, chosen['y'][0], chosen['y'][-1], order=3, geometry=geometry)
        quadrature = {key: abs(integrals[primary][key]-coarse[key])/max(abs(integrals[primary][key]), 1e-30)
                      for key in ('massflow', 'convective_energyflow', 'axial_momentum_flux')}
        entry = dict(station=station, section_normal=[1, 0], profiles=profiles, integrals=integrals,
                     primary_region=primary, capture_status=capture_status,
                     quadrature_relative_difference=quadrature, gas_model='constant_gamma_ideal_gas', diagnostics_only=True)
        records.append(entry)
        for scope, profile in profiles.items():
            profiles_csv += profile_rows(case_id, station, scope, profile)
            integrals_csv.append(integral_row(case_id, station, scope, integrals[scope], profile, settings))
        stations_csv.append(dict(case=case_id, station=station['id'], kind=station['kind'], ramp_stage=station.get('ramp_stage'),
            ramp_fraction=station.get('fraction'), x_m=x, lower_m=lower, full_upper_m=upper,
            primary_upper_m=chosen['y'][-1], primary_region=primary, capture_status=capture_status,
            isolator_distance_m=station.get('isolator_distance')))
    walls = wall_profiles(points, markers, data, settings)
    wall_csv = [dict(case=case_id, ramp_stage=wall['stage']) | {key: values[i] for key, values in wall['profile'].items()}
                for wall in walls for i in range(len(wall['profile']['x']))]
    theory = ideal_stages(settings)
    compression_rows = []
    for wall in walls:
        number = wall['stage']
        candidates = [entry for entry in records if entry['station'].get('ramp_stage') == number
                      and entry['station']['x'] < settings['ramp_length']]
        require(bool(candidates), 'No pre-cowl assessment station for ramp')
        chosen = max(candidates, key=lambda entry: entry['station']['x'])
        average = chosen['integrals'][chosen['primary_region']]['forward_massflow_averages']
        ref = references(settings)
        ideal = next((row for row in theory if row['stage'] == number and row['status'] != 'unavailable'), None)
        compression_rows.append(dict(case=case_id, ramp_stage=number, assessment_station=chosen['station']['id'],
            x_m=chosen['station']['x'], averaging_region=chosen['primary_region'],
            cfd_average_mach=average['mach'], cfd_average_pressure_ratio=average['pressure']/ref['pressure'],
            cfd_average_temperature_ratio=average['temperature']/ref['temperature'],
            cfd_average_density_ratio=average['density']/ref['density'],
            cfd_average_total_pressure_recovery=average['total_pressure']/ref['total_pressure'],
            isolated_theory_mach=ideal['mach'] if ideal else None,
            isolated_theory_pressure_ratio=ideal['pressure_ratio'] if ideal else None,
            isolated_theory_temperature_ratio=ideal['temperature_ratio'] if ideal else None,
            isolated_theory_total_pressure_recovery=ideal['total_pressure_recovery'] if ideal else None,
            theory_scope='uniform post-stage state from isolated shocks; CFD is a nonuniform streamtube average'))
    extracted = dict(case=case_id, settings=settings, source=source, stations=records, wall_profiles=walls,
        ideal_stages=theory, ramp_compression=compression_rows, isolator_start_x=ramp_vertices(settings)[-1][0],
        captured_reference=dict(x=reference_x, lower=reference_lower, upper=reference_upper, integral=reference_integral),
        units='SI unless named mm or deg; mass and fluxes per metre span',
        method='Linear conservative interpolation at mesh-edge intersection knots; five-point Gauss per interval; no uniform resampling',
        limitations=['Postprocessed flux differs from SU2 dual-face flux.',
            'Captured streamtube is a steady 2D inviscid diagnostic anchored on the solid lower wall; viscous transport and separated flow require additional review.',
            'External-full averages include freestream and must not be read as post-ramp core states.',
            'External captured-section mass flow is imposed by construction, not an independent conservation check.',
            'R4 sections and wall samples beyond the cowl lip include cowl-shock interaction.',
            'C-lip spans the physical aperture at the exact lip x; its sharp-tip endpoint retains the saved nodal state and has zero measure in the flux integral.',
            'Isolated-shock theory is a reference, not a CFD validation target for the entire captured streamtube.',
            'No grid, boundary, physical-validation or natural-start certification is provided.'])
    target = output/case_id
    target.mkdir()
    write_json(target/'profiles.json', extracted)
    write_csv(target/'profiles.csv', profiles_csv)
    write_csv(target/'section-integrals.csv', integrals_csv)
    write_csv(target/'stations.csv', stations_csv)
    write_csv(target/'ramp-wall.csv', wall_csv)
    write_csv(target/'ramp-compression.csv', compression_rows)
    print(json.dumps(dict(case=case_id, recorded_stations=len(records), profile_rows=len(profiles_csv),
        wall_rows=len(wall_csv), captured_massflow=captured_mass,
        capture_unavailable=[entry['station']['id'] for entry in records if entry['capture_status'].startswith('unavailable')])), flush=True)
    return extracted, points, markers, geometry


def plot_results(all_cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator
    colors = plt.get_cmap('tab10').colors
    figures = output/'plots'
    figures.mkdir()
    variables = [('mach', 'Mach number'), ('pressure_ratio', 'p / p infinity'), ('temperature_ratio', 'T / T infinity'),
                 ('density_ratio', 'rho / rho infinity'), ('u_over_uinf', 'u / U infinity'), ('v_over_uinf', 'v / U infinity'),
                 ('flow_angle_deg', 'Flow angle [deg]'), ('total_pressure_recovery', 'p0 / p0 infinity'),
                 ('total_temperature_ratio', 'Total-temperature deviation [%]')]
    station_ids = list(dict.fromkeys(entry['station']['id'] for item, *_ in all_cases for entry in item['stations']))
    for station_id in station_ids:
        fig, axes = plt.subplots(3, 3, figsize=(11.8, 10.0), layout='constrained')
        for index, (item, *_) in enumerate(all_cases):
            matched = [entry for entry in item['stations'] if entry['station']['id'] == station_id]
            if not matched:
                continue
            entry = matched[0]
            profile = entry['profiles'][entry['primary_region']]
            label = f"{item['case']} | AOA {item['settings'].get('aoa_deg', 0):+g} deg | x={entry['station']['x']:.4f} m"
            for ax, (key, name) in zip(axes.flat, variables):
                values = np.asarray(profile[key])
                if key == 'total_temperature_ratio':
                    values = 100*(values-1)
                ax.plot(values, profile['eta'], color=colors[index % len(colors)], label=label, lw=1.4)
                ax.set(xlabel=name, ylabel='Normalized section height eta', ylim=(0, 1))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 4), useOffset=False)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{station_id}: transverse profiles | captured streamtube externally, full channel internally', fontsize=12)
        fig.savefig(figures/f'{station_id}.png', dpi=155)
        plt.close(fig)

    stages = max(len(item['wall_profiles']) for item, *_ in all_cases)
    fig, axes = plt.subplots(4, stages, figsize=(3.6*stages, 10.4), squeeze=False, layout='constrained')
    wall_fields = [('pressure_ratio', 'p / p infinity'), ('temperature_ratio', 'T / T infinity'),
                   ('mach', 'Mach number'), ('total_pressure_recovery', 'p0 / p0 infinity')]
    for index, (item, *_) in enumerate(all_cases):
        for wall in item['wall_profiles']:
            column = wall['stage']-1
            profile = wall['profile']
            ideal = next((v for v in item['ideal_stages'] if v['stage'] == wall['stage'] and v['status'] != 'unavailable'), None)
            for row, (key, label) in enumerate(wall_fields):
                ax = axes[row, column]
                ax.plot(profile['ramp_fraction'], profile[key], color=colors[index % len(colors)], lw=1.2, label=item['case'])
                if ideal:
                    ax.axhline(ideal[key], color=colors[index % len(colors)], ls='--', lw=.9)
                if wall['start'][0] < item['settings']['ramp_length'] < wall['end'][0]:
                    fraction = (item['settings']['ramp_length']-wall['start'][0])/(wall['end'][0]-wall['start'][0])
                    ax.axvline(fraction, color='#888', ls=':', lw=.8)
                ax.set(xlabel='Along-ramp fraction', ylabel=label, xlim=(0, 1))
                ax.grid(alpha=.2)
                if row == 0:
                    ax.set_title(f'Ramp {wall["stage"]}')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Ramp-wall distributions | dashed: isolated-shock theory | dotted vertical: cowl lip', fontsize=12)
    fig.savefig(output/'ramp-wall-profiles.png', dpi=165)
    # The fourth ramp extends inside the cowl; retain the full plot above and
    # add a separate pre-cowl view so its pressure rise does not hide compression.
    for row in range(len(wall_fields)):
        for column in range(stages):
            axes[row, column].cla()
    for index, (item, *_) in enumerate(all_cases):
        for wall in item['wall_profiles']:
            column = wall['stage']-1
            profile = wall['profile']
            keep = np.asarray(profile['x']) <= item['settings']['ramp_length']-.005
            ideal = next((v for v in item['ideal_stages'] if v['stage'] == wall['stage'] and v['status'] != 'unavailable'), None)
            for row, (key, label) in enumerate(wall_fields):
                ax = axes[row, column]
                ax.plot(np.asarray(profile['ramp_fraction'])[keep], np.asarray(profile[key])[keep],
                        color=colors[index % len(colors)], lw=1.2, label=item['case'])
                if ideal:
                    ax.axhline(ideal[key], color=colors[index % len(colors)], ls='--', lw=.9)
                ax.set(xlabel='Along-ramp fraction', ylabel=label)
                ax.grid(alpha=.2)
                if row == 0:
                    ax.set_title(f'Ramp {wall["stage"]}')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Pre-cowl ramp-wall compression | last 5 mm before lip excluded | dashed: isolated-shock theory', fontsize=12)
    fig.savefig(output/'ramp-wall-pre-cowl.png', dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout='constrained')
    metrics = [('mach', 'Mass-flux-weighted Mach'), ('pressure', 'Mass-flux-weighted pressure [kPa]'),
               ('temperature', 'Mass-flux-weighted temperature [K]'), ('density', 'Mass-flux-weighted density [kg/m3]'),
               ('total_pressure', 'Mass-flux-weighted p0 / p0 infinity'), ('massflow', 'Section mass flow [kg/s per m]')]
    for index, (item, *_) in enumerate(all_cases):
        x = [entry['station']['x'] for entry in item['stations']]
        for ax, (key, label) in zip(axes.flat, metrics):
            values = [entry['integrals'][entry['primary_region']]['massflow'] if key == 'massflow'
                      else entry['integrals'][entry['primary_region']]['forward_massflow_averages'][key] for entry in item['stations']]
            if key == 'pressure':
                values = np.asarray(values)/1000
            elif key == 'total_pressure':
                values = np.asarray(values)/references(item['settings'])['total_pressure']
            ax.plot(x, values, '-o', color=colors[index % len(colors)], ms=3, lw=1.1, label=item['case'])
            ax.axvline(item['settings']['ramp_length'], color='#888', ls=':', lw=.8)
            ax.axvline(item['isolator_start_x'], color='#888', ls='--', lw=.8)
            ax.set(xlabel='Section x [m]', ylabel=label)
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    axes[1, 2].text(.03, .08, 'External captured mass flow is matched by construction',
                   transform=axes[1, 2].transAxes, fontsize=8)
    fig.suptitle('Streamwise compression and loss | dotted: cowl lip | dashed: isolator start', fontsize=12)
    fig.savefig(output/'streamwise-metrics.png', dpi=165)
    plt.close(fig)

    for item, points, markers, geometry in all_cases:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), gridspec_kw={'height_ratios': [1.5, 1]}, layout='constrained')
        flow = read_csv(Path(item['source']['result'])/'second/restart_flow.csv')
        ids = np.argsort(flow['PointID'])
        flow = primitives({key: value[ids] for key, value in flow.items()}, item['settings'])
        axes[0].tripcolor(geometry.tri, flow['mach'], cmap='Greys_r', shading='gouraud', vmin=0, vmax=item['settings']['mach'], alpha=.32)
        kind_colors = dict(ramp='#216d9f', cowl='#a34b81', isolator='#238059')
        for ax in axes:
            for name in ('lower', 'cowl'):
                ax.add_collection(LineCollection(points[markers[name]], colors='#25323d', linewidths=1.2))
        for entry in item['stations']:
            station, profile = entry['station'], entry['profiles'][entry['primary_region']]
            x, color = station['x'], kind_colors[station['kind']]
            axes[0].plot([x, x], [profile['y'][0], profile['y'][-1]], color=color, lw=1.4)
            if entry['primary_region'] == 'captured':
                full = entry['profiles']['external_full']
                axes[0].plot([x, x], [profile['y'][-1], full['y'][-1]], color=color, lw=.55, ls=':')
            label_x = x+(-.06 if station['id'] == 'C-pre' else .035 if station['id'] == 'C-post' else 0.)
            label_y = 1.015*item['settings']['top_y']
            axes[0].text(label_x, label_y, station['id'], rotation=90, ha='center', va='bottom', fontsize=9, color=color)
            if label_x != x:
                axes[0].plot([x, label_x], [item['settings']['top_y'], label_y], color=color, lw=.6)
            if x >= item['settings']['ramp_length']:
                axes[1].plot([x, x], [profile['y'][0], profile['y'][-1]], color=color, lw=1.3)
                if station['kind'] == 'cowl':
                    dx = 15 if station['id'] == 'C-lip' else 60
                    axes[1].annotate(station['id'], (x, item['settings']['lip_y']),
                        xytext=(dx, 18), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, color=color, arrowprops=dict(arrowstyle='-', color=color, lw=.6))
                else:
                    axes[1].text(x, item['settings']['lip_y']+.001, station['id'], ha='center', va='bottom', fontsize=9, color=color)
        axes[0].set(xlim=(-.05, item['settings']['outlet_x']+.02), ylim=(-.025, item['settings']['top_y']*1.17),
                    xlabel='x [m]', ylabel='y [m]', aspect='equal', title=f"{item['case']}: section locations | solid: recorded primary region | dotted: external full domain")
        lower = min(entry['profiles'][entry['primary_region']]['y'][0] for entry in item['stations']
                    if entry['station']['x'] >= item['settings']['ramp_length'])
        height = item['settings']['lip_y']-lower
        axes[1].set(xlim=(item['settings']['ramp_length']-.01, item['settings']['outlet_x']+.01),
                    ylim=(lower-.15*height, item['settings']['lip_y']+.35*height),
                    xlabel='x [m]', ylabel='y [m]', title='Internal-channel stations (vertical scale enlarged)')
        for ax in axes:
            ax.grid(alpha=.15)
        fig.savefig(output/f'inspection-positions-{item["case"]}.png', dpi=175)
        plt.close(fig)


def run(case_paths, result_paths, labels, output, fractions=(.5, .95), spacing=.1, offset=.05, lip_offset=.005, streamtube_samples=161):
    require(len(case_paths) == len(result_paths) == len(labels) and len(set(labels)) == len(labels), 'Case/result labels differ or repeat')
    require(all(Path(label).name == label and label not in ('', '.', '..') for label in labels), 'Invalid case label')
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output/'source-snapshot'
    snapshots.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, snapshots/path.name)
    manifest = dict(status='running', cases=labels, parameters=dict(ramp_fractions=list(fractions),
        isolator_spacing_m=spacing, isolator_first_offset_m=offset, cowl_offset_m=lip_offset,
        streamtube_samples=streamtube_samples),
        analyzer_sha256=digest(__file__), source_snapshot_sha256={p.name: digest(p) for p in snapshots.glob('*.py')},
        units='SI; kg/s per metre span; pressures absolute', physical_validation=False, grid_independence=False,
        field_retention='No solver fields or existing evidence are modified or deleted.',
        executed_command=[sys.executable, *sys.argv])
    write_json(output/'manifest.json', manifest)
    try:
        all_cases = [extract_case(Path(case), Path(result), label, output, fractions, spacing, offset, lip_offset)
                     for case, result, label in zip(case_paths, result_paths, labels)]
        plot_results(all_cases, output)
        streamtubes = record_streamtubes(all_cases, output, streamtube_samples)
        sources_unchanged = all(digest(path) == expected for item, *_ in all_cases
                                for path, expected in item['source']['sha256'].items())
        require(sources_unchanged, 'Source files changed during analysis')
        manifest.update(status='completed', sources_unchanged=sources_unchanged, streamtubes=streamtubes,
            recorded_cases=[dict(case=item['case'], stations=len(item['stations']),
                source_solver_checks_passed=item['source']['source_solver_checks_passed'],
                max_quadrature_relative_difference=max(max(v['quadrature_relative_difference'].values()) for v in item['stations']),
                all_external_capture_regions_available=all(not v['capture_status'].startswith('unavailable') for v in item['stations']),
                min_primary_intervals=min(len(v['profiles'][v['primary_region']]['y'])-1 for v in item['stations'])) for item, *_ in all_cases],
            artifacts_sha256={str(path.relative_to(output)): digest(path) for path in output.rglob('*')
                              if path.is_file() and 'source-snapshot' not in path.parts and path.name != 'manifest.json'})
        write_json(output/'manifest.json', manifest)
        return manifest
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        write_json(output/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', type=Path, required=True)
    parser.add_argument('--result', action='append', type=Path, required=True)
    parser.add_argument('--label', action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ramp-fractions', nargs='+', type=float, default=[.5, .95])
    parser.add_argument('--isolator-spacing', type=float, default=.1)
    parser.add_argument('--isolator-first-offset', type=float, default=.05)
    parser.add_argument('--cowl-offset', type=float, default=.005)
    parser.add_argument('--streamtube-samples', type=int, default=161)
    args = parser.parse_args()
    manifest = run(args.case, args.result, args.label or [case.name for case in args.case], args.output,
                   args.ramp_fractions, args.isolator_spacing, args.isolator_first_offset, args.cowl_offset, args.streamtube_samples)
    print(json.dumps(dict(status=manifest['status'], recorded_cases=manifest['recorded_cases'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
