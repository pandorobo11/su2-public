#!/usr/bin/env python3
"""Compare interior sections across a direct chain of same-settings RANS runs.

Two consecutive small changes are a provisional plateau check, not proof of
iterative convergence, boundary independence, or mesh independence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from intake_adapt import solution, digest
from intake_euler import primitives, require, read_csv
from intake_section import section


def profile_changes(reference, current, settings):
    """Integrate squared differences on the union of profile knots, using SI."""
    a, b = np.array(reference['y']), np.array(current['y'])
    require(np.all(np.diff(a) > 0) and np.all(np.diff(b) > 0), 'Unordered section coordinates')
    require(np.allclose(a[[0,-1]], b[[0,-1]], rtol=0, atol=1e-12), 'Section limits differ')
    knots = np.unique(np.r_[a, b])
    z, w = np.polynomial.legendre.leggauss(5)
    half = np.diff(knots)/2
    y = ((knots[:-1]+knots[1:])[:, None]/2+half[:, None]*z).ravel()
    weights = (half[:, None]*w).ravel()
    keys = ['Density', 'Momentum_x', 'Momentum_y', 'Energy']
    if settings.get('mode') == 'rans':
        keys += ['Nu_Tilde']
    fields = []
    for profile, coords in ((reference, a), (current, b)):
        data = {k:np.interp(y, coords, profile[k]) for k in keys}
        f = primitives(data, settings)
        if 'Nu_Tilde' in data:
            f['Nu_Tilde'] = data['Nu_Tilde']
        fields.append(f)
    ref, cur = fields
    flux_weight = weights*np.maximum(ref['density']*ref['u'], 0)
    require(flux_weight.sum() > 0, 'No forward reference flow')
    output = {}
    for key in ['density','u','v','pressure','temperature']+(['Nu_Tilde'] if 'Nu_Tilde' in ref else []):
        scale2 = ref['u']**2+ref['v']**2 if key in ('u','v') else ref[key]**2
        diff2 = (cur[key]-ref[key])**2
        norms = {}
        for name, weight in [('mass_weighted_l2', flux_weight), ('unweighted_l2', weights)]:
            numerator, denominator = float(np.sum(weight*diff2)), float(np.sum(weight*scale2))
            # Both identically zero is exact agreement, otherwise avoid a NaN
            # or an arbitrary dimensional floor concealing a newly nonzero field.
            norms[name] = np.sqrt(numerator/denominator) if denominator > 0 else (0. if numerator == 0 else None)
        output[key] = norms
    return output


def compare_sections(previous, current, settings, integral_tolerance=.001, profile_tolerance=.002):
    require(np.isfinite([integral_tolerance, profile_tolerance]).all()
            and integral_tolerance > 0 and profile_tolerance > 0, 'Invalid tolerances')
    integral = {}
    for k in ('massflow','convective_energyflow','axial_momentum_flux'):
        a, b = previous[0][k], current[0][k]
        integral[k] = abs(b-a)/abs(a) if a else (0. if b == 0 else None)
    profiles = profile_changes(previous[1], current[1], settings)
    passed = all(v is not None and v < integral_tolerance for v in integral.values())
    passed &= all(v is not None and v < profile_tolerance for q in profiles.values() for v in q.values())
    return dict(integral_relative_change=integral, profile_relative_change=profiles, passed=bool(passed))


def normalized_config(path):
    values = dict(line.split('= ', 1) for line in path.read_text().splitlines() if '= ' in line)
    require(int(values['LIMITER_ITER']) > int(values['ITER']), 'Limiter must remain live in every block')
    require(int(values.get('CONV_STARTITER', values['ITER'])) > int(values['ITER']),
            'Use fixed-length checkpoints with early convergence disabled')
    for k in ('ITER','OUTPUT_WRT_FREQ','LIMITER_ITER','CONV_STARTITER','RESTART_SOL','READ_BINARY_RESTART'):
        values.pop(k, None)
    return values


def check_completed_block(history, requested):
    iterations = history['Inner_Iter']
    require(len(iterations) > 0 and iterations[0] == 0
            and np.array_equal(iterations, np.arange(requested)),
            'Incomplete fixed-length checkpoint history')


def compare(case, results, x, integral_tolerance=.001, profile_tolerance=.002):
    require(len(results) >= 3, 'Need at least three checkpoints for two consecutive changes')
    require(len({r.resolve() for r in results}) == len(results), 'Repeated checkpoint')
    snapshots, provenance, cfg = [], [], None
    prior = None
    for result in results:
        s, p, c, m, data, flow, record = solution(case, result)
        require(s.get('mode') == 'rans' and s['kind'] == 'intake'
                and s['ramp_length'] < x < s['outlet_x'], 'Invalid RANS intake evaluation plane')
        require(record['second_order'], 'Requires second-order flow checkpoints')
        current_config = normalized_config(result/'second/case.cfg')
        check_completed_block(read_csv(result/'second/history.csv'), record['iterations'])
        require(cfg is None or cfg == current_config, 'Numerical settings differ')
        cfg = current_config
        if prior:
            require(Path(record['restart_from']).resolve() == (prior/'second').resolve(), 'Not a direct restart chain')
            require(record['binary_sha256'] == provenance[-1]['binary_sha256'], 'Solver binary differs')
        snapshots.append(section(p, c, data, s, x, s['ramp_y'], s['lip_y']))
        provenance.append(dict(result=str(result), iterations=record['iterations'],
            mesh_sha256=record['mesh_sha256'], binary_sha256=record['binary_sha256'],
            restart_sha256=digest(result/'second/restart_flow.csv'),
            config_sha256=record['config_sha256'], run_sha256=digest(result/'second/run.json')))
        prior = result
    changes = [compare_sections(a,b,s,integral_tolerance,profile_tolerance) for a,b in zip(snapshots,snapshots[1:])]
    return dict(x=x, tolerances=dict(integrals=integral_tolerance, profiles=profile_tolerance),
        last_two_changes_passed=all(v['passed'] for v in changes[-2:]),
        usable_as_combustor_inlet=False, checks=changes, source=provenance,
        note='Same-mesh, same-settings direct continuation only. A plateau is not proof of convergence. Inspect residuals, conservation, wall profiles and sufficient iteration separation; boundary and mesh sensitivity remain necessary. Velocity components use the reference speed L2 norm; zero reference with nonzero change is null and fails.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case', type=Path)
    ap.add_argument('results', type=Path, nargs='+')
    ap.add_argument('--x', type=float, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    report = compare(a.case, a.results, a.x)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
