#!/usr/bin/env python3
"""Experimental solution-guided, one-pass triangle refinement for 2-D intakes.

Pressure jumps plus compression identify candidates, not discretization error.
Wall quads and every physical boundary edge are protected. No solution transfer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from generate_airfoil_mesh import cell_blocks, read_su2
from intake_euler import (require, read_csv, physics_signature, primitives, polygon,
                          mesh_quality, wall_layer_quality)
from intake_mesh import (boundary_plan, slip_partition, partition_constraints,
                         termination_quality, transition_quality, policy_assessment)
from refine_airfoil_wake import split_triangle, write_su2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solution(case, result):
    """Validate source files and match restart rows before using a pilot solution."""
    s = json.loads((case/'settings.json').read_text())
    stage = result/'second'
    record = json.loads((stage/'run.json').read_text())
    require(record['returncode'] == 0 and 'Exit Success' in (stage/'solver.log').read_text(),
            'Source solver did not succeed')
    require(record['mesh_sha256'] == digest(case/'mesh.su2'), 'Mesh provenance differs')
    require(record['settings_sha256'] == digest(case/'settings.json'), 'Settings provenance differs')
    require(record['physics_sha256'] == physics_signature(s), 'Physics provenance differs')
    require(record['config_sha256'] == digest(stage/'case.cfg'), 'Configuration provenance differs')
    p, c, m = read_su2(case/'mesh.su2')
    data = read_csv(stage/'restart_flow.csv')
    ids = data['PointID']
    require(np.array_equal(np.sort(ids), np.arange(len(p))), 'Point IDs differ')
    data = {k: v[np.argsort(ids)] for k, v in data.items()}
    require(np.allclose(np.column_stack((data['x'], data['y'])), p, atol=1e-12, rtol=0),
            'Restart coordinates differ')
    flow = primitives(data, s)
    if s.get('mode') == 'rans':
        require('Nu_Tilde' in data and np.all(data['Nu_Tilde'] >= 0),
                'Invalid standard-SA source state')
    history = read_csv(stage/'history.csv')
    nonphysical = [v for k, v in history.items() if k.startswith('Nonphysical')]
    require(bool(nonphysical) and all(np.max(v) == 0 for v in nonphysical), 'Nonphysical source points')
    return s, p, c, m, data, flow, record


def sensors(points, cells, flow, pressure_jump=0.08, compression=0.02):
    """Scale-invariant cell candidates; h=sqrt(area), sound-speed normalization."""
    require(np.isfinite([pressure_jump, compression]).all() and pressure_jump > 0 and compression > 0,
            'Invalid sensor thresholds')
    result = {}
    for n, block in cell_blocks(cells).items():
        q = points[block]
        center = q.mean(axis=1)
        area = .5*np.sum(q[:, :, 0]*np.roll(q[:, :, 1], -1, axis=1)
                        -q[:, :, 1]*np.roll(q[:, :, 0], -1, axis=1), axis=1)
        require(np.all(area > 0), 'Nonpositive sensor cell')
        # Centered least squares is exact for linear fields on triangles/quads.
        dx = q-center[:, None, :]
        velocity = np.stack((flow['u'][block], flow['v'][block]), axis=-1)
        grad = np.linalg.solve(np.swapaxes(dx, 1, 2)@dx,
                               np.swapaxes(dx, 1, 2)@(velocity-velocity.mean(axis=1)[:, None, :]))
        div = grad[:, 0, 0]+grad[:, 1, 1]
        jump = np.ptp(flow['pressure'][block], axis=1)/flow['pressure'][block].mean(axis=1)
        comp = np.maximum(-div, 0)*np.sqrt(area)/flow['sound_speed'][block].mean(axis=1)
        result[n] = dict(center=center, pressure_jump=jump, compression=comp,
                         selected=(jump >= pressure_jump) & (comp >= compression))
    return result


def reverse_wall_cells(points, cells, markers, flow, reference_speed):
    """Find reverse mean tangential velocity in wall-adjacent quads, not Cf."""
    quads = cell_blocks(cells).get(4, np.empty((0, 4), int))
    edges = {tuple(sorted((int(a), int(b)))): i for i, q in enumerate(quads)
             for a, b in zip(q, np.roll(q, -1))}
    records = []
    for name in ('lower', 'cowl'):
        for edge in markers.get(name, []):
            i = edges.get(tuple(sorted(edge)))
            if i is None:
                continue
            tangent = points[edge[1]]-points[edge[0]]
            tangent /= np.linalg.norm(tangent)
            if tangent[0] < 0:
                tangent = -tangent
            q = quads[i]
            speed = float(np.mean(flow['u'][q])*tangent[0]+np.mean(flow['v'][q])*tangent[1])
            if speed < -1e-3*reference_speed:
                records.append(dict(marker=name, quad_index=i, midpoint_m=points[edge].mean(axis=0).tolist(),
                                    tangential_velocity=speed))
    return records


def refine_once(points, cells, markers, selected, max_nodes):
    """Bisect unprotected edges of selected triangles, conforming at neighbors."""
    blocks = cell_blocks(cells)
    triangles = blocks.get(3, np.empty((0, 3), int))
    quads = blocks.get(4, np.empty((0, 4), int))
    require(selected.shape == (len(triangles),) and selected.dtype == bool, 'Invalid triangle selection')
    fixed = {tuple(sorted(e)) for es in markers.values() for e in es}
    fixed.update(tuple(sorted(e)) for q in quads for e in zip(q, np.roll(q, -1)))
    proposed = {tuple(sorted(e)) for t in triangles[selected] for e in zip(t, np.roll(t, -1))}
    edges = sorted(proposed-fixed)
    require(len(points)+len(edges) <= max_nodes, 'Refinement exceeds node limit')
    require(bool(edges), 'No unprotected refinement edges')
    mids = {e: len(points)+i for i, e in enumerate(edges)}
    p = np.vstack((points, points[np.asarray(edges)].mean(axis=1)))
    tris = np.asarray([t for c in triangles for t in split_triangle(c, mids, p)], dtype=int)
    return p, {3: tris, 4: quads.copy()}, dict(added_nodes=len(edges), selected_triangles=int(selected.sum()),
        protected_candidate_edges=len(proposed & fixed), preserved_quads=len(quads), passes=1)


def adapt(case, result, output, pressure_jump=.08, compression=.02, max_nodes=500000):
    case, result, output = case.resolve(), result.resolve(), output.resolve()
    require(not output.exists(), 'Output already exists')
    s, p, c, m, data, flow, record = solution(case, result)
    require(s.get('mode') == 'rans' and 'mesh_policy' in s, 'Requires RANS with geometry mesh policy')
    found = sensors(p, c, flow, pressure_jump, compression)
    selected = found[3]['selected']
    speed = s['mach']*np.sqrt(s['gamma']*s['gas_constant']*s['temperature'])
    reverse = reverse_wall_cells(p, c, m, flow, speed)
    refined, blocks, counts = refine_once(p, c, m, selected, max_nodes)
    poly, names = polygon(s)
    part = None
    if s['mesh_policy']['name'] == 'geometry_v2':
        poly, names, part = slip_partition(poly, names, s['boundary_layer'], s['mesh_policy'])
    poly, names, constraints = boundary_plan(poly, names, s['boundary_layer'], skip_markers={'upstream'} if part else ())
    if part:
        constraints += partition_constraints(part, s['boundary_layer'], s['mesh_policy'])
    report = dict(provisional=True, method='one_pass_pressure_jump_and_compression',
        thresholds=dict(pressure_jump=pressure_jump, compression=compression), **counts,
        selected_quads=int(found[4]['selected'].sum()), reverse_wall_cells=reverse,
        source=dict(mesh_sha256=record['mesh_sha256'], settings_sha256=record['settings_sha256'],
                    physics_sha256=record['physics_sha256'], config_sha256=record['config_sha256'],
                    restart_sha256=digest(result/'second/restart_flow.csv'), run_sha256=digest(result/'second/run.json')),
        selected_triangle_centers_m=found[3]['center'][selected].tolist(),
        limitations='Feature detector, not an error estimator. Wall quads and physical edges remain fixed; wall-layer shock and reverse-flow candidates require separate wall refinement. No solution transfer or CFD acceptance.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.intake-adapt-', dir=output.parent) as temp:
        tmp = Path(temp)
        write_su2(tmp/'mesh.su2', refined, blocks, m)
        rp, rc, rm = read_su2(tmp/'mesh.su2')
        require(np.array_equal(rp[:len(p)], p), 'Original points changed')
        require(np.array_equal(cell_blocks(rc)[4], cell_blocks(c)[4]), 'Wall quads changed')
        require(set(rm) == set(m) and all(np.array_equal(rm[k], v) for k, v in m.items()), 'Markers changed')
        q = mesh_quality(rp, rc, rm, poly, names)
        q.update(mesh_sha256=digest(tmp/'mesh.su2'), boundary_layer=wall_layer_quality(rp, rc, rm, s),
                 layer_transition=transition_quality(rp, rc), mesh_policy=s['mesh_policy'],
                 layer_termination=termination_quality(rp, rm, constraints, s['boundary_layer']))
        q['policy_assessment'] = policy_assessment(q['layer_transition'], s['mesh_policy'])
        if part:
            q['internal_partition'] = part
        q['adaptation'] = report
        # Keep settings bytes and physics unchanged. input.json is deliberately
        # absent: generation alone cannot reproduce a solution-guided mesh.
        shutil.copy2(case/'settings.json', tmp/'settings.json')
        (tmp/'quality.json').write_text(json.dumps(q, indent=2, allow_nan=False)+'\n')
        (tmp/'adaptation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        output.mkdir()
        for f in tmp.iterdir():
            shutil.move(str(f), output/f.name)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case', type=Path)
    ap.add_argument('result', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--pressure-jump', type=float, default=.08)
    ap.add_argument('--compression', type=float, default=.02)
    ap.add_argument('--max-nodes', type=int, default=500000)
    a = ap.parse_args()
    q = adapt(a.case, a.result, a.output, a.pressure_jump, a.compression, a.max_nodes)
    print(json.dumps({k:q[k] for k in ('nodes','triangles','quadrilaterals','policy_assessment')}, indent=2))


if __name__ == '__main__':
    main()
