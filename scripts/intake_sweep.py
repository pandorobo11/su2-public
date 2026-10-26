#!/usr/bin/env python3
"""Mach x AoA Euler sweep of a multi-ramp intake, from one JSON file to tables and figures.

  prepare   design -> structured mesh -> per-case SU2 configs -> frozen plan (no CFD)
  submit    one PBS job per case plus one aggregation job (depend=afterany); or --backend local
  run-case  executed inside a job: first order, then second-order attempts with the
            lip-section gate and the automatic retry ladder, then per-case analysis
  aggregate CSV tables, figures and summary.md over all cases (safe to rerun)
  status    compact progress table
  compare   overlay the lip figures of several runs

Every run lives in its own new directory <output_root>/<name>/ and the plan is frozen
at prepare time (hashes of the mesh, configs, binary path and the script snapshot the
jobs execute). The model is not in the loop: stops, retries and progression are
decided here. Fields are never deleted during a run; restarts are binary and compact by
default, with no intermediate VTU (docs/su2-data-retention.md). See docs/intake-sweep-pbs.md.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SNAPSHOT = ('intake_sweep.py', 'intake_sweep_metrics.py', 'intake_sweep_theory.py', 'intake_sweep_plots.py',
            'intake_design.py', 'intake_structured_mesh.py', 'intake_euler.py', 'intake_mesh.py',
            'intake_section.py', 'intake_profiles.py', 'generate_airfoil_mesh.py', 'refine_airfoil_wake.py',
            'benchmark_su2.py')
LOCAL = ('p0_recovery', 'T_ratio', 'p_ratio', 'rho_ratio', 'mach')  # station averages get a _massavg suffix
LEGACY_OUTPUT = dict(restart_ascii=True, restart_compact=False, checkpoint_volume=True, adopted_volume=False)
REFERENCE_CAPTURE = 0.35  # capture height of the 2026-10-11/13 fine-quad mesh the resolution rule was fitted to

DEFAULTS = {
    'name': None,
    'output_root': 'runs',
    'design': dict(mach=None, temperature_ratio=None, stages=None, capture_height_m=None,
                   contraction_ratio=None, duct_gap_height_m=None, duct_length_m=None,
                   cowl_thickness_m=None, cowl_nose_length_m=None, upstream_length_m=None, gamma=1.4),
    'freestream': dict(temperature_K=220., pressure_Pa=2500., gas_constant=287.05),
    'sweep': dict(mach=None, aoa_deg=None),
    'mesh': dict(length_scale=None),
    'numerics': dict(convective='HLLC', venkat_coeff=0.05, linear_solver_iter=20, linear_solver_error=0.01),
    'stages': dict(first_iterations=1000, second_iterations=40000, checkpoint_period=2000, cfl=1.0, retry_cfl=0.5,
                   limiter_iter=1500, delayed_limiter_iter=10000, max_second_order_attempts=4,
                   max_stage_seconds=14400, case_budget_seconds=43200),
    'gate': dict(min_second_order_iterations=6000, residual_decades=5., imbalance=1e-3, lip_relative_change=1e-4),
    # restart_ascii/restart_compact: SU2 RESTART_ASCII and WRT_RESTART_COMPACT; checkpoint_volume: a VTU at every
    # checkpoint; adopted_volume: one VTU of the adopted checkpoint, written afterwards by SU2_CFD at ITER=0
    'output': dict(restart_ascii=False, restart_compact=True, checkpoint_volume=False, adopted_volume=True),
    'analysis': dict(pre_lip_offset_m=None, isolator_offset_m=None, isolator_spacing_m=None, outlet_inset_m=None,
                     profile_stations=['C-lip', 'I01', 'MID', 'OUT']),
    'execution': dict(backend='pbs', ranks=8, su2_cfd='SU2_CFD', mpirun=['mpirun', '-np', '{ranks}'],
                      python='python3', setup=[], storage_reserve_gb=20., local_exclusive=True,
                      pbs=dict(qsub='qsub', queue=None, resources=['select=1:ncpus=8:mpiprocs=8'],
                               job_prefix='isw', extra_directives=[], analysis_queue=None,
                               analysis_resources=['select=1:ncpus=1'])),
}


# ----------------------------------------------------------------------------- helpers

class Cancelled(Exception):
    """SIGTERM/SIGINT received (qdel, PBS preemption, Ctrl-C)."""


def _cancel(signum, frame):
    raise Cancelled(f'signal {signum}')

def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=_plain)+'\n')
    tmp.replace(path)


def _plain(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def load(path):
    return json.loads(Path(path).read_text())


def write_cfg(path, cfg):
    Path(path).write_text(''.join(f'{k}= {v}\n' for k, v in cfg.items()))


def read_cfg(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('%', 1)[0].strip()
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def merge(base, extra, where=''):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        require(k in base, f'Unknown setting {where}{k}')
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict) and k != 'mpirun':
            out[k] = merge(base[k], v, f'{where}{k}.')
        else:
            out[k] = v
    return out


def case_id(mach, aoa):
    m = f'{int(mach):02d}' if float(mach).is_integer() else f'{mach:g}'
    a = f'{aoa:+g}'
    return f'm{m}_a{a}'


def num(v):
    return v is not None and not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


# ----------------------------------------------------------------------------- configuration

def read_config(path):
    path = Path(path).resolve()
    raw = load(path)
    require(isinstance(raw, dict), 'Config must be a JSON object')
    raw = {k: v for k, v in raw.items() if not k.startswith('_')}
    cfg = merge(DEFAULTS, raw)
    d = cfg['design']
    require(isinstance(cfg['name'], str) and cfg['name'] and all(c.isalnum() or c in '-_.' for c in cfg['name']),
            'name must be a non-empty string of letters, digits, - _ .')
    for k in ('mach', 'temperature_ratio', 'capture_height_m', 'duct_length_m', 'cowl_thickness_m',
              'cowl_nose_length_m', 'upstream_length_m', 'gamma'):
        require(num(d[k]) and d[k] > 0, f'design.{k} must be a positive number')
    require(isinstance(d['stages'], int) and 1 <= d['stages'] <= 6, 'design.stages must be 1..6')
    require((d['contraction_ratio'] is None) != (d['duct_gap_height_m'] is None),
            'Give exactly one of design.contraction_ratio or design.duct_gap_height_m')
    require(d['contraction_ratio'] is None or (num(d['contraction_ratio']) and d['contraction_ratio'] >= 1),
            'design.contraction_ratio must be >= 1 (the duct may not be higher than the lip opening)')
    for k in ('mach', 'aoa_deg'):
        v = cfg['sweep'][k]
        require(isinstance(v, list) and v and all(num(x) for x in v) and len(set(v)) == len(v), f'sweep.{k} must be a list')
    require(all(m > 1 for m in cfg['sweep']['mach']), 'sweep Mach must exceed 1')
    ex = cfg['execution']
    require(ex['backend'] in ('pbs', 'local'), 'execution.backend must be pbs or local')
    require(isinstance(ex['ranks'], int) and ex['ranks'] >= 1, 'execution.ranks must be a positive integer')
    require(isinstance(ex['mpirun'], list) and ex['mpirun'], 'execution.mpirun must be a command list')
    require(all(isinstance(v, bool) for v in cfg['output'].values()), 'output settings must be true or false')
    scale = cfg['mesh']['length_scale']
    if scale is None:
        cfg['mesh']['length_scale'] = d['capture_height_m']/REFERENCE_CAPTURE
    a = cfg['analysis']
    ls = cfg['mesh']['length_scale']
    for k, v in (('pre_lip_offset_m', .005), ('isolator_offset_m', .05), ('isolator_spacing_m', .1), ('outlet_inset_m', .001)):
        if a[k] is None:
            a[k] = v*ls
    root = Path(cfg['output_root'])
    cfg['run_dir'] = str((root if root.is_absolute() else path.parent/root).resolve()/cfg['name'])
    cfg['config_path'] = str(path)
    return cfg


def output_settings(plan):
    """Plans frozen before 2026-10-26 have no output section and wrote the legacy files."""
    return plan.get('output', LEGACY_OUTPUT)


def restart_suffix(output):
    return '.csv' if output['restart_ascii'] else '.dat'


def cfg_template(s, phys, numerics, output):
    V = phys['mach']*math.sqrt(phys['gamma']*phys['gas_constant']*phys['temperature'])
    a = math.radians(phys['aoa_deg'])
    return {
        'SOLVER': 'EULER', 'MATH_PROBLEM': 'DIRECT', 'RESTART_SOL': 'NO', 'SYSTEM_MEASUREMENTS': 'SI',
        'FLUID_MODEL': 'IDEAL_GAS', 'GAMMA_VALUE': repr(phys['gamma']), 'GAS_CONSTANT': repr(phys['gas_constant']),
        'MACH_NUMBER': repr(float(phys['mach'])), 'AOA': repr(float(phys['aoa_deg'])),
        'FREESTREAM_PRESSURE': repr(phys['pressure']), 'FREESTREAM_TEMPERATURE': repr(phys['temperature']),
        'REF_DIMENSIONALIZATION': 'DIMENSIONAL', 'REF_LENGTH': repr(s['ramp_length']), 'REF_AREA': '1',
        'MARKER_EULER': '( lower, cowl )', 'MARKER_FAR': '( farfield )',
        'MARKER_SUPERSONIC_INLET': f"( inlet, {phys['temperature']!r}, {phys['pressure']!r}, {V*math.cos(a)!r}, {V*math.sin(a)!r}, 0 )",
        'MARKER_SUPERSONIC_OUTLET': '( outlet, external_outlet )',
        'MARKER_PLOTTING': '( lower, cowl )', 'MARKER_MONITORING': '( lower, cowl )',
        'MARKER_ANALYZE': '( inlet, outlet, farfield, external_outlet )', 'MARKER_ANALYZE_AVERAGE': 'MASSFLUX',
        'NUM_METHOD_GRAD': 'WEIGHTED_LEAST_SQUARES', 'NUM_METHOD_GRAD_RECON': 'WEIGHTED_LEAST_SQUARES',
        'CONV_NUM_METHOD_FLOW': numerics['convective'], 'MUSCL_FLOW': 'NO',
        'SLOPE_LIMITER_FLOW': 'VENKATAKRISHNAN', 'VENKAT_LIMITER_COEFF': repr(numerics['venkat_coeff']),
        'LIMITER_ITER': '1500', 'TIME_DISCRE_FLOW': 'EULER_IMPLICIT', 'CFL_NUMBER': '1.0', 'CFL_ADAPT': 'NO',
        'MGLEVEL': '0', 'LINEAR_SOLVER': 'FGMRES', 'LINEAR_SOLVER_PREC': 'ILU',
        'LINEAR_SOLVER_ERROR': repr(numerics['linear_solver_error']), 'LINEAR_SOLVER_ITER': str(numerics['linear_solver_iter']),
        'ITER': '1000', 'CONV_FIELD': '( RMS_DENSITY, RMS_ENERGY )', 'CONV_RESIDUAL_MINVAL': '-16', 'CONV_STARTITER': '1001',
        'MESH_FORMAT': 'SU2', 'MESH_FILENAME': 'mesh.su2', 'TABULAR_FORMAT': 'CSV', 'CONV_FILENAME': 'history',
        'HISTORY_OUTPUT': '( ITER, RMS_RES, FLOW_COEFF_SURF, NONPHYSICAL_POINTS )',
        'SCREEN_OUTPUT': '( INNER_ITER, RMS_DENSITY, RMS_ENERGY, NONPHYSICAL_POINTS )',
        'SCREEN_WRT_FREQ_INNER': '100', 'HISTORY_WRT_FREQ_INNER': '1',
        'OUTPUT_FILES': '( RESTART, RESTART_ASCII, PARAVIEW )' if output['restart_ascii'] else '( RESTART, PARAVIEW )',
        'OUTPUT_WRT_FREQ': '1000', 'RESTART_FILENAME': 'restart_flow', 'SOLUTION_FILENAME': 'seed_restart',
        'READ_BINARY_RESTART': 'NO' if output['restart_ascii'] else 'YES',
        'VOLUME_FILENAME': 'flow', 'SURFACE_FILENAME': 'surface_flow',
        'VOLUME_OUTPUT': '( COORDINATES, SOLUTION, PRIMITIVE, LIMITER )',
        'WRT_RESTART_OVERWRITE': 'NO', 'WRT_VOLUME_OVERWRITE': 'NO',
        'WRT_RESTART_COMPACT': 'YES' if output['restart_compact'] else 'NO',
    }


# ----------------------------------------------------------------------------- prepare

def design_geometry(cfg):
    sys.path.insert(0, str(HERE))
    import intake_design
    d = cfg['design']
    opts = dict(capture_height=d['capture_height_m'], gamma=d['gamma'], duct_length=d['duct_length_m'],
                cowl_thickness=d['cowl_thickness_m'], cowl_nose_length=d['cowl_nose_length_m'])
    base = intake_design.design(d['mach'], d['temperature_ratio'], d['stages'], **opts)
    gap = base['geometry']['gap_height_m']
    duct = d['duct_gap_height_m'] if d['duct_gap_height_m'] is not None else gap/d['contraction_ratio']
    require(duct <= gap+1e-12, f'Duct height {duct:.4f} m exceeds the lip opening {gap:.4f} m (contraction < 1)')
    result = intake_design.design(d['mach'], d['temperature_ratio'], d['stages'], duct_gap_height=duct, **opts)
    raw = intake_design.mesh_input(result, cfg['freestream']['temperature_K'], cfg['freestream']['pressure_Pa'])
    raw.update(upstream_length=d['upstream_length_m'], gas_constant=cfg['freestream']['gas_constant'])
    return result, raw, gap, duct


def build_mesh(raw, out, length_scale):
    from intake_euler import settings as resolve
    import intake_structured_mesh as ism
    from refine_airfoil_wake import write_su2
    from generate_airfoil_mesh import read_su2
    import numpy as np
    out.mkdir(parents=True)
    (out/'intake-input.json').write_text(json.dumps(dict(raw, output='.'), indent=2)+'\n')
    s = resolve(out/'intake-input.json')
    columns = ism.rule_columns(s, length_scale)
    points, quads, markers, meta = ism.mesh(s, columns, length_scale=length_scale)
    quality = ism.check(points, quads, markers, s)
    write_su2(out/'mesh.su2', points, {4: quads}, markers)
    rp, _, rm = read_su2(out/'mesh.su2')
    require(np.array_equal(points, rp) and all(np.array_equal(markers[k], rm[k]) for k in markers), 'Mesh round trip differs')
    s = dict(s, output=str(out), mesh_policy='structured_two_block_quad', column_rule='rule_columns', length_scale=length_scale)
    (out/'settings.json').write_text(json.dumps(s, indent=2)+'\n')
    dump(out/'quality.json', dict(quality, meta=meta))
    return s, quality


def theory_rows(cfg, s, gap, duct):
    from intake_euler import ramp_vertices
    import intake_sweep_theory as T
    verts = [(0., 0.)]+[tuple(v) for v in ramp_vertices(s)]
    corners, angles = verts[:-1], [r['angle_deg'] for r in s['ramps']]
    rows = []
    for M in cfg['sweep']['mach']:
        for a in cfg['sweep']['aoa_deg']:
            r = dict(case=case_id(M, a), mach=M, aoa_deg=a)
            try:
                r.update(T.lip_theory(M, a, corners, angles, (s['ramp_length'], s['lip_y']), cfg['design']['gamma']))
                r['geometric_contraction'] = gap/duct
                r['kantrowitz_margin'] = r['kantrowitz_max_contraction']/r['geometric_contraction']
            except ValueError as ex:
                r['theory_error'] = str(ex)
            rows.append(r)
    return rows


def prepare(config_path):
    cfg = read_config(config_path)
    run = Path(cfg['run_dir'])
    require(not run.exists(), f'Run directory already exists: {run} (choose a new name)')
    sys.path.insert(0, str(HERE))
    result, raw, gap, duct = design_geometry(cfg)
    run.mkdir(parents=True)
    try:
        shutil.copy2(config_path, run/'sweep.json')
        snap = run/'scripts'
        snap.mkdir()
        for n in SNAPSHOT:
            shutil.copy2(HERE/n, snap/n)
        dump(run/'design'/'design.json', result)
        try:
            import intake_design
            intake_design.plot(result, run/'design')
        except Exception as ex:  # plotting is optional provenance
            log(f'design plot skipped: {ex!r}')
        s, quality = build_mesh(raw, run/'mesh', cfg['mesh']['length_scale'])
        theory = theory_rows(cfg, s, gap, duct)
        write_csv(run/'analysis'/'theory.csv', theory)
        phys0 = dict(gamma=cfg['design']['gamma'], gas_constant=cfg['freestream']['gas_constant'],
                     temperature=cfg['freestream']['temperature_K'], pressure=cfg['freestream']['pressure_Pa'])
        cases, order = {}, []
        for M in cfg['sweep']['mach']:
            for a in cfg['sweep']['aoa_deg']:
                cid = case_id(M, a)
                phys = dict(phys0, mach=float(M), aoa_deg=float(a))
                d = run/'cases'/cid
                d.mkdir(parents=True)
                write_cfg(d/'base.cfg', cfg_template(s, phys, cfg['numerics'], cfg['output']))
                cases[cid] = dict(mach=float(M), aoa_deg=float(a), physics=phys, base_cfg_sha256=sha(d/'base.cfg'))
                order.append(cid)
        ex = cfg['execution']
        plan = dict(
            name=cfg['name'], run_dir=str(run), created_unix=time.time(), config=cfg,
            purpose='Lip-section compression performance over Mach x AoA (isolator values are diagnostics)',
            geometry=dict(stages=cfg['design']['stages'], design_mach=cfg['design']['mach'],
                          temperature_ratio=cfg['design']['temperature_ratio'], capture_height_m=s['lip_y'],
                          ramp_angles_deg=[r['angle_deg'] for r in s['ramps']], lip=[s['ramp_length'], s['lip_y']],
                          lip_opening_m=gap, duct_gap_height_m=duct, contraction_ratio=gap/duct,
                          ramp_end_point=s.get('ramp_end_point'), outlet_x_m=s['outlet_x'], cowl_radius_m=0.,
                          design_p0_recovery=result['performance']['total_pressure_recovery']),
            mesh=dict(sha256=sha(run/'mesh'/'mesh.su2'), length_scale=cfg['mesh']['length_scale'],
                      nodes=quality['nodes'], quads=quality['quads'],
                      minimum_scaled_jacobian=quality['minimum_scaled_jacobian'], generator='intake_structured_mesh.rule_columns'),
            scripts={n: sha(snap/n) for n in SNAPSHOT},
            execution=dict(backend=ex['backend'], ranks=ex['ranks'], rank_budget=ex['ranks'], omp_threads=1,
                           su2_cfd=ex['su2_cfd'], mpirun=ex['mpirun'], python=ex['python'], setup=ex['setup'],
                           pbs=ex['pbs'], storage_reserve_gb=ex['storage_reserve_gb'], local_exclusive=ex['local_exclusive'],
                           su2_cfd_sha256=sha(ex['su2_cfd']) if Path(ex['su2_cfd']).is_file() else None),
            stages=cfg['stages'], gate=cfg['gate'], analysis=cfg['analysis'], output=cfg['output'],
            ladder=['second: from first order, LIMITER_ITER=limiter_iter, CFL cfl',
                    'after failed (nonphysical/nonfinite/crash): same seed, LIMITER_ITER=delayed_limiter_iter; then CFL retry_cfl',
                    'after unconverged (healthy, gate not met): from its last checkpoint, LIMITER_ITER=delayed_limiter_iter',
                    'stop at max_second_order_attempts, when no rule applies, or at the case budget'],
            order=order, cases=cases,
            retention='all fields kept during the run; finalization after review is scripts/sweep_retention.py',
            not_claimed=['mesh independence', 'natural start beyond the recorded pre-lip check', 'physical validation',
                         'viscous or real-gas effects'])
        dump(run/'plan.json', plan)
        try:  # boundary-condition and mesh figures to check before submitting
            import intake_sweep_plots as P
            (run/'analysis'/'figures').mkdir(parents=True, exist_ok=True)
            P.setup_figures(run, plan, run/'analysis'/'figures')
        except Exception as ex:
            log(f'setup figures skipped: {ex!r}')
    except Exception:
        log(f'prepare failed; partial run directory left for inspection: {run}')
        raise
    log(f'prepared {run}')
    print(json.dumps(dict(run_dir=str(run), cases=len(order), mesh_nodes=quality['nodes'],
                          lip_opening_m=round(gap, 5), duct_gap_height_m=round(duct, 5),
                          contraction_ratio=round(gap/duct, 4), length_scale=round(cfg['mesh']['length_scale'], 4),
                          min_kantrowitz_margin=round(min(r.get('kantrowitz_margin', float('nan')) for r in theory), 3)),
                     indent=1))
    return run


# ----------------------------------------------------------------------------- submit

def job_script(plan, run, cid, analysis=False):
    ex = plan['execution']
    pbs = ex['pbs']
    name = pbs['job_prefix']+('AGG' if analysis else f"{plan['order'].index(cid)+1:03d}")
    lines = ['#!/bin/bash', f'#PBS -N {name}', '#PBS -j oe', f"#PBS -o {run/'pbs'/((cid or 'aggregate')+'.out')}"]
    queue = pbs['analysis_queue'] if analysis and pbs['analysis_queue'] else pbs['queue']
    if queue:
        lines.append(f'#PBS -q {queue}')
    for r in (pbs['analysis_resources'] if analysis else pbs['resources']) or []:
        lines.append(f'#PBS -l {r}')
    lines += [f'#PBS {d}' for d in pbs['extra_directives']]
    lines += [*ex['setup'], 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
              f'export MPLCONFIGDIR={run}/pbs/mpl', f'cd {run}']
    cmd = 'aggregate' if analysis else f'run-case {cid}'
    lines.append(f'exec {ex["python"]} {run}/scripts/intake_sweep.py {cmd} {run}')
    return '\n'.join(lines)+'\n'


def submit(run, backend=None, only=None, dry_run=False, resubmit=False):
    run = Path(run).resolve()
    plan = load(run/'plan.json')
    backend = backend or plan['execution']['backend']
    cases = only or plan['order']
    require(all(c in plan['cases'] for c in cases), 'Unknown case in --only')
    for c in cases:
        st = run/'cases'/c/'state.json'
        if st.exists():
            raise ValueError(f'{c} already has a recorded state ({load(st).get("status")}); '
                             'never resubmit a started case automatically')
    ledger_path = run/'pbs'/'jobs.json'
    ledger = load(ledger_path) if ledger_path.exists() else dict(cases={}, aggregate=[])
    again = set(cases) & set(ledger['cases'])
    require(not again or (resubmit and only), 'Some cases were already submitted (see pbs/jobs.json); '
            'archive them with archive-case and use --only <case> --resubmit')
    for c in again:
        require(not any((run/'cases'/c).glob('first*')), f'{c} still has stage folders; run archive-case first')
        ledger.setdefault('history', []).append(dict(case=c, **ledger['cases'].pop(c)))
    (run/'pbs'/'mpl').mkdir(parents=True, exist_ok=True)
    if backend == 'local':
        require(not dry_run, 'dry run applies to pbs')
        for c in cases:
            ledger['cases'][c] = dict(backend='local', submitted_unix=time.time())
            dump(ledger_path, ledger)
            with (run/'pbs'/f'{c}.out').open('w') as out:
                rc = subprocess.run([sys.executable, str(run/'scripts'/'intake_sweep.py'), 'run-case', c, str(run)],
                                    stdout=out, stderr=subprocess.STDOUT, cwd=run).returncode
            ledger['cases'][c]['returncode'] = rc
            dump(ledger_path, ledger)
            sp = run/'cases'/c/'state.json'
            log(f'{c}: job exit {rc}, outcome {load(sp).get("outcome") if sp.exists() else "not started (see pbs/"+c+".out)"}')
        aggregate(run)
        return
    qsub = plan['execution']['pbs']['qsub']
    ids = []
    for c in cases:
        script = run/'pbs'/f'{c}.sh'
        script.write_text(job_script(plan, run, c))
        if dry_run:
            print(script.read_text())
            continue
        out = subprocess.run([qsub, str(script)], capture_output=True, text=True, cwd=run)
        require(out.returncode == 0, f'qsub failed for {c}: {out.stderr.strip()}')
        jid = out.stdout.strip().splitlines()[-1]
        ids.append(jid)
        ledger['cases'][c] = dict(backend='pbs', job_id=jid, script=script.name, submitted_unix=time.time())
        dump(ledger_path, ledger)
        log(f'{c}: {jid}')
    script = run/'pbs'/'aggregate.sh'
    script.write_text(job_script(plan, run, None, analysis=True))
    if dry_run:
        print(script.read_text())
        return
    all_ids = [v['job_id'] for v in ledger['cases'].values() if v.get('job_id')]
    out = subprocess.run([qsub, '-W', 'depend=afterany:'+':'.join(all_ids), str(script)],
                         capture_output=True, text=True, cwd=run)
    require(out.returncode == 0, f'qsub failed for the aggregation job: {out.stderr.strip()}')
    ledger['aggregate'].append(dict(job_id=out.stdout.strip().splitlines()[-1], depends_on=all_ids, submitted_unix=time.time()))
    dump(ledger_path, ledger)
    log(f'aggregation job {ledger["aggregate"][-1]["job_id"]} after {len(all_ids)} case jobs')


# ----------------------------------------------------------------------------- one SU2 stage

class History:
    """Incremental reader of complete SU2 history rows."""
    KEYS = ('Inner_Iter', 'rms[Rho]', 'rms[RhoE]', 'Nonphysical_Points')

    def __init__(self, path):
        self.path, self.offset, self.header = Path(path), 0, None
        self.rows = {k: [] for k in self.KEYS}

    def read(self):
        if not self.path.exists():
            return None
        with self.path.open() as f:
            if self.header is None:
                line = f.readline()
                if not line.endswith('\n'):
                    return None
                self.header = [z.strip().strip('"') for z in line.split(',')]
                require(all(k in self.header for k in self.KEYS), f'history.csv lacks {self.KEYS}')
                self.offset = f.tell()
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line or not line.endswith('\n'):
                    break
                vals = line.split(',')
                if len(vals) != len(self.header):
                    raise ValueError('Malformed history row')
                for k in self.KEYS:
                    self.rows[k].append(float(vals[self.header.index(k)]))
                self.offset = f.tell()
        return self.rows if self.rows['Inner_Iter'] else None


def checkpoints(stage, suffix):
    return sorted(Path(stage).glob('restart_flow_*'+suffix), key=lambda z: int(z.stem.rsplit('_', 1)[1]))


def link_seed(source, target):
    """Place a seed without new data blocks: hard link, else APFS clone, else copy. Returns the method."""
    try:
        os.link(source, target)
        return 'hardlink'
    except OSError:
        pass
    if sys.platform == 'darwin' and subprocess.run(['cp', '-c', str(source), str(target)],
                                                   capture_output=True).returncode == 0:
        return 'clone'
    shutil.copy2(source, target)
    return 'copy'


def solver_processes():
    try:
        ps = subprocess.check_output(['ps', '-axo', 'pid,comm'], text=True).splitlines()[1:]
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(z.split()[0]) for z in ps if z.split()[-1].rsplit('/', 1)[-1] in ('SU2_CFD', 'mpirun', 'prterun', 'orted')]


class Stage:
    """Run one SU2 stage with monitoring; the second-order gate stops it when qualified."""

    def __init__(self, ctx, name, *, order, iterations, cfl, limiter, seed_file, first_reference=None):
        self.ctx, self.name, self.order = ctx, name, order
        self.iterations, self.cfl, self.limiter = iterations, cfl, limiter
        self.dir = ctx.case_dir/name
        self.seed_file, self.first_reference = seed_file, first_reference
        self.output = output_settings(ctx.plan)
        self.suffix = restart_suffix(self.output)

    def write_frequencies(self, files):
        """OUTPUT_WRT_FREQ per OUTPUT_FILES entry: restarts at every checkpoint, VTU only if asked."""
        if self.order != 2:
            return str(self.iterations)
        period = self.ctx.plan['stages']['checkpoint_period']
        volume = period if self.output['checkpoint_volume'] else self.iterations
        names = files.strip('( )').replace(',', ' ').split()
        return ', '.join(str(volume if n.startswith('PARAVIEW') else period) for n in names)

    def run(self):
        ctx, plan, p = self.ctx, self.ctx.plan, self.ctx.plan['stages']
        require(not self.dir.exists(), f'Duplicate stage {self.dir} refused')
        self.dir.mkdir()
        os.symlink(ctx.run/'mesh'/'mesh.su2', self.dir/'mesh.su2')
        cfg = read_cfg(ctx.case_dir/'base.cfg')
        second = self.order == 2
        cfg.update(ITER=str(self.iterations), CFL_NUMBER=repr(self.cfl), LIMITER_ITER=str(self.limiter),
                   MUSCL_FLOW='YES' if second else 'NO', CONV_STARTITER=str(self.iterations+1),
                   OUTPUT_WRT_FREQ=self.write_frequencies(cfg['OUTPUT_FILES']))
        seed_link = None
        if self.seed_file:
            require(Path(self.seed_file).suffix == self.suffix, f'Seed {self.seed_file} is not a {self.suffix} restart')
            # SU2 8.5.0 crashes on ASCII restart paths longer than ~100 characters: place it under a short name.
            seed_link = link_seed(self.seed_file, self.dir/('seed_restart'+self.suffix))
            cfg.update(RESTART_SOL='YES', SOLUTION_FILENAME='seed_restart')
        else:
            cfg.update(RESTART_SOL='NO')
        write_cfg(self.dir/'case.cfg', cfg)
        ex = plan['execution']
        cmd = [z.replace('{ranks}', str(ex['ranks'])) for z in ex['mpirun']]+[ex['su2_cfd'], 'case.cfg']
        rec = dict(case=ctx.cid, stage=self.name, order=self.order, status='dispatching', returncode=None, command=cmd,
                   ranks=ex['ranks'], omp_threads=1, iterations=self.iterations, cfl=self.cfl, limiter_iter=self.limiter,
                   seed=str(Path(self.seed_file).relative_to(ctx.run)) if self.seed_file else None,
                   seed_sha256=sha(self.seed_file) if self.seed_file else None, seed_link=seed_link,
                   config_sha256=sha(self.dir/'case.cfg'), mesh_sha256=plan['mesh']['sha256'],
                   pbs_job_id=os.environ.get('PBS_JOBID'), host=None, started_unix=time.time(),
                   max_solver_seconds=self.ctx.stage_seconds(), model_in_loop=False, retention='all fields kept')
        dump(self.dir/'run.json', rec)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        hist = History(self.dir/'history.csv')
        gates, checked, cache, reason, proc, nonphys, first_np = [], set(), {}, None, None, 0., None
        start = time.monotonic()
        lastsave = 0.
        try:
            with (self.dir/'solver.log').open('w') as logf:
                proc = subprocess.Popen(cmd, cwd=self.dir, stdout=logf, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env)
                rec.update(status='running', pid=proc.pid)
                dump(self.dir/'run.json', rec)
                limit = rec['max_solver_seconds']
                while proc.poll() is None:
                    elapsed = time.monotonic()-start
                    if elapsed >= limit and not reason:
                        reason = 'time_limit'
                    if shutil.disk_usage(ctx.run).free < ex['storage_reserve_gb']*1e9 and not reason:
                        reason = 'disk_reserve'
                    h = hist.read()
                    if h:
                        last = {k: v[-1] for k, v in h.items()}
                        if not all(math.isfinite(v) for v in last.values()) and not reason:
                            reason = 'nonfinite_history'
                        if last['Nonphysical_Points'] > 0 and first_np is None:
                            first_np = int(last['Inner_Iter'])
                        nonphys = max(nonphys, last['Nonphysical_Points'])
                        if nonphys > 0 and second and not reason:
                            reason = 'nonphysical_points'
                        rec['progress'] = dict(iteration=int(last['Inner_Iter']), elapsed_solver_s=round(elapsed, 1),
                                               rms_rho=last['rms[Rho]'], rms_rhoe=last['rms[RhoE]'], nonphysical_max=nonphys)
                        if second and not reason:
                            for fp in checkpoints(self.dir, self.suffix):
                                it = int(fp.stem.rsplit('_', 1)[1])
                                if it in checked:
                                    continue
                                pair = [fp, fp.with_suffix('.dat')]
                                if not (all(z.exists() and time.time()-z.stat().st_mtime > 3 for z in pair)
                                        and last['Inner_Iter'] > it):
                                    break
                                g = self.evaluate(it, h, nonphys, cache)
                                checked.add(it)
                                gates.append(g)
                                rec['last_gate'] = {k: g[k] for k in ('iteration', 'qualified', 'gates', 'lip_relative_change', 'started')}
                                if g['qualified']:
                                    reason = 'lip_gate_qualified'
                                    break
                    if reason and 'stop_signal_s' not in rec:
                        rec.update(stop_reason=reason, stop_signal_s=round(elapsed, 1))
                        dump(self.dir/'run.json', rec)
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                    if elapsed-lastsave >= 30:
                        dump(self.dir/'run.json', rec)
                        ctx.progress(self.name, rec.get('progress'))
                        lastsave = elapsed
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            rec.update(returncode=proc.returncode, elapsed_solver_s=round(time.monotonic()-start, 1))
            text = (self.dir/'solver.log').read_text(errors='replace')
            rec['exit_success'] = 'Exit Success' in text
            h = hist.read()
            if h:
                arr = h['Nonphysical_Points']
                nonphys = max([nonphys]+arr)
                hit = [int(i) for i, v in zip(h['Inner_Iter'], arr) if v > 0]
                first_np = hit[0] if hit else first_np
                rec['last_iteration'] = int(h['Inner_Iter'][-1])
            rec.update(nonphysical_max=nonphys, first_nonphysical_iteration=first_np)
            if second and reason is None and nonphys > 0:
                reason = 'nonphysical_points'  # seen only after the solver had already ended
            if second and reason is None and proc.returncode == 0 and h:
                for fp in checkpoints(self.dir, self.suffix):  # natural end at ITER: gate the remaining checkpoints
                    it = int(fp.stem.rsplit('_', 1)[1])
                    if it not in checked and it <= h['Inner_Iter'][-1]:
                        g = self.evaluate(it, h, nonphys, cache)
                        checked.add(it)
                        gates.append(g)
                        if g['qualified']:
                            reason = 'lip_gate_qualified'
                            break
            rec['status'] = self.classify(rec, reason, gates)
        except (Exception, Cancelled) as ex_:
            if proc and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            rec.update(status='cancelled' if isinstance(ex_, Cancelled) else 'error', error=repr(ex_),
                       traceback=traceback.format_exc(), returncode=proc.returncode if proc else None,
                       elapsed_solver_s=round(time.monotonic()-start, 1))
            if isinstance(ex_, Cancelled):
                dump(self.dir/'run.json', rec)
                raise
        rec['stop_reason'] = rec.get('stop_reason') or reason
        rec['gates'] = [{k: g[k] for k in ('iteration', 'qualified', 'gates', 'lip_relative_change', 'residual_decades', 'started', 'checkpoint')}
                        for g in gates]
        healthy = [g for g in gates if g['gates']['health']]
        rec['last_healthy_checkpoint'] = healthy[-1]['checkpoint'] if healthy else None
        rec['qualified_checkpoint'] = next((g['checkpoint'] for g in gates if g['qualified']), None)
        rec['finished_unix'] = time.time()
        dump(self.dir/'run.json', rec)
        return rec

    def classify(self, rec, reason, gates):
        if self.order == 1:
            ok = rec['returncode'] == 0 and rec['exit_success'] and self.final_restart() is not None
            if ok:
                import intake_sweep_metrics as M
                b = M.balance(self.ctx.mesh(), self.ctx.mesh().state(self.final_restart()), self.ctx.phys)
                rec['health'] = b
                ok = bool(b['finite'] and b['density_min'] > 0 and b['pressure_min'] > 0)
            return 'completed' if ok else 'failed'
        if any(g['qualified'] for g in gates):
            return 'qualified'
        if reason in ('nonfinite_history', 'nonphysical_points'):
            return 'failed'
        if reason == 'disk_reserve':
            return 'stopped_disk'
        if rec['returncode'] not in (0, None) and reason is None:
            return 'failed'  # solver crash
        if not gates or not gates[-1]['gates']['health']:
            return 'failed'
        return 'unconverged'

    def final_restart(self):
        p = self.dir/('restart_flow'+self.suffix)
        if p.exists():
            return p
        c = checkpoints(self.dir, self.suffix)
        return c[-1] if c else None

    def evaluate(self, iteration, history, nonphys, cache):
        import intake_sweep_metrics as M
        ctx = self.ctx
        files = {int(z.stem.rsplit('_', 1)[1]): z for z in checkpoints(self.dir, self.suffix)}

        def metrics(it):
            if it not in cache:
                d = ctx.mesh().state(files[it])
                lip = M.lip_sections(ctx.mesh(), d, ctx.phys, ctx.plan['analysis']['pre_lip_offset_m'])
                cache[it] = dict(lip=lip['lip'][0], pre=lip['pre'][0], balance=M.balance(ctx.mesh(), d, ctx.phys))
            return cache[it]
        earlier = [i for i in files if i < iteration]
        cur = metrics(iteration)
        prev = metrics(max(earlier)) if earlier else None
        g = M.gate(cur, prev, history, iteration, self.first_reference, nonphys, ctx.plan['gate'])
        g.update(checkpoint=str(files[iteration].relative_to(ctx.run)), checkpoint_sha256=sha(files[iteration]),
                 metrics=dict(lip=cur['lip'], pre=cur['pre']), balance=cur['balance'])
        dump(self.dir/f'gate-{iteration:06d}.json', g)
        return g


# ----------------------------------------------------------------------------- one case

class CaseContext:
    def __init__(self, run, cid):
        self.run, self.cid = Path(run).resolve(), cid
        self.plan = load(self.run/'plan.json')
        self.spec = self.plan['cases'][cid]
        self.phys = self.spec['physics']
        self.case_dir = self.run/'cases'/cid
        self._mesh = None
        self.state_path = self.case_dir/'state.json'
        self.state = None
        self.started = time.monotonic()

    def mesh(self):
        if self._mesh is None:
            import intake_sweep_metrics as M
            self._mesh = M.Mesh(self.run/'mesh')
        return self._mesh

    def used(self):
        return sum(load(p).get('elapsed_solver_s') or 0 for p in self.case_dir.glob('*/run.json'))

    def stage_seconds(self):
        st = self.plan['stages']
        return max(60., min(st['max_stage_seconds'], st['case_budget_seconds']-self.used()))

    def save(self, **kw):
        self.state.update(kw, updated_unix=time.time())
        dump(self.state_path, self.state)

    def progress(self, stage, progress):
        if progress:
            self.save(current_stage=stage, progress=progress)


def next_attempt(attempts, rules):
    """The retry ladder. attempts: list of dict(name, seed_kind, seed, limiter, cfl, status, last_checkpoint)."""
    if len(attempts) >= rules['max_second_order_attempts']:
        return None
    last = attempts[-1]
    if last['status'] in ('failed', 'error'):
        if last['limiter'] < rules['delayed_limiter_iter']:
            return dict(seed=last['seed'], limiter=rules['delayed_limiter_iter'], cfl=last['cfl'], why='delayed limiter freeze after failure')
        if last['cfl'] > rules['retry_cfl']:
            return dict(seed=last['seed'], limiter=last['limiter'], cfl=rules['retry_cfl'], why='lower CFL after failure')
        return None
    if last['status'] == 'unconverged' and last.get('last_checkpoint'):
        if any(a.get('why', '').startswith('continue') for a in attempts):
            return None
        return dict(seed=last['last_checkpoint'], limiter=rules['delayed_limiter_iter'], cfl=last['cfl'],
                    why='continue from last checkpoint with delayed limiter freeze')
    return None


def run_case(run, cid):
    ctx = CaseContext(run, cid)
    ctx.case_dir.mkdir(exist_ok=True)
    lock = (ctx.case_dir/'case.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f'{cid}: another job holds the case lock')
    if ctx.state_path.exists():
        prev = load(ctx.state_path)
        raise SystemExit(f'{cid}: state already recorded ({prev.get("status")}); refusing to rerun a started case')
    plan = ctx.plan
    require(sha(ctx.case_dir/'base.cfg') == ctx.spec['base_cfg_sha256'], 'Frozen base.cfg changed')
    require(sha(ctx.run/'mesh'/'mesh.su2') == plan['mesh']['sha256'], 'Frozen mesh changed')
    for n, h in plan['scripts'].items():
        require(sha(ctx.run/'scripts'/n) == h, f'Script snapshot {n} changed')
    if plan['execution']['backend'] == 'local' and plan['execution'].get('local_exclusive', True):
        require(not solver_processes(), 'Another SU2/MPI process is running on this machine')
    ctx.state = dict(case=cid, mach=ctx.spec['mach'], aoa_deg=ctx.spec['aoa_deg'], status='running',
                     pbs_job_id=os.environ.get('PBS_JOBID'), host=os.uname().nodename, started_unix=time.time(),
                     attempts=[])
    ctx.save()
    st = plan['stages']
    outcome = None
    signal.signal(signal.SIGTERM, _cancel)
    signal.signal(signal.SIGINT, _cancel)
    try:
        # first order from the uniform freestream; one lower-CFL retry
        first = None
        for name, cfl in (('first', st['cfl']), ('first-retry', st['retry_cfl'])):
            ctx.save(current_stage=name)
            rec = Stage(ctx, name, order=1, iterations=st['first_iterations'], cfl=cfl, limiter=st['limiter_iter'],
                        seed_file=None).run()
            ctx.state['attempts'].append(dict(name=name, order=1, cfl=cfl, status=rec['status'],
                                              returncode=rec['returncode'], stop_reason=rec.get('stop_reason')))
            ctx.save()
            if rec['status'] == 'completed':
                first = Stage(ctx, name, order=1, iterations=0, cfl=cfl, limiter=0, seed_file=None)
                break
        if first is None:
            outcome = 'failed'
            ctx.save(failure='first-order stage failed twice')
        else:
            import intake_sweep_metrics  # noqa: F401  (fail early if numpy/matplotlib are missing)
            fh = History(first.dir/'history.csv').read()
            ref = {k: float(max(fh[k][:10])) for k in ('rms[Rho]', 'rms[RhoE]')}
            seed0 = first.final_restart()
            attempts = []
            nxt = dict(seed=str(seed0), limiter=st['limiter_iter'], cfl=st['cfl'], why='standard')
            while nxt is not None:
                if st['case_budget_seconds']-ctx.used() < 120:
                    ctx.save(budget_exhausted=True)
                    break
                name = 'second' if not attempts else f'second-{len(attempts)+1}'
                ctx.save(current_stage=name)
                rec = Stage(ctx, name, order=2, iterations=st['second_iterations'], cfl=nxt['cfl'],
                            limiter=nxt['limiter'], seed_file=Path(nxt['seed']), first_reference=ref).run()
                ckpt = rec.get('last_healthy_checkpoint')
                a = dict(name=name, order=2, seed=nxt['seed'], seed_rel=rec.get('seed'), limiter=nxt['limiter'], cfl=nxt['cfl'],
                         why=nxt['why'], status=rec['status'], returncode=rec['returncode'], stop_reason=rec.get('stop_reason'),
                         last_iteration=rec.get('last_iteration'), first_nonphysical_iteration=rec.get('first_nonphysical_iteration'),
                         last_checkpoint=str(ctx.run/ckpt) if ckpt else None, qualified_checkpoint=rec.get('qualified_checkpoint'),
                         elapsed_solver_s=rec.get('elapsed_solver_s'), last_gate=(rec['gates'] or [None])[-1])
                attempts.append(a)
                ctx.state['attempts'].append(a)
                ctx.save()
                if rec['status'] == 'qualified':
                    break
                if rec['status'] == 'stopped_disk':
                    break
                nxt = next_attempt(attempts, st)
            outcome = attempts[-1]['status'] if attempts else 'failed'
        ctx.save(status='analysing', outcome=outcome, current_stage=None)
        analyze_case(ctx)
        ctx.save(status='done')
    except Cancelled as ex:
        ctx.save(status='cancelled', outcome='cancelled', error=repr(ex), current_stage=None)
        write_diagnosis(ctx)
        raise SystemExit(f'{cid}: cancelled ({ex})')
    except Exception as ex:
        ctx.save(status='error', error=repr(ex), traceback=traceback.format_exc(), outcome=outcome or 'error')
        write_diagnosis(ctx)
        raise
    write_diagnosis(ctx)
    log(f'{cid}: {ctx.state["outcome"]} ({ctx.state.get("use")})')


def chosen_attempt(state):
    """Qualified attempt, else the last healthy unconverged one (diagnostic values)."""
    att = [a for a in state['attempts'] if a.get('order') == 2]
    q = [a for a in att if a['status'] == 'qualified']
    if q:
        return q[0], q[0]['qualified_checkpoint']
    u = [a for a in att if a['status'] == 'unconverged' and a.get('last_checkpoint')]
    if u:
        return u[-1], u[-1]['last_checkpoint']
    return None, None


def analyze_case(ctx):
    import intake_sweep_metrics as M
    att, ckpt = chosen_attempt(ctx.state)
    ana = ctx.plan['analysis']
    out = dict(case=ctx.cid, mach=ctx.spec['mach'], aoa_deg=ctx.spec['aoa_deg'], outcome=ctx.state['outcome'])
    if att is None:
        out.update(use='none')
        dump(ctx.case_dir/'result.json', out)
        ctx.save(use='none')
        plot_case(ctx, None, None)
        return
    path = Path(ckpt) if Path(ckpt).is_absolute() else ctx.run/ckpt
    mesh = ctx.mesh()
    d = mesh.state(path)
    lip = M.lip_sections(mesh, d, ctx.phys, ana['pre_lip_offset_m'], full=True)
    started = M.started(lip['pre'][0])
    qualified = att['status'] == 'qualified'
    use = 'performance' if qualified and started else 'diagnostic'
    sts, profiles = [], {'C-lip': lip['lip'][1]}
    target = lip['lip'][0]['massflow']
    stations = M.stations(mesh.settings, ana)
    isolators = [s for s in stations if s['kind'] == 'isolator']
    mid = isolators[len(isolators)//2]['id'] if isolators else None
    for s in stations:
        try:
            m, prof = (lip['pre'] if s['id'] == 'C-pre' else lip['lip'] if s['id'] == 'C-lip'
                       else M.station_metrics(mesh, d, ctx.phys, s, target))
            sts.append(dict(station=s['id'], kind=s['kind'], isolator_distance_m=s.get('isolator_distance_m'), **m))
            if s['id'] in ana['profile_stations'] or (s['id'] == mid and 'MID' in ana['profile_stations']):
                profiles[s['id']] = prof
        except ValueError as ex:
            sts.append(dict(station=s['id'], kind=s['kind'], x_m=s['x'], error=str(ex)))
    out.update(use=use, started=started, qualified=qualified, attempt=att['name'], attempts_used=len(
        [a for a in ctx.state['attempts'] if a.get('order') == 2]), checkpoint=str(path.relative_to(ctx.run)),
        checkpoint_sha256=sha(path), iteration=int(path.stem.rsplit('_', 1)[1]) if '_' in path.stem else None,
        lip=lip['lip'][0], pre=lip['pre'][0], balance=M.balance(mesh, d, ctx.phys), stations=sts,
        profile_stations=list(profiles), mid_isolator_station=mid)
    import numpy as np
    dump(ctx.case_dir/'profiles.json', {k: {n: np.asarray(v).tolist() for n, v in p.items()} for k, p in profiles.items()})
    dump(ctx.case_dir/'result.json', out)
    ctx.save(use=use, started=started)
    plot_case(ctx, out, profiles)
    adopted_volume(ctx, path)


def adopted_volume(ctx, checkpoint):
    """One VTU of the adopted checkpoint: SU2_CFD on one rank with ITER=0 restarts from it and writes the
    volume output without iterating (SU2_SOL would write only the restart's fields). A failure is recorded."""
    output, ex = output_settings(ctx.plan), ctx.plan['execution']
    if not output['adopted_volume'] or output['checkpoint_volume']:
        return
    out = ctx.case_dir/'volume'
    rec = dict(checkpoint=str(checkpoint.relative_to(ctx.run)), checkpoint_sha256=sha(checkpoint), status='failed',
               note='SU2_CFD at ITER=0: solution fields as saved; limiter fields recomputed from the saved state')
    try:
        out.mkdir()
        os.symlink(ctx.run/'mesh'/'mesh.su2', out/'mesh.su2')
        os.link(checkpoint, out/('adopted'+checkpoint.suffix))
        cfg = read_cfg(checkpoint.parent/'case.cfg')
        cfg.update(ITER='0', RESTART_SOL='YES', SOLUTION_FILENAME='adopted', OUTPUT_FILES='( PARAVIEW )',
                   OUTPUT_WRT_FREQ='1', VOLUME_FILENAME='flow', WRT_VOLUME_OVERWRITE='YES', CONV_FILENAME='history',
                   READ_BINARY_RESTART='NO' if checkpoint.suffix == '.csv' else 'YES')
        write_cfg(out/'volume.cfg', cfg)
        cmd = [z.replace('{ranks}', '1') for z in ex['mpirun']]+[ex['su2_cfd'], 'volume.cfg']
        start = time.monotonic()
        with (out/'solver.log').open('w') as logf:
            rc = subprocess.run(cmd, cwd=out, stdout=logf, stderr=subprocess.STDOUT, timeout=1800,
                                env=dict(os.environ, OMP_NUM_THREADS='1')).returncode
        vtu = out/'flow.vtu'
        h = History(out/'history.csv').read()
        rec.update(command=cmd, returncode=rc, elapsed_s=round(time.monotonic()-start, 1), iterations_run=len(h['Inner_Iter']) if h else 0)
        if rc == 0 and vtu.is_file() and not h:
            rec.update(status='written', vtu=str(vtu.relative_to(ctx.run)), vtu_sha256=sha(vtu))
    except Exception as ex_:
        rec.update(error=repr(ex_))
    finally:
        (out/('adopted'+checkpoint.suffix)).unlink(missing_ok=True)
    dump(ctx.case_dir/'volume.json', rec)
    log(f'{ctx.cid}: adopted VTU {rec["status"]}')


def plot_case(ctx, result, profiles):
    try:
        import intake_sweep_plots as P
        P.case_figures(ctx.run, ctx.cid, ctx.state, result, profiles)
    except Exception as ex:
        ctx.save(plot_error=repr(ex))


def write_diagnosis(ctx):
    """Human-readable account of what happened (Japanese, for the workstation user)."""
    s = ctx.state
    L = [f'# {ctx.cid}（Mach {s["mach"]:g}、迎角 {s["aoa_deg"]:+g}°）', '',
         f'- 判定: **{s.get("outcome")}**　用途: {s.get("use", "-")}　状態: {s.get("status")}',
         f'- PBSジョブ: {s.get("pbs_job_id") or "-"}　ホスト: {s.get("host")}', '']
    if s.get('error'):
        L += ['## スクリプトの異常終了', '', '```', s.get('traceback', s['error'])[-3000:], '```', '']
    L += ['## 試行の経過', '', '| 段 | 種 | LIMITER_ITER | CFL | 結果 | 停止理由 | 最終反復 | 最初の非物理点 | 残差の低下（ρ / ρE 桁） | リップ値の変化 | 始動 |',
          '|---|---|---:|---:|---|---|---:|---:|---|---:|---|']
    for a in s['attempts']:
        g = a.get('last_gate') or {}
        dec = g.get('residual_decades') or {}
        decs = ' / '.join(f'{v:.2f}' if v is not None else '-' for v in dec.values()) if dec else '-'
        ch = g.get('lip_relative_change')
        L.append(f"| {a['name']} | {Path(a['seed']).parent.name if a.get('seed') else '一様流'} | {a.get('limiter', '-')} | {a['cfl']:g} | "
                 f"{a['status']} | {a.get('stop_reason') or '-'} | {a.get('last_iteration') or '-'} | "
                 f"{a.get('first_nonphysical_iteration') or '-'} | {decs} | {'%.1e' % ch if ch is not None else '-'} | "
                 f"{g.get('started', '-')} |")
    L += ['', '## 読み方と次の手', '']
    o = s.get('outcome')
    if o == 'qualified' and s.get('started') is False:
        L.append('- 受入判定は満たしたが、リップ手前の断面に逆流または亜音速の流れがある（不始動の疑い）。性能値として使わない。')
    elif o == 'qualified':
        L.append('- 受入。リップ断面の値を性能として使う。')
    elif o == 'unconverged':
        L.append('- 非物理点は出ていないが、受入判定（残差・リップ値の安定）を満たさなかった。最後の健全な保存点の値を診断用として集計に含める。')
        L.append('- 次の手の例: より長い反復、CFLの変更、格子の確認。リップ値の変化が1e-4付近なら値自体は安定している可能性が高い。')
    elif o in ('failed', 'error'):
        L.append('- 非物理点・発散・ソルバーの異常終了で、使える二次精度解が得られなかった。各段の solver.log と history.csv を確認する。')
        L.append('- 最初の非物理点の反復が LIMITER_ITER の直後なら、limiter 固定の時期の影響が疑われる（前回の掃引と同じ傾向）。')
    if s.get('budget_exhausted'):
        L.append('- ケースの計算時間の上限に達したため、残りの試行を行わなかった。')
    (ctx.case_dir/'diagnosis.md').write_text('\n'.join(L)+'\n')


# ----------------------------------------------------------------------------- aggregate

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def _fmt(v):
    if isinstance(v, float):
        return '' if not math.isfinite(v) else f'{v:.6g}'
    return '' if v is None else v


def aggregate(run):
    run = Path(run).resolve()
    plan = load(run/'plan.json')
    theory = {r['case']: r for r in csv.DictReader((run/'analysis'/'theory.csv').open())}
    ana = run/'analysis'
    lips, profiles, isos, streams, settings, execution, failures, cases = [], [], [], [], [], [], [], []
    for cid in plan['order']:
        spec = plan['cases'][cid]
        cdir = run/'cases'/cid
        st = load(cdir/'state.json') if (cdir/'state.json').exists() else dict(status='not_started', attempts=[])
        res = load(cdir/'result.json') if (cdir/'result.json').exists() else {}
        th = theory.get(cid, {})
        status = st.get('status')
        outcome = (st.get('outcome') if status in ('done', 'error', 'cancelled')
                   else 'unknown' if status in ('running', 'analysing') else status)
        second = [a for a in st.get('attempts', []) if a.get('order') == 2]
        row = dict(case=cid, mach=spec['mach'], aoa_deg=spec['aoa_deg'], outcome=outcome, use=res.get('use', 'none'),
                   started=res.get('started'), attempts=len(second), attempt_used=res.get('attempt'), iteration=res.get('iteration'))
        lip, pre = res.get('lip') or {}, res.get('pre') or {}
        for k in ('p0_recovery', 'T_ratio', 'p_ratio', 'mach', 'flow_angle_deg', 'massflow_over_rhoV_capture',
                  'p0_recovery_mixedout', 'T_ratio_mixedout', 'p_ratio_mixedout', 'mach_mixedout',
                  'p0_recovery_p05', 'p0_recovery_p95', 'T_ratio_p05', 'T_ratio_p95', 'freestream_state_mass_fraction',
                  'reverse_massflow_fraction', 'subsonic_forward_massflow_fraction', 'T0_error'):
            row[k if k.endswith(('mixedout', 'p05', 'p95', 'fraction', 'error', 'capture', 'deg')) else k+'_massavg'] = lip.get(k)
        row['pre_lip_p0_recovery'] = pre.get('p0_recovery')
        row['pre_lip_T_ratio'] = pre.get('T_ratio')
        for k in ('theory_p0_recovery', 'theory_T_ratio', 'theory_p_ratio', 'theory_lip_mach', 'theory_waves_cross_ahead_of_lip',
                  'theory_waves_above_lip', 'kantrowitz_max_contraction', 'geometric_contraction', 'kantrowitz_margin'):
            row[k] = _num(th.get(k))
        for k, t in (('p0_recovery_massavg', 'theory_p0_recovery'), ('T_ratio_massavg', 'theory_T_ratio'),
                     ('p_ratio_massavg', 'theory_p_ratio'), ('mach_massavg', 'theory_lip_mach')):
            row['d_'+k.replace('_massavg', '')+'_vs_theory'] = (row[k]-row[t]) if num(row.get(k)) and num(row.get(t)) else None
        lips.append(row)
        if (cdir/'profiles.json').exists():
            for sid, p in load(cdir/'profiles.json').items():
                if sid != 'C-lip':
                    continue
                for i in range(len(p['height_above_wall_mm'])):
                    profiles.append(dict(case=cid, mach=spec['mach'], aoa_deg=spec['aoa_deg'],
                                         **{k: p[k][i] for k in ('height_above_wall_mm', 'p0_recovery', 'T_ratio', 'p_ratio')},
                                         mach_local=p['mach'][i], flow_angle_deg=p['flow_angle_deg'][i]))
        for s in res.get('stations', []):
            r = dict(case=cid, mach=spec['mach'], aoa_deg=spec['aoa_deg'], use=res.get('use'), **{
                (k+'_massavg' if k in LOCAL else k): v for k, v in s.items() if not isinstance(v, (dict, list))})
            streams.append(r)
            if s['kind'] in ('isolator', 'outlet'):
                isos.append(r)
        cfgs = sorted((cdir).glob('*/case.cfg'))
        settings.append(dict(case=cid, mach=spec['mach'], aoa_deg=spec['aoa_deg'],
                             first_order_config=_rel(cfgs, 'first', run), final_config=_rel(cfgs, res.get('attempt') or '', run),
                             inlet=read_cfg(cdir/'base.cfg')['MARKER_SUPERSONIC_INLET']))
        for a in st.get('attempts', []):
            rj = cdir/a['name']/'run.json'
            r = load(rj) if rj.exists() else {}
            execution.append(dict(case=cid, stage=a['name'], order=a.get('order'), seed=r.get('seed'), limiter_iter=a.get('limiter'),
                                  cfl=a.get('cfl'), why=a.get('why'), status=a['status'], returncode=a.get('returncode'),
                                  stop_reason=a.get('stop_reason'), last_iteration=r.get('last_iteration'),
                                  first_nonphysical_iteration=r.get('first_nonphysical_iteration'),
                                  solver_seconds=r.get('elapsed_solver_s'), pbs_job_id=r.get('pbs_job_id'),
                                  config=str((cdir/a['name']/'case.cfg').relative_to(run))))
        if row['use'] != 'performance':
            last = (second or st.get('attempts') or [{}])[-1]
            failures.append(dict(case=cid, mach=spec['mach'], aoa_deg=spec['aoa_deg'], outcome=outcome, use=row['use'],
                                 started=res.get('started'), attempts=len(second), last_stage=last.get('name'),
                                 last_status=last.get('status'), last_stop_reason=last.get('stop_reason'),
                                 first_nonphysical_iteration=last.get('first_nonphysical_iteration'),
                                 error=st.get('error'), diagnosis=str((cdir/'diagnosis.md').relative_to(run))))
        cases.append(dict(row=row, state=st, result=res))
    write_csv(ana/'lip-results.csv', lips)
    write_csv(ana/'lip-profiles.csv', profiles)
    write_csv(ana/'isolator-sections.csv', isos)
    write_csv(ana/'streamwise-sections.csv', streams)
    write_csv(ana/'case-settings.csv', settings)
    write_csv(ana/'execution.csv', execution)
    write_csv(ana/'failures.csv', failures or [dict(case='', outcome='none')])
    try:
        import intake_sweep_plots as P
        figs = P.sweep_figures(run, plan, lips, streams, cases)
    except Exception as ex:
        figs = []
        log(f'figures failed: {ex!r}')
        (ana/'figures-error.txt').write_text(traceback.format_exc())
    summary(run, plan, lips, failures, figs)
    dump(ana/'manifest.json', dict(created_unix=time.time(), plan_sha256=sha(run/'plan.json'),
                                   mesh_sha256=plan['mesh']['sha256'], scripts=plan['scripts'],
                                   files={p.name: sha(p) for p in sorted(ana.glob('*.csv'))}))
    log(f'aggregated {len(lips)} cases into {ana}')


def _num(v):
    if v in (None, ''):
        return None
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return float(v)
    except ValueError:
        return v


def _rel(cfgs, stage, run):
    for c in cfgs:
        if c.parent.name == stage:
            return str(c.relative_to(run))
    return None


def summary(run, plan, lips, failures, figs):
    g = plan['geometry']
    count = {}
    for r in lips:
        key = r['use'] if r['outcome'] != 'unknown' else 'unknown'
        count[key] = count.get(key, 0)+1
    L = [f"# {plan['name']}", '',
         f"{g['stages']}段ランプ、設計 Mach {g['design_mach']:g}・ランプ後静温比 {g['temperature_ratio']:g}、"
         f"捕獲高さ {g['capture_height_m']:g} m、リップ開口 {g['lip_opening_m']*1000:.1f} mm、流路 {g['duct_gap_height_m']*1000:.1f} mm"
         f"（収縮比 {g['contraction_ratio']:.3f}）。格子 {plan['mesh']['nodes']:,} 節点。", '',
         '## 判定の内訳', '', '| 用途 | ケース数 |', '|---|---:|'] + [f'| {k} | {v} |' for k, v in sorted(count.items())]
    L += ['', '`performance` は受入かつリップ手前で超音速（始動）。`diagnostic` は残差未達または不始動の疑いで、値は参考。'
          '`none` は使える解がない。詳細は failures.csv と各ケースの diagnosis.md。', '',
          '## リップ断面（質量流束平均）', '', '| case | 判定 | 用途 | p0回復率 | 静温比 | 静圧比 | Mach | 理論p0 | 差 |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    f = lambda v, n=3: '' if not num(v) else f'{v:.{n}f}'
    for r in lips:
        L.append(f"| {r['case']} | {r['outcome']} | {r['use']} | {f(r['p0_recovery_massavg'])} | {f(r['T_ratio_massavg'])} | "
                 f"{f(r['p_ratio_massavg'], 1)} | {f(r['mach_massavg'], 2)} | {f(r['theory_p0_recovery'])} | {f(r['d_p0_recovery_vs_theory'])} |")
    L += ['', '## 図', ''] + [f'- [{p}](figures/{p})' for p in figs]
    L += ['', '数値はすべて無次元（一様流の静圧・静温・全圧で割った値）。格子独立性・粘性・実在気体・自然始動は確認していない。']
    (run/'analysis'/'summary.md').write_text('\n'.join(L)+'\n')


# ----------------------------------------------------------------------------- status / compare

def archive_case(run, cid, confirm):
    """Move a finished, cancelled or dead case's records aside so it can be resubmitted."""
    run = Path(run).resolve()
    plan = load(run/'plan.json')
    require(cid in plan['cases'], f'Unknown case {cid}')
    cdir = run/'cases'/cid
    st = load(cdir/'state.json') if (cdir/'state.json').exists() else {}
    if st.get('status') in ('running', 'analysing'):
        require(confirm, f'{cid} is recorded as {st["status"]}: check with qstat that its job is gone, '
                         'then repeat with --confirm-not-running')
    lock = (cdir/'case.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise ValueError(f'{cid}: a job holds the case lock (still running)')
    n = 1
    while (cdir/f'archive-{n}').exists():
        n += 1
    dest = cdir/f'archive-{n}'
    dest.mkdir()
    moved = []
    for p in sorted(cdir.iterdir()):
        if p.name in ('base.cfg', 'case.lock') or p.name.startswith('archive-'):
            continue
        p.rename(dest/p.name)
        moved.append(p.name)
    dump(dest/'archived.json', dict(case=cid, archived_unix=time.time(), previous_status=st.get('status'),
                                    previous_outcome=st.get('outcome'), moved=moved))
    log(f'{cid}: moved {len(moved)} entries to {dest.relative_to(run)}; resubmit with '
        f'submit {run} --only {cid} --resubmit')


def status(run):
    run = Path(run).resolve()
    plan = load(run/'plan.json')
    ledger = load(run/'pbs'/'jobs.json') if (run/'pbs'/'jobs.json').exists() else dict(cases={})
    print(f"{plan['name']}: {len(plan['order'])} cases")
    print(f"{'case':12s} {'job':14s} {'status':11s} {'stage':10s} {'iter':>7s} {'outcome':12s} use")
    for cid in plan['order']:
        p = run/'cases'/cid/'state.json'
        st = load(p) if p.exists() else {}
        pr = st.get('progress') or {}
        job = (ledger['cases'].get(cid) or {}).get('job_id', '-')
        print(f"{cid:12s} {str(job)[:14]:14s} {st.get('status', 'queued' if cid in ledger['cases'] else '-'):11s} "
              f"{str(st.get('current_stage') or '-'):10s} {str(pr.get('iteration', '-')):>7s} "
              f"{str(st.get('outcome') or '-'):12s} {st.get('use', '-')}")


def compare(runs, output):
    import intake_sweep_plots as P
    out = Path(output)
    require(not out.exists(), f'{out} exists')
    P.compare_figures([Path(r).resolve() for r in runs], out)
    log(f'comparison figures in {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('config')
    p = sub.add_parser('submit')
    p.add_argument('run')
    p.add_argument('--backend', choices=('pbs', 'local'))
    p.add_argument('--only', nargs='+', help='submit only these never-started cases')
    p.add_argument('--dry-run', action='store_true', help='print the PBS scripts without qsub')
    p.add_argument('--resubmit', action='store_true', help='with --only: resubmit cases moved aside by archive-case')
    p = sub.add_parser('archive-case', help='move a finished/dead case aside before resubmitting it')
    p.add_argument('run')
    p.add_argument('case')
    p.add_argument('--confirm-not-running', action='store_true')
    p = sub.add_parser('run-case')
    p.add_argument('case')
    p.add_argument('run')
    p = sub.add_parser('aggregate')
    p.add_argument('run')
    p = sub.add_parser('status')
    p.add_argument('run')
    p = sub.add_parser('compare')
    p.add_argument('runs', nargs='+')
    p.add_argument('--output', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    try:
        if a.cmd == 'prepare':
            prepare(a.config)
        elif a.cmd == 'submit':
            submit(a.run, a.backend, a.only, a.dry_run, a.resubmit)
        elif a.cmd == 'archive-case':
            archive_case(a.run, a.case, a.confirm_not_running)
        elif a.cmd == 'run-case':
            run_case(a.run, a.case)
        elif a.cmd == 'aggregate':
            aggregate(a.run)
        elif a.cmd == 'status':
            status(a.run)
        else:
            compare(a.runs, a.output)
    except ValueError as ex:
        ap.exit(2, f'Error: {ex}\n')


if __name__ == '__main__':
    main()
