#!/usr/bin/env python3
"""Deterministic stand-in for SU2_CFD used by test_intake_sweep.py. It is not a CFD solver.

Reads the config named on the command line and writes history.csv and restart files
(a uniform freestream field on the mesh) at OUTPUT_WRT_FREQ: a binary restart, the ASCII
one only with RESTART_ASCII, and a placeholder flow.vtu per PARAVIEW frequency. ITER=0 only
checks the solution file and writes flow.vtu (the adopted-checkpoint VTU). The behaviour per
case and stage comes from the JSON file named by FAKE_SU2_PLAN, keyed by
"<case>:<stage>" or "<case>:*", with modes:
  converge     residuals fall 7 decades by 4000 iterations (default)
  nonphysical  Nonphysical_Points > 0 from iteration 1600
  stall        residuals fall 2 decades only (healthy, gate not met)
  crash        exit code 1 at iteration 50 without restart output
"""
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_airfoil_mesh import read_su2  # noqa: E402


def cfg():
    out = {}
    for line in Path(sys.argv[-1]).read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def write_dat(name, fields, rows):
    with open(name, 'wb') as f:
        f.write(struct.pack('<5i', 535532, len(fields), len(rows), 0, 0))
        for n in fields:
            f.write(n.encode().ljust(33, b'\0'))
        for r in rows:
            f.write(struct.pack(f'<{len(r)}d', *r))


def main():
    c = cfg()
    files = c['OUTPUT_FILES'].strip('( )').replace(',', ' ').split()
    freqs = [int(v) for v in c['OUTPUT_WRT_FREQ'].replace(',', ' ').split()]
    freqs += [freqs[-1]]*(len(files)-len(freqs))
    freq = dict(zip(files, freqs))
    if c.get('RESTART_SOL') == 'YES':
        seed = Path(c['SOLUTION_FILENAME']+('.dat' if c['READ_BINARY_RESTART'] == 'YES' else '.csv'))
        if not seed.is_file():
            sys.exit(f'missing seed {seed}')
    stage, case = Path.cwd().name, Path.cwd().parent.name
    plan = json.loads(Path(os.environ['FAKE_SU2_PLAN']).read_text()) if os.environ.get('FAKE_SU2_PLAN') else {}
    mode = plan.get(f'{case}:{stage}', plan.get(f'{case}:*', 'converge'))
    if c['MUSCL_FLOW'] == 'NO' and mode != 'crash-first':
        mode = 'converge'
    iters = int(c['ITER'])
    if iters == 0:
        Path('history.csv').write_text('"Inner_Iter","rms[Rho]","rms[RhoU]","rms[RhoV]","rms[RhoE]","Nonphysical_Points"\n')
        Path(c['VOLUME_FILENAME']+'.vtu').write_text('<VTKFile fake/>\n')
        print('Exit Success (SU2_CFD)', flush=True)
        return
    pts, _, _ = read_su2(Path(c['MESH_FILENAME']))
    g, R = float(c['GAMMA_VALUE']), float(c['GAS_CONSTANT'])
    p, T, M = float(c['FREESTREAM_PRESSURE']), float(c['FREESTREAM_TEMPERATURE']), float(c['MACH_NUMBER'])
    a = math.radians(float(c['AOA']))
    rho = p/(R*T)
    V = M*math.sqrt(g*R*T)
    u, v = V*math.cos(a), V*math.sin(a)
    E = p/(g-1)+.5*rho*V*V

    fields = ['x', 'y', 'Density', 'Momentum_x', 'Momentum_y', 'Energy']
    if c['WRT_RESTART_COMPACT'] == 'NO':
        fields.append('Pressure')
    rows = [(x, y, rho, rho*u, rho*v, E, p)[:len(fields)] for x, y in pts]

    def restart(name, final=False, it=None):
        due = (lambda kind: kind in freq and (final or it % freq[kind] == 0))
        if due('RESTART'):
            write_dat(name+'.dat', fields, rows)
        if due('RESTART_ASCII'):
            with open(name+'.csv', 'w') as f:
                f.write(','.join(f'"{n}"' for n in ['PointID']+fields)+'\n')
                for i, r in enumerate(rows):
                    f.write(f'{i}'+''.join(f', {z:.15e}' for z in r)+'\n')
        if due('PARAVIEW'):
            Path(name.replace('restart_flow', 'flow')+'.vtu').write_text('<VTKFile fake/>\n')
    with open('history.csv', 'w') as h:
        h.write('"Inner_Iter","rms[Rho]","rms[RhoU]","rms[RhoV]","rms[RhoE]","Nonphysical_Points"\n')
        for it in range(iters):
            if mode in ('crash', 'crash-first') and it == 50:
                print('Segmentation fault (fake)', flush=True)
                sys.exit(1)
            drop = 2*min(it/4000, 1) if mode == 'stall' else 7*min(it/4000, 1)
            r = -1-drop
            npnt = 5 if mode == 'nonphysical' and it >= 1600 else 0
            h.write(f'{it}, {r:.6f}, {r:.6f}, {r:.6f}, {r:.6f}, {npnt}\n')
            if it % 200 == 0:
                h.flush()
            if any((it+1) % f == 0 for f in freq.values()) and it+1 < iters:
                h.flush()
                restart(f'restart_flow_{it+1:06d}', it=it+1)
                time.sleep(float(os.environ.get('FAKE_SU2_PAUSE', '0')))
    restart('restart_flow', final=True)
    print('Exit Success (SU2_CFD)', flush=True)


if __name__ == '__main__':
    main()
