#!/usr/bin/env python3
"""Run a local SU2 benchmark manifest sequentially, preserving all evidence.

Input paths in the manifest are relative to the manifest directory. Run directories
must not exist: an existing result is never silently replaced. Use --resume to skip
only directories containing a finished measurement. RSS is a sampled process-tree
sum, not a simultaneous kernel high-water mark or unique physical memory.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_config(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.split('%', 1)[0].strip()
        if '=' in line:
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip()
    return result


def process_tree_rss(root):
    output = subprocess.check_output(
        ['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    processes = [tuple(map(int, row.split())) for row in output.splitlines()]
    members = {root}
    while True:
        added = {pid for pid, parent, _ in processes if parent in members}
        expanded = members | added
        if expanded == members:
            break
        members = expanded
    return sum(rss for pid, _, rss in processes if pid in members)


def history_metrics(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        rows = [{key.strip(): float(value) for key, value in row.items()}
                for row in reader]
    if not rows:
        return {}
    final = rows[-1]
    tail = rows[-100:]
    return {'iterations': int(final['Inner_Iter']) + 1, 'final': final,
            'finite_history': all(math.isfinite(x) for r in rows for x in r.values()),
            'last100_span': {key: max(r[key] for r in tail) - min(r[key] for r in tail)
                             for key in ('CL', 'CD') if key in final}}


def run(manifest, item, base, output):
    name = item['name']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be a single directory component')
    target = output / name
    target.mkdir(parents=True, exist_ok=False)
    case = manifest['cases'][item['case']]
    config_path = (base / case['config']).resolve()
    mesh_path = (base / case['mesh']).resolve()
    binary = (base / manifest['binaries'][item['binary']]).resolve()
    config = read_config(config_path)
    config.update({key: str(value) for key, value in item.get('overrides', {}).items()})
    config['MESH_FILENAME'] = str(mesh_path)
    (target / 'case.cfg').write_text(''.join(f'{key}= {value}\n' for key, value in config.items()))
    command = [manifest.get('mpirun', 'mpirun'), '--bind-to', 'none', '-np', str(item['ranks']),
               str(binary), 'case.cfg']
    result = {'name': name, 'case': item['case'], 'variant': item['variant'],
              'binary': item['binary'], 'ranks': item['ranks'], 'repeat': item['repeat'],
              'command': command, 'started_unix': time.time(), 'load_before': os.getloadavg(),
              'sha256': {label: sha256(path) for label, path in
                         [('binary', binary), ('mesh', mesh_path), ('config', target / 'case.cfg')]},
              'rss_sample_interval_s': manifest.get('sample_interval_s', 0.5)}
    start = time.perf_counter()
    rss = []
    timed_out = False
    with (target / 'solver.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=target, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                elapsed = time.perf_counter() - start
                rss.append([elapsed, process_tree_rss(proc.pid)])
                if elapsed > manifest.get('timeout_s', 600):
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                try:
                    proc.wait(timeout=result['rss_sample_interval_s'])
                except subprocess.TimeoutExpired:
                    pass
            proc.wait()
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    result.update(elapsed_s=time.perf_counter() - start, returncode=proc.returncode,
                  timed_out=timed_out, load_after=os.getloadavg(),
                  sampled_peak_tree_rss_mib=max((r[1] for r in rss), default=0)/1024)
    (target / 'rss.json').write_text(json.dumps(rss))
    log_text = (target / 'solver.log').read_text()
    result['solver_converged'] = 'All convergence criteria satisfied.' in log_text
    result['solver_success'] = 'Exit Success (SU2_CFD)' in log_text
    result.update(history_metrics(target / 'history.csv'))
    result['output_bytes'] = sum(p.stat().st_size for p in target.iterdir() if p.is_file())
    (target / 'measurement.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('name', 'elapsed_s', 'iterations',
          'solver_converged', 'returncode', 'sampled_peak_tree_rss_mib')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.resolve().parent
    output = (base / manifest['output']).resolve()
    for item in manifest['runs']:
        if args.resume and (output / item['name'] / 'measurement.json').exists():
            continue
        run(manifest, item, base, output)


if __name__ == '__main__':
    main()
