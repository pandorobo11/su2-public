#!/usr/bin/env python3
"""Finalize a reviewed intake sweep run (scripts/intake_sweep.py) to its adopted fields.

  plan     list what would be removed, linked or compressed (default; writes nothing)
  apply    do it, after --review-complete, writing sweep-retention.json first
  restore  regenerate the removed restart CSVs from their DATs and decompress histories

Per finished case it keeps the adopted checkpoint DAT (result.json), the qualified, last healthy
and last numbered checkpoint DAT of every stage, each stage's final restart_flow.dat (a hard link
when it equals a kept checkpoint), one VTU, and every JSON, cfg, log, gate, plot and mesh file.
It removes the other numbered DAT and VTU, restart CSVs that regenerate byte-identically from a
kept DAT (seed copies included), and gzips history.csv. Cases that are not done, archive-*
folders and stages listed in local/output-retention-policy.json are left untouched.
See docs/operations/su2-data-retention.md.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from restart_csv import csv_from_dat

RECEIPT = 'sweep-retention.json'
SCHEMA = 'su2-sweep-retention-v1'
NUMBERED = re.compile(r'restart_flow_(\d+)\.(dat|csv)')
VTU = re.compile(r'flow_(\d+)\.vtu')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def busy():
    ps = subprocess.check_output(['ps', '-axo', 'pid,comm'], text=True)
    return re.search(r'(?:^|\s|/)(?:SU2_CFD(?:_AD)?|SU2_SOL|mpirun|mpiexec)(?:\s|$)', ps) is not None


def preserved(repo):
    p = repo/'local'/'output-retention-policy.json'
    if not p.is_file():
        return []
    policy = load(p)
    return [repo/s for s in policy.get('protected_stages', [])+policy.get('in_progress_stages', [])]


def iteration(name):
    m = NUMBERED.fullmatch(name)
    return int(m[1]) if m else None


def stage_of(run, relative):
    return (run/relative).parent if relative else None


def check_run(run, repo):
    """The run must be a real directory under results/ whose aggregation is current."""
    run = Path(run).absolute()
    if run.resolve() != run or not run.is_relative_to(repo/'results'):
        raise ValueError('Run must be a real directory inside results/')
    plan, manifest = run/'plan.json', run/'analysis'/'manifest.json'
    if not plan.is_file() or not manifest.is_file():
        raise ValueError('Run has no plan.json or aggregated analysis/manifest.json')
    m = load(manifest)
    if m.get('plan_sha256') != sha(plan):
        raise ValueError('Aggregation refers to a different plan')
    for name, digest in m.get('files', {}).items():
        if sha(run/'analysis'/name) != digest:
            raise ValueError(f'Aggregated table changed after aggregation: {name}')
    return run, load(plan), m


def case_actions(run, cid, state, aggregated_unix, skip):
    """Actions for one done case: list of dict(action, path, ...); paths relative to the run."""
    cdir = run/'cases'/cid
    if any(cdir == s or cdir.is_relative_to(s) or s.is_relative_to(cdir) for s in skip):
        return None, 'protected or in progress (output-retention-policy.json)'
    if state.get('status') != 'done':
        return None, f'case status {state.get("status")}'
    if (state.get('updated_unix') or 0) > aggregated_unix:
        return None, 'case changed after aggregation'
    result = load(cdir/'result.json') if (cdir/'result.json').is_file() else {}
    adopted = result.get('checkpoint')
    stages = [cdir/a['name'] for a in state.get('attempts', []) if (cdir/a['name']).is_dir()]
    keep_iter, healthy_vtu = {}, {}
    for st in stages:
        rec = load(st/'run.json') if (st/'run.json').is_file() else {}
        its = {iteration(p.name) for p in st.iterdir() if iteration(p.name) is not None}
        mine = {iteration(Path(r).name) for r in (rec.get('qualified_checkpoint'), rec.get('last_healthy_checkpoint'), adopted)
                if r and stage_of(run, r) == st}
        keep_iter[st] = (mine | ({max(its)} if its else set())) - {None}
        healthy_vtu[st] = {iteration(Path(r).name) for r in (rec.get('last_healthy_checkpoint'),) if r and stage_of(run, r) == st}
        healthy_vtu[st] |= {max(its)} if its else set()
    vtu_keep = choose_vtu(run, cdir, adopted, stages, healthy_vtu)
    kept_dat = {p for st in stages for p in {st/f'restart_flow_{i:06d}.dat' for i in keep_iter[st]} | {st/'restart_flow.dat'}
                if p.is_file() and not p.is_symlink()}

    def rel(p):
        return str(p.relative_to(run))
    acts = []
    for st in stages:
        rec = load(st/'run.json') if (st/'run.json').is_file() else {}
        for p in sorted(st.iterdir()):
            if p.is_symlink() or not p.is_file():
                continue
            i = iteration(p.name)
            if p.suffix == '.dat' and i is not None and i not in keep_iter[st]:
                acts.append(dict(action='delete', path=rel(p)))
            elif p.suffix == '.csv' and (i is not None or p.name == 'restart_flow.csv'):
                dat = p.with_suffix('.dat')
                if dat in kept_dat:
                    acts.append(dict(action='delete-regenerable', path=rel(p), dat=rel(dat)))
                elif i is not None and i not in keep_iter[st]:
                    acts.append(dict(action='delete', path=rel(p)))
            elif p.name == 'seed_restart.csv' and rec.get('seed'):
                dat = (run/rec['seed']).with_suffix('.dat')
                if dat in kept_dat:
                    acts.append(dict(action='delete-regenerable', path=rel(p), dat=rel(dat)))
            elif p.name == 'seed_restart.dat' and rec.get('seed') and (run/rec['seed']).is_file():
                if not p.samefile(run/rec['seed']):
                    acts.append(dict(action='hardlink', path=rel(p), target=rec['seed']))
            elif p.suffix == '.vtu' and p not in vtu_keep:
                acts.append(dict(action='delete', path=rel(p)))
            elif p.name == 'history.csv':
                acts.append(dict(action='gzip', path=rel(p)))
        final = st/'restart_flow.dat'
        twins = [st/f'restart_flow_{i:06d}.dat' for i in sorted(keep_iter[st])]
        if final.is_file() and twins and twins[-1].is_file() and not final.samefile(twins[-1]):
            acts.append(dict(action='hardlink-if-identical', path=rel(final), target=rel(twins[-1])))
    return acts, None


def choose_vtu(run, cdir, adopted, stages, healthy_vtu):
    """One VTU of the adopted checkpoint; without one, those of the last and last healthy checkpoints."""
    volume = cdir/'volume'/'flow.vtu'
    if adopted:
        st, i = stage_of(run, adopted), iteration(Path(adopted).name)
        if volume.is_file():
            return {volume}
        numbered = st/f'flow_{i:06d}.vtu'
        if numbered.is_file():
            return {numbered}
        same = (st/'restart_flow.dat').is_file() and (run/adopted).with_suffix('.dat').is_file() and \
            sha(st/'restart_flow.dat') == sha((run/adopted).with_suffix('.dat'))
        return {st/'flow.vtu'} if same and (st/'flow.vtu').is_file() else set()
    keep = {volume} if volume.is_file() else set()
    for st in stages:
        keep |= {st/f'flow_{i:06d}.vtu' for i in healthy_vtu[st]} | {st/'flow.vtu'}
    return keep


def regenerable(run, act):
    """Hash the CSV and its DAT; the CSV qualifies only when it regenerates byte for byte."""
    csv_, dat = run/act['path'], run/act['dat']
    text, binary = csv_.read_bytes(), dat.read_bytes()
    try:
        same = csv_from_dat(binary) == text
    except ValueError:
        same = False
    return dict(act, bytes=len(text), sha256=hashlib.sha256(text).hexdigest(),
                dat_sha256=hashlib.sha256(binary).hexdigest(), regenerates=same)


def plan_run(run, *, repo=None, verify=False, workers=4):
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    run, plan, manifest = check_run(run, repo)
    skip = preserved(repo)
    acts, skipped = [], {}
    for cid in plan['order']:
        sp = run/'cases'/cid/'state.json'
        state = load(sp) if sp.is_file() else dict(status='not_started')
        a, why = case_actions(run, cid, state, manifest['created_unix'], skip)
        if a is None:
            skipped[cid] = why
        else:
            acts += a
    for a in acts:
        a['bytes'] = (run/a['path']).stat().st_size
    if verify:
        checks = [a for a in acts if a['action'] == 'delete-regenerable']
        with concurrent.futures.ProcessPoolExecutor(workers) as pool:
            done = dict(zip(map(id, checks), pool.map(regenerable, [run]*len(checks), checks)))
        acts = [done.get(id(a), a) for a in acts]
        acts = [a for a in acts if a.get('regenerates', True)]
    freed = sum(a['bytes'] for a in acts if a['action'].startswith(('delete', 'hardlink')))
    return dict(schema=SCHEMA, run=str(run.relative_to(repo)), plan_sha256=sha(run/'plan.json'),
                analysis_manifest_sha256=sha(run/'analysis'/'manifest.json'), verified=verify,
                skipped_cases=skipped, actions=acts, bytes_freed_before_gzip=freed,
                history_bytes=sum(a['bytes'] for a in acts if a['action'] == 'gzip'))


def same_bytes(a, b):
    return a.stat().st_size == b.stat().st_size and sha(a) == sha(b)


def apply_run(run, *, review_complete=False, repo=None, workers=4, check_processes=True):
    if review_complete is not True:
        raise ValueError('Review completion must be declared with --review-complete')
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    if check_processes and busy():
        raise ValueError('SU2 or MPI is running; no cleanup')
    rec = plan_run(run, repo=repo, verify=True, workers=workers)
    run = repo/rec['run']
    receipt = run/RECEIPT
    previous = []
    if receipt.exists():
        old = load(receipt)
        if old.get('status') != 'applied':
            raise ValueError('An interrupted cleanup left sweep-retention.json; review it first')
        previous = old.get('previous', [])+[{k: old[k] for k in old if k != 'previous'}]
    for a in rec['actions']:
        if a['action'] in ('delete', 'gzip', 'hardlink', 'hardlink-if-identical') and 'sha256' not in a:
            a['sha256'] = sha(run/a['path'])
    rec.update(time=time.strftime('%Y-%m-%dT%H:%M:%S'), status='applying', review_complete=True, previous=previous)
    tmp = receipt.with_name(RECEIPT+'.tmp')
    tmp.write_text(json.dumps(rec, indent=1)+'\n')
    os.replace(tmp, receipt)
    for a in rec['actions']:
        p = run/a['path']
        if sha(p) != a['sha256']:
            a['status'] = 'kept-changed'
            continue
        if a['action'] in ('delete', 'delete-regenerable'):
            if a['action'] == 'delete-regenerable' and sha(run/a['dat']) != a['dat_sha256']:
                a['status'] = 'kept-changed'
                continue
            p.unlink()
        elif a['action'] == 'gzip':
            gz = p.with_name(p.name+'.gz')
            with p.open('rb') as src, gzip.GzipFile(gz, 'wb', compresslevel=6, mtime=0) as dst:
                for block in iter(lambda: src.read(1 << 20), b''):
                    dst.write(block)
            check = hashlib.sha256()
            with gzip.open(gz, 'rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    check.update(block)
            if check.hexdigest() != a['sha256']:
                gz.unlink()
                raise ValueError(f'gzip round trip differs: {p}')
            st = p.stat()
            os.utime(gz, ns=(st.st_atime_ns, st.st_mtime_ns))
            a['gz_bytes'] = gz.stat().st_size
            p.unlink()
        else:  # replace an identical copy by a hard link to the kept file
            target = run/a['target']
            if not same_bytes(p, target):
                a['status'] = 'kept-different'
                continue
            link = p.with_name(p.name+'.link-tmp')
            os.link(target, link)
            os.replace(link, p)
        a['status'] = 'done'
    rec['status'] = 'applied'
    rec['bytes_freed'] = sum(a['bytes'] for a in rec['actions'] if a.get('status') == 'done' and a['action'] != 'gzip') + \
        sum(a['bytes']-a['gz_bytes'] for a in rec['actions'] if a.get('status') == 'done' and a['action'] == 'gzip')
    tmp.write_text(json.dumps(rec, indent=1)+'\n')
    os.replace(tmp, receipt)
    return rec


def restore_run(run):
    """Regenerate removed CSVs and decompressed histories; deleted DAT and VTU cannot come back."""
    run = Path(run).absolute()
    rec = load(run/RECEIPT)
    restored, missing = [], []
    for r in rec.get('previous', [])+[rec]:
        for a in r['actions']:
            p = run/a['path']
            if a.get('status') != 'done' or p.exists():
                continue
            if a['action'] == 'delete-regenerable':
                binary = (run/a['dat']).read_bytes()
                if hashlib.sha256(binary).hexdigest() != a['dat_sha256']:
                    raise ValueError(f'DAT changed after CSV removal: {a["dat"]}')
                text = csv_from_dat(binary)
                if hashlib.sha256(text).hexdigest() != a['sha256']:
                    raise ValueError(f'Regenerated CSV differs: {p}')
                p.write_bytes(text)
                restored.append(a['path'])
            elif a['action'] == 'gzip':
                with gzip.open(p.with_name(p.name+'.gz'), 'rb') as f:
                    data = f.read()
                if hashlib.sha256(data).hexdigest() != a['sha256']:
                    raise ValueError(f'Decompressed history differs: {p}')
                p.write_bytes(data)
                restored.append(a['path'])
            elif a['action'] == 'delete':
                missing.append(a['path'])
    return dict(restored=restored, not_restorable=missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('run')
    p.add_argument('--verify', action='store_true', help='hash files and check CSV regeneration (slow)')
    p.add_argument('--manifest', type=Path, help='write the full plan here')
    a = sub.add_parser('apply')
    a.add_argument('run')
    a.add_argument('--review-complete', action='store_true', required=True)
    a.add_argument('--workers', type=int, default=4)
    for q in (p, a):
        q.add_argument('--repo', type=Path, help='repository whose results/ and local/ apply (default: this checkout)')
    r = sub.add_parser('restore')
    r.add_argument('run')
    args = ap.parse_args()
    if args.command == 'plan':
        out = plan_run(args.run, repo=args.repo, verify=args.verify)
        if args.manifest:
            args.manifest.write_text(json.dumps(out, indent=1)+'\n')
        counts = {}
        for x in out['actions']:
            counts[x['action']] = counts.get(x['action'], 0)+1
        print(json.dumps(dict(run=out['run'], actions=counts, skipped_cases=out['skipped_cases'],
                              gib_freed_before_gzip=round(out['bytes_freed_before_gzip']/2**30, 2),
                              history_gib=round(out['history_bytes']/2**30, 2)), indent=1))
    elif args.command == 'apply':
        out = apply_run(args.run, review_complete=args.review_complete, repo=args.repo, workers=args.workers)
        print(json.dumps(dict(run=out['run'], status=out['status'], gib_freed=round(out['bytes_freed']/2**30, 2),
                              skipped_cases=out['skipped_cases']), indent=1))
    else:
        print(json.dumps(restore_run(args.run), indent=1))


if __name__ == '__main__':
    main()
