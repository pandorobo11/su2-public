"""Remove ASCII restart CSVs that regenerate byte-identically from the binary restart."""
import argparse
import array
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

RECEIPT = 'restart-csv-retention.json'
NAME = re.compile(r'restart_flow(_\d+)?\.csv')
MAGIC = 535532


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def csv_from_dat(data):
    """SU2 binary restart: int32[5] header, 33-byte field names, float64 values per point."""
    magic, nvar, npoint, _, _ = struct.unpack('<5i', data[:20])
    if magic != MAGIC:
        raise ValueError('Not an SU2 binary restart')
    names = [data[20+33*i:20+33*(i+1)].split(b'\0')[0].decode() for i in range(nvar)]
    offset = 20 + 33*nvar
    values = array.array('d')
    values.frombytes(data[offset:offset+8*nvar*npoint])
    row = '%d' + ', %.15e'*nvar + '\n'
    lines = [','.join(f'"{n}"' for n in ['PointID']+names) + '\n']
    lines += [row % ((i,)+tuple(values[i*nvar:(i+1)*nvar])) for i in range(npoint)]
    return ''.join(lines).encode()


def preserved(repo):
    policy = json.loads((repo/'local/output-retention-policy.json').read_text())
    return [repo/p for p in policy['protected_stages'] + policy.get('in_progress_stages', [])]


def check_directory(directory, names, apply=False):
    directory = Path(directory)
    entries = []
    for name in sorted(names):
        csv, dat = directory/name, directory/(name[:-4]+'.dat')
        entry = {'csv': name, 'dat': dat.name}
        try:
            if csv.is_symlink() or dat.is_symlink():
                raise ValueError('Symlink restart')
            text, binary = csv.read_bytes(), dat.read_bytes()
            same = csv_from_dat(binary) == text
        except Exception as error:
            entry.update(status='kept-error', error=repr(error))
            entries.append(entry)
            continue
        entry.update(csv_sha256=sha256(text), csv_size=len(text), csv_mtime_ns=csv.stat().st_mtime_ns,
                     dat_sha256=sha256(binary), status='verified' if same else 'kept-mismatch')
        entries.append(entry)
    verified = [e for e in entries if e['status'] == 'verified']
    if apply and verified:
        receipt = directory/RECEIPT
        previous = json.loads(receipt.read_text())['entries'] if receipt.is_file() else []
        removed = {e['csv'] for e in verified}
        record = {'schema': 'su2-restart-csv-retention-v1', 'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'note': 'CSV removed after byte-identical regeneration from DAT; recreate with scripts/restart_csv.py regen',
                  'entries': [e for e in previous if e['csv'] not in removed] + [dict(e, status='removed') for e in verified]}
        temporary = receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=1))
        os.replace(temporary, receipt)
        for entry in verified:
            p = directory/entry['csv']
            st = p.stat()
            if st.st_size == entry['csv_size'] and st.st_mtime_ns == entry['csv_mtime_ns']:
                p.unlink()
                entry['status'] = 'removed'
            else:
                entry['status'] = 'kept-changed'
    return str(directory), entries


def prune(root, *, apply=False, workers=4, repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    root = Path(root).absolute()
    if root.resolve() != root or not (root == repo/'results' or root.is_relative_to(repo/'results')):
        raise ValueError('Target must be a real directory inside results')
    if apply:
        processes = subprocess.check_output(['ps', '-axo', 'pid,comm'], text=True)
        if re.search(r'(?:^|\s|/)(?:SU2_CFD(?:_AD)?|mpirun|mpiexec)(?:\s|$)', processes):
            raise ValueError('SU2 or MPI is running; no cleanup')
    skip = preserved(repo)
    jobs = {}
    for directory, _, files in os.walk(root):
        d = Path(directory)
        if any(d == s or d.is_relative_to(s) for s in skip):
            continue
        names = [f for f in files if NAME.fullmatch(f) and (d/(f[:-4]+'.dat')).is_file()]
        if names:
            jobs[d] = names
    entries = []
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for d, result in pool.map(check_directory, jobs, jobs.values(), [apply]*len(jobs)):
            entries += [dict(e, dir=str(Path(d).relative_to(repo))) for e in result]
    status = {}
    for e in entries:
        status[e['status']] = status.get(e['status'], 0) + 1
    done = 'removed' if apply else 'verified'
    return {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'root': str(root.relative_to(repo)), 'applied': apply,
            'status': status, 'bytes': sum(e['csv_size'] for e in entries if e['status'] == done),
            'entries': entries}


def regenerate(root):
    restored = []
    for directory, _, files in os.walk(root):
        if RECEIPT not in files:
            continue
        d = Path(directory)
        for entry in json.loads((d/RECEIPT).read_text())['entries']:
            csv = d/entry['csv']
            if csv.exists():
                continue
            binary = (d/entry['dat']).read_bytes()
            if sha256(binary) != entry['dat_sha256']:
                raise ValueError(f'DAT changed after CSV removal: {d/entry["dat"]}')
            text = csv_from_dat(binary)
            if sha256(text) != entry['csv_sha256']:
                raise ValueError(f'Regenerated CSV differs: {csv}')
            csv.write_bytes(text)
            os.utime(csv, ns=(entry['csv_mtime_ns'], entry['csv_mtime_ns']))
            restored.append(str(csv))
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prune', help='verify (default) or remove regenerable restart CSVs')
    p.add_argument('root')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--manifest', type=Path)
    r = sub.add_parser('regen', help='recreate removed CSVs from their receipts')
    r.add_argument('root')
    args = parser.parse_args()
    if args.command == 'prune':
        result = prune(args.root, apply=args.apply, workers=args.workers)
        if args.manifest:
            args.manifest.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ('root', 'applied', 'status', 'bytes')}))
        for e in result['entries']:
            if e['status'] not in ('verified', 'removed'):
                print(e['status'], e['dir'], e['csv'])
    else:
        print(f'regenerated {len(regenerate(args.root))} CSV files')


if __name__ == '__main__':
    main()
