"""Resolve the starting MPI count without changing explicit run policies."""
import argparse
import json
import os
import subprocess
import sys


def default_mpi_settings(profile=None):
    profile = profile or os.environ.get('SU2_MACHINE_PROFILE', 'auto')
    if profile not in ('auto', 'macbook', 'macmini'):
        raise ValueError('SU2_MACHINE_PROFILE must be auto, macbook, or macmini')
    source = 'explicit_profile'
    if profile == 'auto':
        profile, source = 'unknown', 'fallback'
        if sys.platform == 'darwin':
            try:
                result = subprocess.run(
                    ['/usr/sbin/system_profiler', 'SPHardwareDataType', '-json'],
                    capture_output=True, text=True, check=True, timeout=10)
                name = json.loads(result.stdout)['SPHardwareDataType'][0]['machine_name'].lower()
                if 'macbook' in name:
                    profile, source = 'macbook', 'hardware_model'
                elif name == 'mac mini':
                    profile, source = 'macmini', 'hardware_model'
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                pass
    available = os.cpu_count()
    # Unknown CPU capacity must not imply permission to start twelve processes.
    requested = 12 if profile == 'macmini' else 4
    ranks = min(requested, available or 1)
    return dict(profile=profile, source=source, ranks=ranks, available_cpus=available)


def default_mpi_ranks():
    return default_mpi_settings()['ranks']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('auto', 'macbook', 'macmini'))
    args = parser.parse_args()
    print(json.dumps(default_mpi_settings(args.profile)))
