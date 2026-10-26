"""Sweep finalization on fake-solver runs: adopted fields kept, regenerable files restored."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intake_sweep as S  # noqa: E402
import sweep_retention as R  # noqa: E402
from test_intake_sweep import config  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SweepRetentionTests(unittest.TestCase):
    def sweep(self, tmp, **extra):
        repo = Path(tmp)/'repo'
        (repo/'local').mkdir(parents=True)
        (repo/'local'/'output-retention-policy.json').write_text(json.dumps(dict(protected_stages=[], in_progress_stages=[])))
        cfgp = config(tmp, output_root=str(repo/'results'), sweep=dict(mach=[6], aoa_deg=[0, -2]), **extra)
        (Path(tmp)/'fake.json').write_text(json.dumps({'m06_a-2:*': 'stall'}))
        env = dict(os.environ, FAKE_SU2_PLAN=str(Path(tmp)/'fake.json'), MPLCONFIGDIR=str(Path(tmp)/'mpl'))
        py = [sys.executable, str(HERE/'intake_sweep.py')]
        subprocess.run(py+['prepare', str(cfgp)], check=True, env=env, capture_output=True)
        run = repo/'results'/'t-ramp2'
        out = subprocess.run(py+['submit', str(run)], env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        return repo.resolve(), run.resolve(), env, py

    def test_compact_run_keeps_adopted_fields_and_restores_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, run, env, py = self.sweep(tmp)
            case = run/'cases'/'m06_a+0'
            result = json.loads((case/'result.json').read_text())
            adopted = run/result['checkpoint']
            history = sha(case/'second'/'history.csv')
            plan = R.plan_run(run, repo=repo)
            self.assertEqual(plan['skipped_cases'], {})
            deleted = {a['path'] for a in plan['actions'] if a['action'] == 'delete'}
            self.assertIn('cases/m06_a+0/second/restart_flow_002000.dat', deleted)
            self.assertNotIn(result['checkpoint'], deleted)
            self.assertIn('cases/m06_a+0/first/flow.vtu', deleted)
            rec = R.apply_run(run, review_complete=True, repo=repo, check_processes=False)
            self.assertEqual(rec['status'], 'applied')
            self.assertTrue(adopted.is_file())
            self.assertEqual(sha(adopted), result['checkpoint_sha256'])
            self.assertTrue((case/'first'/'restart_flow.dat').is_file())
            self.assertTrue((case/'volume'/'flow.vtu').is_file())
            self.assertEqual(sorted(p.name for p in case.rglob('*.vtu')), ['flow.vtu'])
            self.assertFalse((case/'second'/'history.csv').exists())
            self.assertTrue((case/'second'/'seed_restart.dat').samefile(case/'first'/'restart_flow.dat'))
            # unconverged case: the diagnostic checkpoint and its continuation seed stay
            stall = json.loads((run/'cases'/'m06_a-2'/'result.json').read_text())
            self.assertTrue((run/stall['checkpoint']).is_file())
            for a in json.loads((run/'cases'/'m06_a-2'/'state.json').read_text())['attempts']:
                if a.get('seed'):
                    self.assertTrue(Path(a['seed']).is_file(), a['seed'])
            # figures still build from the compressed histories
            out = subprocess.run(py+['aggregate', str(run)], env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            again = R.plan_run(run, repo=repo)
            self.assertEqual([a for a in again['actions'] if a['action'] != 'hardlink-if-identical'], [])
            restored = R.restore_run(run)
            self.assertIn('cases/m06_a+0/second/history.csv', restored['restored'])
            self.assertEqual(sha(case/'second'/'history.csv'), history)
            with gzip.open(case/'second'/'history.csv.gz') as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), history)

    def test_legacy_run_drops_regenerable_csv_and_restores_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, run, env, py = self.sweep(tmp, output=dict(S.LEGACY_OUTPUT))
            case = run/'cases'/'m06_a+0'
            result = json.loads((case/'result.json').read_text())
            it = int(Path(result['checkpoint']).stem.rsplit('_', 1)[1])
            R.apply_run(run, review_complete=True, repo=repo, check_processes=False)
            self.assertEqual(list(case.rglob('*.csv')), [])
            self.assertTrue((case/'second'/f'restart_flow_{it:06d}.dat').is_file())
            self.assertEqual(sorted(p.name for p in case.rglob('*.vtu')), [f'flow_{it:06d}.vtu'])
            restored = R.restore_run(run)
            self.assertIn(result['checkpoint'], restored['restored'])
            self.assertIn('cases/m06_a+0/second/seed_restart.csv', restored['restored'])
            self.assertEqual(sha(run/result['checkpoint']), result['checkpoint_sha256'])
            self.assertTrue(restored['not_restorable'])

    def test_unfinished_protected_or_stale_cases_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, run, env, py = self.sweep(tmp)
            policy = repo/'local'/'output-retention-policy.json'
            policy.write_text(json.dumps(dict(protected_stages=[], in_progress_stages=[str((run/'cases'/'m06_a-2').relative_to(repo))])))
            st = run/'cases'/'m06_a+0'/'state.json'
            state = json.loads(st.read_text())
            st.write_text(json.dumps(dict(state, status='analysing')))
            plan = R.plan_run(run, repo=repo)
            self.assertEqual(set(plan['skipped_cases']), {'m06_a+0', 'm06_a-2'})
            self.assertEqual(plan['actions'], [])
            st.write_text(json.dumps(dict(state, updated_unix=state['updated_unix']+1e6)))
            self.assertIn('after aggregation', R.plan_run(run, repo=repo)['skipped_cases']['m06_a+0'])
            with self.assertRaises(ValueError):
                R.apply_run(run, repo=repo, check_processes=False)
            (run/'analysis'/'lip-results.csv').write_text('changed\n')
            with self.assertRaises(ValueError):
                R.plan_run(run, repo=repo)


if __name__ == '__main__':
    unittest.main()
