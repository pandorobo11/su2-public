"""Intake sweep pipeline: theory, mesh scaling, retry ladder and an end-to-end fake-solver run."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intake_sweep as S  # noqa: E402
import intake_sweep_theory as T  # noqa: E402
import intake_structured_mesh as ism  # noqa: E402
from intake_euler import settings as read_settings  # noqa: E402


def config(directory, **extra):
    cfg = dict(name='t-ramp2', output_root='runs',
               design=dict(mach=6., temperature_ratio=1.8, stages=2, capture_height_m=.35, contraction_ratio=1.2,
                           duct_length_m=.4, cowl_thickness_m=.02, cowl_nose_length_m=.1, upstream_length_m=.2),
               sweep=dict(mach=[6, 7], aoa_deg=[-2, 0]), mesh=dict(length_scale=4.),
               # the fake uniform field crosses the walls, so its open-boundary balance cannot close
               gate=dict(imbalance=10.),
               stages=dict(first_iterations=200, second_iterations=8000, checkpoint_period=2000,
                           max_stage_seconds=600, case_budget_seconds=3000),
               execution=dict(backend='local', ranks=1, su2_cfd=str(HERE/'intake_sweep_fake_su2.py'),
                              mpirun=[sys.executable], python=sys.executable, local_exclusive=False))
    for k, v in extra.items():
        cfg[k] = dict(cfg.get(k, {}), **v) if isinstance(v, dict) else v
    path = Path(directory)/'sweep.json'
    path.write_text(json.dumps(cfg))
    return path


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


class TheoryTests(unittest.TestCase):
    def test_design_point_returns_design_temperature_ratio(self):
        import intake_design
        d = intake_design.design(9., 2.5, 4, capture_height=1.)
        corners = [(0., 0.)]+[tuple(v) for v in d['geometry']['ramp_vertices_m'][:-1]]
        r = T.lip_theory(9., 0., corners, [x['angle_deg'] for x in d['geometry']['ramps']], d['geometry']['cowl_lip_m'])
        self.assertAlmostEqual(r['theory_T_ratio'], 2.5, places=9)
        self.assertAlmostEqual(r['theory_p0_recovery'], d['performance']['total_pressure_recovery'], places=9)
        self.assertFalse(r['theory_waves_cross_ahead_of_lip'])

    def test_positive_aoa_weakens_compression_and_kantrowitz_is_monotone(self):
        args = ([(0., 0.), (1., .1)], [5.7, 11.], (2., .5))
        lo, hi = T.lip_theory(8., -4., *args), T.lip_theory(8., 4., *args)
        self.assertGreater(lo['theory_T_ratio'], hi['theory_T_ratio'])
        self.assertLess(T.kantrowitz_contraction(4.), T.kantrowitz_contraction(6.))
        self.assertAlmostEqual(T.kantrowitz_contraction(1.), 1., places=9)


class MeshScaleTests(unittest.TestCase):
    def test_scaled_geometry_gives_the_same_relative_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = dict(kind='intake', mach=6., temperature=220., pressure=2500., outlet_x=1.6, upstream_length=.2,
                        cowl_thickness=.01, cowl_nose_length=.05, cowl_lip_height=.16,
                        ramps=[dict(angle_deg=5., length=.5), dict(angle_deg=10., length=.4)], output='o')
            meshes = []
            for k, f in enumerate((1., 3.)):
                raw = dict(base, outlet_x=1.6*f, upstream_length=.2*f, cowl_thickness=.01*f, cowl_nose_length=.05*f,
                           cowl_lip_height=.16*f, ramps=[dict(r, length=r['length']*f) for r in base['ramps']])
                p = Path(tmp)/f'i{k}.json'
                p.write_text(json.dumps(raw))
                s = read_settings(p)
                pts, q, mk, meta = ism.mesh(s, ism.rule_columns(s, f), length_scale=f)
                ism.check(pts, q, mk, s)
                meshes.append(pts/f)
            self.assertEqual(meshes[0].shape, meshes[1].shape)
            self.assertLess(np.max(np.abs(meshes[0]-meshes[1])), 1e-9)


class LadderTests(unittest.TestCase):
    R = dict(max_second_order_attempts=4, delayed_limiter_iter=10000, retry_cfl=.5)

    def test_failure_then_delayed_then_lower_cfl(self):
        a = [dict(name='second', seed='f', limiter=1500, cfl=1., status='failed')]
        n = S.next_attempt(a, self.R)
        self.assertEqual((n['seed'], n['limiter'], n['cfl']), ('f', 10000, 1.))
        a.append(dict(name='second-2', seed='f', limiter=10000, cfl=1., status='failed', why=n['why']))
        n = S.next_attempt(a, self.R)
        self.assertEqual((n['limiter'], n['cfl']), (10000, .5))
        a.append(dict(name='second-3', seed='f', limiter=10000, cfl=.5, status='failed', why=n['why']))
        self.assertIsNone(S.next_attempt(a, self.R))

    def test_unconverged_continues_once_from_its_checkpoint(self):
        a = [dict(name='second', seed='f', limiter=1500, cfl=1., status='unconverged', last_checkpoint='c1')]
        n = S.next_attempt(a, self.R)
        self.assertEqual((n['seed'], n['limiter']), ('c1', 10000))
        a.append(dict(name='second-2', seed='c1', limiter=10000, cfl=1., status='unconverged', last_checkpoint='c2', why=n['why']))
        self.assertIsNone(S.next_attempt(a, self.R))

    def test_qualified_never_retries_and_budget_of_attempts_holds(self):
        self.assertIsNone(S.next_attempt([dict(status='qualified', limiter=1500, cfl=1., seed='f')], self.R))
        many = [dict(status='failed', limiter=1500, cfl=1., seed='f')]*4
        self.assertIsNone(S.next_attempt(many, self.R))


class ConfigTests(unittest.TestCase):
    def test_rejects_unknown_keys_and_contraction_below_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                S.read_config(config(tmp, design=dict(contraction_ratio=.9)))
            with self.assertRaises(ValueError):
                S.read_config(config(tmp, typo=1))
            cfg = S.read_config(config(tmp))
            self.assertTrue(cfg['run_dir'].endswith('runs/t-ramp2'))

    def test_pbs_script_carries_queue_resources_and_snapshot(self):
        plan = dict(order=['m09_a+0'], execution=dict(python='python3', setup=['module load su2'],
                    pbs=dict(job_prefix='isw', queue='calc_8p', resources=['select=1:ncpus=8:mpiprocs=8'],
                             extra_directives=[], analysis_queue=None, analysis_resources=['select=1:ncpus=1'])))
        text = S.job_script(plan, Path('/w/run'), 'm09_a+0')
        self.assertIn('#PBS -q calc_8p', text)
        self.assertIn('#PBS -l select=1:ncpus=8:mpiprocs=8', text)
        self.assertIn('/w/run/scripts/intake_sweep.py run-case m09_a+0 /w/run', text)
        self.assertIn('#PBS -N isw001', text)
        self.assertIn('aggregate /w/run', S.job_script(plan, Path('/w/run'), None, analysis=True))


class EndToEndTests(unittest.TestCase):
    """prepare -> local submit (fake SU2) -> ladder -> aggregate; checks tables and figures."""

    def test_sweep_with_fake_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfgp = config(tmp)
            plan = {'m06_a-2:second': 'nonphysical', 'm06_a-2:second-2': 'converge',
                    'm07_a+0:*': 'stall', 'm07_a-2:*': 'crash'}
            (Path(tmp)/'fake.json').write_text(json.dumps(plan))
            env = dict(os.environ, FAKE_SU2_PLAN=str(Path(tmp)/'fake.json'), MPLCONFIGDIR=str(Path(tmp)/'mpl'))
            py = [sys.executable, str(HERE/'intake_sweep.py')]
            subprocess.run(py+['prepare', str(cfgp)], check=True, env=env, capture_output=True)
            run = Path(tmp)/'runs'/'t-ramp2'
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(py+['prepare', str(cfgp)], check=True, env=env, capture_output=True)
            out = subprocess.run(py+['submit', str(run)], env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            rows = {r['case']: r for r in read(run/'analysis'/'lip-results.csv')}
            self.assertEqual(rows['m06_a+0']['outcome'], 'qualified')
            self.assertEqual(rows['m06_a+0']['use'], 'performance')
            self.assertAlmostEqual(float(rows['m06_a+0']['p0_recovery_massavg']), 1., places=6)
            self.assertEqual(rows['m06_a-2']['outcome'], 'qualified')
            self.assertEqual(rows['m06_a-2']['attempts'], '2')
            self.assertEqual(rows['m07_a+0']['outcome'], 'unconverged')
            self.assertEqual(rows['m07_a+0']['use'], 'diagnostic')
            self.assertEqual(rows['m07_a-2']['outcome'], 'failed')
            self.assertEqual(rows['m07_a-2']['use'], 'none')
            ex = read(run/'analysis'/'execution.csv')
            second = [e for e in ex if e['case'] == 'm06_a-2' and e['order'] == '2']
            self.assertEqual([e['limiter_iter'] for e in second], ['1500', '10000'])
            crash = [e for e in ex if e['case'] == 'm07_a-2' and e['order'] == '2']
            self.assertEqual([(e['limiter_iter'], e['cfl']) for e in crash], [('1500', '1'), ('10000', '1'), ('10000', '0.5')])
            iso = read(run/'analysis'/'isolator-sections.csv')
            self.assertTrue(any(r['station'] == 'OUT' for r in iso))
            for name in ('lip-vs-mach.png', 'lip-vs-aoa.png', 'theory-diff.png', 'outcome-map.png', 'geometry.png',
                         'kantrowitz-margin.png', 'residuals.png', 'isolator-vs-mach.png', 'streamwise-m6.png',
                         'lip-profiles-a+0.png', 'boundary-conditions.png', 'mesh.png'):
                self.assertTrue((run/'analysis'/'figures'/name).exists(), name)
            self.assertIn('nonphysical', (run/'cases'/'m06_a-2'/'diagnosis.md').read_text())
            fails = {r['case'] for r in read(run/'analysis'/'failures.csv')}
            self.assertEqual(fails, {'m07_a+0', 'm07_a-2'})
            # started cases are never rerun
            again = subprocess.run(py+['run-case', 'm06_a+0', str(run)], env=env, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('refusing to rerun', again.stderr+again.stdout)
            # a failed case can be moved aside and resubmitted on purpose
            subprocess.run(py+['archive-case', str(run), 'm07_a-2'], check=True, env=env, capture_output=True)
            self.assertTrue((run/'cases'/'m07_a-2'/'archive-1'/'state.json').exists())
            (Path(tmp)/'fake.json').write_text(json.dumps({}))
            out = subprocess.run(py+['submit', str(run), '--only', 'm07_a-2', '--resubmit'], env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            rows = {r['case']: r for r in read(run/'analysis'/'lip-results.csv')}
            self.assertEqual(rows['m07_a-2']['outcome'], 'qualified')
            cmp_ = subprocess.run(py+['compare', str(run), str(run), '--output', str(Path(tmp)/'cmp')], env=env,
                                  capture_output=True, text=True)
            self.assertEqual(cmp_.returncode, 0, cmp_.stderr[-2000:])
            self.assertTrue((Path(tmp)/'cmp'/'compare-lip-vs-aoa.png').exists())
            self.check_compact_output(run)

    def check_compact_output(self, run):
        from intake_euler import read_dat
        case = run/'cases'/'m06_a+0'
        second = case/'second'
        self.assertEqual(list(case.rglob('restart_flow*.csv')), [])
        self.assertEqual(sorted(p.name for p in second.glob('flow_*.vtu')), [])
        self.assertEqual(sorted(p.name for p in second.glob('restart_flow_*.dat')),
                         ['restart_flow_002000.dat', 'restart_flow_004000.dat', 'restart_flow_006000.dat'])
        cfg = S.read_cfg(second/'case.cfg')
        self.assertEqual((cfg['OUTPUT_FILES'], cfg['OUTPUT_WRT_FREQ'], cfg['WRT_RESTART_COMPACT'], cfg['READ_BINARY_RESTART']),
                         ('( RESTART, PARAVIEW )', '2000, 8000', 'YES', 'YES'))
        self.assertEqual(list(read_dat(second/'restart_flow_002000.dat')), ['x', 'y', 'Density', 'Momentum_x', 'Momentum_y', 'Energy'])
        seed, source = second/'seed_restart.dat', case/'first'/'restart_flow.dat'
        self.assertTrue(seed.samefile(source))
        self.assertEqual(json.loads((second/'run.json').read_text())['seed_link'], 'hardlink')
        result = json.loads((case/'result.json').read_text())
        self.assertTrue(result['checkpoint'].endswith('.dat'))
        volume = json.loads((case/'volume.json').read_text())
        self.assertEqual((volume['status'], volume['checkpoint']), ('written', result['checkpoint']))
        self.assertTrue((case/'volume'/'flow.vtu').is_file())
        self.assertFalse((case/'volume'/'adopted.dat').exists())


class LegacyOutputTests(unittest.TestCase):
    def test_legacy_output_keeps_ascii_restarts_and_checkpoint_vtu(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfgp = config(tmp, sweep=dict(mach=[6], aoa_deg=[0]), output=dict(S.LEGACY_OUTPUT))
            env = dict(os.environ, MPLCONFIGDIR=str(Path(tmp)/'mpl'))
            py = [sys.executable, str(HERE/'intake_sweep.py')]
            subprocess.run(py+['prepare', str(cfgp)], check=True, env=env, capture_output=True)
            run = Path(tmp)/'runs'/'t-ramp2'
            out = subprocess.run(py+['submit', str(run)], env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            second = run/'cases'/'m06_a+0'/'second'
            cfg = S.read_cfg(second/'case.cfg')
            self.assertEqual((cfg['OUTPUT_FILES'], cfg['OUTPUT_WRT_FREQ'], cfg['WRT_RESTART_COMPACT']),
                             ('( RESTART, RESTART_ASCII, PARAVIEW )', '2000, 2000, 2000', 'NO'))
            self.assertTrue((second/'seed_restart.csv').is_file())
            self.assertTrue((second/'flow_002000.vtu').is_file())
            result = json.loads((run/'cases'/'m06_a+0'/'result.json').read_text())
            self.assertTrue(result['checkpoint'].endswith('.csv'))
            self.assertFalse((run/'cases'/'m06_a+0'/'volume.json').exists())


class RestartReaderTests(unittest.TestCase):
    def test_binary_restart_matches_ascii_and_rejects_truncation(self):
        import restart_csv
        from intake_euler import read_csv, read_dat
        sys.path.insert(0, str(HERE))
        import intake_sweep_fake_su2 as F
        with tempfile.TemporaryDirectory() as tmp:
            dat, csv_ = Path(tmp)/'r.dat', Path(tmp)/'r.csv'
            F.write_dat(dat, ['x', 'y', 'Density'], [(0., 1., 1.25), (.5, -2., 1e-3)])
            csv_.write_bytes(restart_csv.csv_from_dat(dat.read_bytes()))
            a, b = read_dat(dat), read_csv(csv_)
            self.assertEqual(list(a), ['x', 'y', 'Density'])
            for k in a:
                np.testing.assert_array_equal(a[k], b[k])
            dat.write_bytes(dat.read_bytes()[:-8])
            with self.assertRaises(ValueError):
                read_dat(dat)


class CancelTests(unittest.TestCase):
    def test_sigterm_stops_the_solver_and_records_cancelled(self):
        import signal
        import time
        with tempfile.TemporaryDirectory() as tmp:
            cfgp = config(tmp, sweep=dict(mach=[6], aoa_deg=[0]))
            env = dict(os.environ, FAKE_SU2_PAUSE='2', MPLCONFIGDIR=str(Path(tmp)/'mpl'))
            py = [sys.executable, str(HERE/'intake_sweep.py')]
            subprocess.run(py+['prepare', str(cfgp)], check=True, env=env, capture_output=True)
            run = Path(tmp)/'runs'/'t-ramp2'
            job = subprocess.Popen(py+['run-case', 'm06_a+0', str(run)], env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            for _ in range(300):
                if (run/'cases'/'m06_a+0'/'second'/'restart_flow_002000.dat').exists():
                    break
                time.sleep(.1)
            job.send_signal(signal.SIGTERM)
            self.assertNotEqual(job.wait(timeout=60), 0)
            st = json.loads((run/'cases'/'m06_a+0'/'state.json').read_text())
            self.assertEqual(st['status'], 'cancelled')
            rec = json.loads((run/'cases'/'m06_a+0'/'second'/'run.json').read_text())
            self.assertEqual(rec['status'], 'cancelled')
            ps = subprocess.check_output(['ps', '-axo', 'command'], text=True)
            self.assertNotIn(str(run), ps)


if __name__ == '__main__':
    unittest.main()
