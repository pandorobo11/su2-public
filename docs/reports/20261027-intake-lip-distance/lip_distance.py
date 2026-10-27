"""Tabulate and plot the ramp-tip-to-lip horizontal distance of the theoretical
intake design (scripts/intake_design.py); does not launch a CFD calculation."""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'scripts'))
from intake_design import design  # noqa: E402

MACHS = [round(3 + 0.25*i, 2) for i in range(45)]  # 3.00 ... 14.00
RATIOS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
STAGES = range(1, 7)
GAMMA = 1.4

rows = []
for n in STAGES:
    for ratio in RATIOS:
        for mach in MACHS:
            try:
                result = design(mach, ratio, n, capture_height=1.0, gamma=GAMMA)
            except ValueError as error:
                rows.append(dict(stages=n, temperature_ratio=ratio, mach=mach, lip_x_over_capture_height='',
                                 first_shock_angle_deg='', total_pressure_recovery='', infeasible_reason=str(error)))
                continue
            x = result['geometry']['cowl_lip_m'][0]
            beta = result['stage_states'][0]['ray_angle_deg']
            # The lip lies on the first shock from the tip: x/h = cot(beta_1).
            assert abs(x*math.tan(math.radians(beta))-1) < 1e-9
            rows.append(dict(stages=n, temperature_ratio=ratio, mach=mach, lip_x_over_capture_height=f'{x:.6f}',
                             first_shock_angle_deg=f'{beta:.6f}',
                             total_pressure_recovery=f"{result['performance']['total_pressure_recovery']:.6f}",
                             infeasible_reason=''))
with (ROOT/'lip-distance.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def curve(n, ratio):
    sel = [r for r in rows if r['stages'] == n and r['temperature_ratio'] == ratio]
    return (np.array([r['mach'] for r in sel]),
            np.array([float(r['lip_x_over_capture_height'] or 'nan') for r in sel]))


def panels(groups, lines, label, title, cmap, name, key):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, group in zip(axes.flat, groups):
        for i, line in enumerate(lines):
            m, x = curve(*key(group, line))
            ax.plot(m, x, color=cmap(i/len(lines)), lw=1.8, label=label(line))
        ax.set_title(title(group))
        ax.set_yscale('log')
        ax.grid(alpha=.3, which='both')
    for ax in axes[1]:
        ax.set_xlabel('Freestream Mach number')
    for ax in axes[:, 0]:
        ax.set_ylabel('x_lip / h_capture')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Horizontal distance from ramp tip to cowl lip / capture height\n'
                 '(constant gamma=1.4, equal-strength shocks focused on the lip, T ratio after the ramps)')
    fig.tight_layout()
    fig.savefig(ROOT/'figures'/name, dpi=150)


(ROOT/'figures').mkdir(exist_ok=True)
panels(STAGES, RATIOS, lambda r: f'T/T_inf={r:g}', lambda n: f'{n} stage' + ('s' if n > 1 else ''),
       plt.get_cmap('viridis'), 'lip-distance-by-stage.png', lambda n, r: (n, r))
panels(RATIOS, list(STAGES), lambda n: f'N={n}', lambda r: f'T/T_inf = {r:g}',
       plt.get_cmap('plasma'), 'lip-distance-by-ratio.png', lambda r, n: (n, r))
