#!/usr/bin/env python3
"""Perfect-gas reference values for the intake off-design sweep (no CFD).

Sequential uniform waves along the ramp surface: each ramp corner turns the flow
by (ramp angle - previous flow angle) with a weak oblique shock, or with a
Prandtl-Meyer expansion when that turn is negative. The freestream direction is
the angle of attack in the SU2 sign convention (+AoA tilts the flow toward the
ramp angle, so the first turn is ramp1 - AoA and compression weakens).

The theory state is the one after all ramp waves (the state on the last ramp,
ahead of the cowl shock). It is the reference for the CFD lip-section averages.
Straight waves are extended to the lip plane; when two of them cross below the
lip before it, the flags mark the condition where the sequential model does not
describe the captured flow (wave coalescence, CFD differs). Kantrowitz uses the
theory Mach at the lip: the largest lip-opening / duct-height ratio that
self-starts with a normal shock at the lip and an isentropic contraction to M=1.
"""
import math


def beta_weak(mach, theta_deg, gamma):
    """Weak-branch oblique shock angle [rad]; ValueError when detached."""
    theta = math.radians(theta_deg)
    mu = math.asin(1/mach)

    def turn(b):
        return math.atan(2/math.tan(b)*(mach*mach*math.sin(b)**2-1)/(mach*mach*(gamma+math.cos(2*b))+2))
    grid = [mu+(math.pi/2-mu)*i/4000 for i in range(1, 4000)]
    peak = max(grid, key=turn)
    if turn(peak) < theta:
        raise ValueError(f'Detached shock: Mach {mach:.3f}, turn {theta_deg:.3f} deg')
    lo, hi = mu, peak
    for _ in range(200):
        mid = (lo+hi)/2
        lo, hi = (lo, mid) if turn(mid) > theta else (mid, hi)
    return (lo+hi)/2


def oblique(state, theta_deg, gamma):
    """Return the state after a compressive turn and the absolute wave angle [deg]."""
    b = beta_weak(state['mach'], theta_deg, gamma)
    mn = state['mach']*math.sin(b)
    pr = 1+2*gamma/(gamma+1)*(mn*mn-1)
    rr = (gamma+1)*mn*mn/((gamma-1)*mn*mn+2)
    mn2 = math.sqrt((1+(gamma-1)/2*mn*mn)/(gamma*mn*mn-(gamma-1)/2))
    p0r = rr**(gamma/(gamma-1))*(1/pr)**(1/(gamma-1))
    new = dict(mach=mn2/math.sin(b-math.radians(theta_deg)), p_ratio=state['p_ratio']*pr,
               T_ratio=state['T_ratio']*pr/rr, p0_recovery=state['p0_recovery']*p0r,
               angle_deg=state['angle_deg']+theta_deg)
    return new, state['angle_deg']+math.degrees(b)


def prandtl_meyer(mach, gamma):
    a = math.sqrt((gamma+1)/(gamma-1))
    return a*math.atan(math.sqrt((mach*mach-1)/a/a))-math.atan(math.sqrt(mach*mach-1))


def expansion(state, theta_deg, gamma):
    """Isentropic expansion by -theta_deg (theta_deg < 0); the wave angle is the fan centre."""
    target = prandtl_meyer(state['mach'], gamma)-math.radians(theta_deg)
    lo, hi = state['mach'], 100.
    for _ in range(200):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if prandtl_meyer(mid, gamma) < target else (lo, mid)
    m2 = (lo+hi)/2
    tr = (1+(gamma-1)/2*state['mach']**2)/(1+(gamma-1)/2*m2*m2)
    new = dict(mach=m2, p_ratio=state['p_ratio']*tr**(gamma/(gamma-1)), T_ratio=state['T_ratio']*tr,
               p0_recovery=state['p0_recovery'], angle_deg=state['angle_deg']+theta_deg)
    mu1, mu2 = math.degrees(math.asin(1/state['mach'])), math.degrees(math.asin(1/m2))
    return new, (state['angle_deg']+mu1+new['angle_deg']+mu2)/2


def kantrowitz_contraction(mach, gamma=1.4):
    """Largest self-starting area ratio for a normal shock at Mach `mach`."""
    m2 = math.sqrt((1+(gamma-1)/2*mach*mach)/(gamma*mach*mach-(gamma-1)/2))
    return (1/m2)*((2/(gamma+1))*(1+(gamma-1)/2*m2*m2))**((gamma+1)/(2*(gamma-1)))


def lip_theory(mach, aoa_deg, corners, angles, lip, gamma=1.4):
    """Sequential-wave state on the last ramp and the wave-crossing flags.

    corners: (x, y) of each ramp start (first is (0, 0)); angles: absolute ramp
    angles [deg]; lip: (x, y) of the cowl tip.
    """
    state = dict(mach=mach, p_ratio=1., T_ratio=1., p0_recovery=1., angle_deg=aoa_deg)
    waves = []
    for (cx, cy), angle in zip(corners, angles):
        turn = angle-state['angle_deg']
        if abs(turn) < 1e-12:
            continue
        state, wave = (oblique if turn > 0 else expansion)(state, turn, gamma)
        waves.append(dict(x=cx, y=cy, angle_deg=wave, kind='shock' if turn > 0 else 'expansion'))
    xl, yl = lip
    # Height of each straight wave at the lip plane (None when it starts downstream of it).
    heights = [w['y']+(xl-w['x'])*math.tan(math.radians(w['angle_deg'])) for w in waves]
    crossed = any(heights[j+1] > heights[j]+1e-12 and min(heights[j], heights[j+1]) < yl
                  for j in range(len(heights)-1))
    above = [h for h in heights if h > yl+1e-9]
    return dict(theory_lip_mach=state['mach'], theory_p0_recovery=state['p0_recovery'],
                theory_T_ratio=state['T_ratio'], theory_p_ratio=state['p_ratio'],
                theory_flow_angle_deg=state['angle_deg'],
                theory_waves_cross_ahead_of_lip=bool(crossed),
                theory_waves_above_lip=len(above),
                theory_first_wave_height_at_lip_m=heights[0] if heights else None,
                kantrowitz_max_contraction=kantrowitz_contraction(state['mach'], gamma))
