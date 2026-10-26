#!/usr/bin/env python3
"""Define 1–6 planar compression ramps and a sharp cowl; no CFD dispatch.

Constant-gamma design uses the standard library; NASA7 requires Aerophysics.
--plot requires Matplotlib.
See docs/intake-design.md for definitions, assumptions and examples.
"""
import argparse
import json
import math
from pathlib import Path

MAX_STAGES = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, f'{name} must be finite and positive')


def normal_state(normal_mach_squared, gamma):
    x, g = normal_mach_squared, gamma
    pressure = 1 + 2*g/(g+1)*(x-1)
    density = (g+1)*x/((g-1)*x+2)
    recovery = math.exp((g*math.log(density)-math.log(pressure))/(g-1))
    return pressure, density, recovery


def turning(beta, mach, gamma):
    return math.atan(2/math.tan(beta)*(mach*mach*math.sin(beta)**2-1)
                     / (mach*mach*(gamma+math.cos(2*beta))+2))


def weak_limit(mach, gamma):
    """Beta at maximum turning; theta(beta) is unimodal on the attached branch."""
    lo, hi = math.asin(1/mach), math.pi/2
    q = (math.sqrt(5)-1)/2
    for _ in range(90):
        a, b = hi-q*(hi-lo), lo+q*(hi-lo)
        if turning(a, mach, gamma) < turning(b, mach, gamma):
            lo = a
        else:
            hi = b
    return (lo+hi)/2


def nasa7_gas():
    try:
        from aerophysics import AIR_NASA7
    except ImportError as exc:
        raise ImportError('NASA7 design requires Aerophysics installed in this Python environment') from exc
    return AIR_NASA7


def thermal_stage(mach, temperature, target):
    """Invert static-temperature rise, then independently verify the weak branch."""
    from aerophysics import normal_shock, oblique_shock
    from aerophysics.exceptions import ModelRangeError
    from aerophysics.shocks import theta_from_shock_angle
    gas = nasa7_gas()
    gas.standard_enthalpy(temperature*target)  # No extrapolation.
    lo, hi = 1.0, mach
    for _ in range(60):
        mid = (lo+hi)/2
        try:
            rise = float(normal_shock(mid, gas=gas, upstream_temperature=temperature).static_temperature_ratio)
        except ModelRangeError:
            hi = mid
            continue
        if rise < target:
            lo = mid
        else:
            hi = mid
    beta = math.asin((lo+hi)/2/mach)
    delta = float(theta_from_shock_angle(mach, beta, gas=gas, upstream_temperature=temperature))
    shock = oblique_shock(mach, delta, gas=gas, upstream_temperature=temperature)
    require(abs(float(shock.static_temperature_ratio)/target-1) < 1e-9
            and abs(float(shock.shock_angle)-beta) < 1e-9,
            'Temperature target unavailable on the NASA7 weak-shock branch')
    return beta, delta, shock


def thermal_energy(mach, temperature):
    gas = nasa7_gas()
    return float(gas.standard_enthalpy(temperature)) + .5*(mach*float(gas.speed_of_sound(temperature)))**2


def cowl_state(result):
    """Weak shock turning the post-ramp flow onto the horizontal cowl."""
    p, f = result['parameters'], result['stage_states'][-1]
    m, g, theta = f['mach_out'], p['gamma'], math.radians(f['angle_deg'])
    if p.get('gas_model') == 'nasa7':
        from aerophysics import oblique_shock
        q = oblique_shock(m, theta, gas=nasa7_gas(), upstream_temperature=f['temperature_out_K'])
        beta = float(q.shock_angle)
        require(float(q.downstream_mach) > 1, 'Cowl shock downstream flow is not supersonic')
        require(-math.pi/2 < theta-beta < 0, 'Cowl shock must point toward lower wall')
        return dict(mach_in=m, mach_out=float(q.downstream_mach), beta_deg=math.degrees(beta),
                    delta_deg=math.degrees(theta), ray_angle_deg=math.degrees(theta-beta),
                    pressure_ratio=float(q.static_pressure_ratio), density_ratio=float(q.static_density_ratio),
                    temperature_ratio=float(q.static_temperature_ratio), total_pressure_ratio=float(q.total_pressure_ratio))
    lo, hi = math.asin(1/m), weak_limit(m, g)
    require(theta < turning(hi, m, g), 'Cowl turn has no attached weak shock')
    for _ in range(90):
        mid = (lo+hi)/2
        if turning(mid, m, g) < theta:
            lo = mid
        else:
            hi = mid
    beta = (lo+hi)/2
    normal_squared = (m*math.sin(beta))**2
    pressure, density, recovery = normal_state(normal_squared, g)
    mn_out = math.sqrt((1+(g-1)/2*normal_squared)/(g*normal_squared-(g-1)/2))
    downstream = mn_out/math.sin(beta-theta)
    require(downstream > 1, 'Cowl shock downstream flow is not supersonic')
    require(-math.pi/2 < theta-beta < 0, 'Cowl shock must point toward lower wall')
    return dict(mach_in=m, mach_out=downstream, beta_deg=math.degrees(beta),
                delta_deg=math.degrees(theta), ray_angle_deg=math.degrees(theta-beta),
                pressure_ratio=pressure, density_ratio=density,
                temperature_ratio=pressure/density, total_pressure_ratio=recovery)


def design(mach, temperature_ratio, stages, *, gap_height=None, capture_height=None,
           gamma=None, duct_length=1.0, cowl_thickness=0.02, cowl_nose_length=0.1,
           temperature_location='after_ramps', ramp_end=None, gas_model='constant_gamma',
           upstream_temperature=None, duct_gap_height=None):
    """Design to a static-temperature ratio relative to the freestream.

    The after_cowl option inverts the equal-strength ramp family including the
    subsequent cowl turn; it does not optimize recovery across all shocks.
    """
    require(gas_model in ('constant_gamma', 'nasa7'), 'Invalid gas_model')
    require(gas_model != 'nasa7' or gamma is None, '--gamma is only valid for constant_gamma')
    gamma = 1.4 if gamma is None else gamma
    if gas_model == 'nasa7':
        positive(upstream_temperature, 'upstream_temperature')
        nasa7_gas().standard_enthalpy(upstream_temperature)
    if duct_gap_height is not None:
        positive(duct_gap_height, 'duct_gap_height')
        require(ramp_end is None, 'duct_gap_height cannot be combined with ramp_end')
        require(temperature_location == 'after_ramps', 'Fixed duct height requires after_ramps target')
    require(temperature_location in ('after_ramps', 'after_cowl'), 'Invalid temperature_location')
    if ramp_end is None:
        ramp_end = 'lip_shock' if temperature_location == 'after_cowl' else 'lip_x'
    require(ramp_end in ('lip_x', 'lip_shock'), 'Invalid ramp_end')
    require(temperature_location != 'after_cowl' or ramp_end == 'lip_shock',
            'after_cowl temperature target requires ramp_end=lip_shock')
    options = dict(gap_height=gap_height, capture_height=capture_height, gamma=gamma,
                   duct_length=duct_length, cowl_thickness=cowl_thickness, cowl_nose_length=cowl_nose_length,
                   gas_model=gas_model, upstream_temperature=upstream_temperature)
    # Validate all original inputs even if the requested target needs inversion.
    positive(temperature_ratio, 'temperature_ratio')
    require(temperature_ratio > 1, 'temperature_ratio must exceed 1')
    if temperature_location == 'after_ramps':
        result = _design_ramps(mach, temperature_ratio, stages, **options)
        if duct_gap_height is not None:
            return fixed_duct_height(result, duct_gap_height)
        if ramp_end == 'lip_x':
            return result  # Preserve the legacy lip-plane geometry.
        cowl = cowl_state(result)
    else:
        # Validate the non-target inputs before searching; high candidates may
        # exceed the connected attached/supersonic branch and are bracket bounds.
        low = 1+min(1e-6, (temperature_ratio-1)*.01)
        result = _design_ramps(mach, low, stages, **options)
        cowl_state(result)
        high = temperature_ratio
        solved = False
        for _ in range(85):
            mid = (low+high)/2
            try:
                candidate = _design_ramps(mach, mid, stages, **options)
                shock = cowl_state(candidate)
            except ValueError:
                high = mid
                continue
            cumulative = candidate['performance']['temperature_ratio']*shock['temperature_ratio']
            if abs(cumulative/temperature_ratio-1) < 1e-12:
                result, cowl, solved = candidate, shock, True
                break
            if cumulative < temperature_ratio:
                low = mid
            else:
                high = mid
        require(solved, 'Target unavailable on the attached, supersonic cowl-shock branch')
    g, p = result['geometry'], result['parameters']
    lip, old_end = g['cowl_lip_m'], g['ramp_vertices_m'][-1]
    ramp_slope = math.tan(math.radians(result['stage_states'][-1]['angle_deg']))
    shock_slope = math.tan(math.radians(cowl['ray_angle_deg']))
    dx = (lip[1]-old_end[1])/(ramp_slope-shock_slope)
    end = [lip[0]+dx, old_end[1]+dx*ramp_slope]
    require(lip[0] < end[0] < g['outlet_x_m'], 'Lip shock intersection must lie before outlet')
    require(end[1] < lip[1], 'Ramp closes the inlet')
    # Original ramp lengths still locate the lip for intake_euler input replay.
    g['ramp_end_point_m'] = end
    g['walls_m']['lower'] = g['ramp_vertices_m'][:-1]+[end, [g['outlet_x_m'], end[1]]]
    g['duct_gap_height_m'] = lip[1]-end[1]
    p.update(temperature_location=temperature_location, temperature_ratio=temperature_ratio,
             ramp_temperature_ratio=result['performance']['temperature_ratio'], ramp_end=ramp_end)
    result['cowl_shock'] = cowl | dict(origin_m=lip, wall_intersection_m=end)
    before = result['performance']
    result['performance_after_cowl'] = dict(
        mach=cowl['mach_out'], temperature_ratio=before['temperature_ratio']*cowl['temperature_ratio'],
        pressure_ratio=before['pressure_ratio']*cowl['pressure_ratio'],
        density_ratio=before['density_ratio']*cowl['density_ratio'],
        total_pressure_recovery=before['total_pressure_recovery']*cowl['total_pressure_ratio'])
    after = result['performance_after_cowl']
    energy_error = abs((1+(gamma-1)/2*mach**2)
                       -after['temperature_ratio']*(1+(gamma-1)/2*after['mach']**2))
    mass_ratio = (after['density_ratio']*after['mach']/mach*math.sqrt(after['temperature_ratio'])
                  *g['duct_gap_height_m']/g['capture_height_m'])
    if gas_model == 'nasa7':
        t = upstream_temperature*after['temperature_ratio']
        h0 = thermal_energy(mach, upstream_temperature)
        energy_error = abs(thermal_energy(after['mach'], t)/h0-1)
        gas = nasa7_gas()
        mass_ratio = (after['density_ratio']*after['mach']/mach
                      *float(gas.speed_of_sound(t))/float(gas.speed_of_sound(upstream_temperature))
                      *g['duct_gap_height_m']/g['capture_height_m'])
    energy_tolerance = 1e-9 if gas_model == 'nasa7' else 1e-9*max(1, mach**2)
    require(energy_error < energy_tolerance, 'Post-cowl energy consistency check failed')
    require(abs(mass_ratio-1) < 1e-9, 'Post-cowl mass conservation check failed')
    result['checks'].update(post_cowl_energy_error=energy_error, post_cowl_mass_flux_ratio=mass_ratio)
    result['limitations'].append('Post-cowl target uses isolated shocks; CFD interaction and grid errors may change it.')
    return result


def fixed_duct_height(result, height):
    g, p = result['geometry'], result['parameters']
    require(height <= g['gap_height_m'], 'duct_gap_height exceeds lip opening')
    lip = g['cowl_lip_m']
    end_y = lip[1]-height
    slope = math.tan(math.radians(g['ramps'][-1]['angle_deg']))
    end_x = lip[0]+(end_y-g['ramp_vertices_m'][-1][1])/slope
    require(end_x < g['outlet_x_m'], 'Duct ramp endpoint must lie before outlet')
    end = [end_x, end_y]
    g.update(ramp_end_point_m=end, duct_gap_height_m=height)
    g['walls_m']['lower'] = g['ramp_vertices_m'][:-1]+[end, [g['outlet_x_m'], end_y]]
    p['duct_gap_height'] = height
    return result


def _design_ramps(mach, temperature_ratio, stages, *, gap_height=None, capture_height=None,
           gamma=1.4, duct_length=1.0, cowl_thickness=0.02, cowl_nose_length=0.1,
           gas_model='constant_gamma', upstream_temperature=None):
    """Equal-strength, common-focus design; height is vertical, lengths horizontal.

    Equal normal Mach numbers maximize recovery at fixed cumulative temperature
    ratio in this calorically perfect, isolated-shock model when feasible. Reject
    infeasible weak-branch/forward geometry instead of changing the target.
    """
    for name, value in [('mach', mach), ('temperature_ratio', temperature_ratio),
                        ('gamma', gamma), ('duct_length', duct_length),
                        ('cowl_thickness', cowl_thickness), ('cowl_nose_length', cowl_nose_length)]:
        positive(value, name)
    require(mach > 1 and gamma > 1, 'mach and gamma must exceed 1')
    require(temperature_ratio > 1, 'temperature_ratio must exceed 1')
    require(isinstance(stages, int) and not isinstance(stages, bool)
            and 1 <= stages <= MAX_STAGES, 'stages must be an integer from 1 to 6')
    require((gap_height is None) != (capture_height is None), 'Specify exactly one height')
    height_kind = 'gap_height' if gap_height is not None else 'capture_height'
    height = gap_height if gap_height is not None else capture_height
    positive(height, height_kind)
    require(duct_length > cowl_nose_length, 'duct_length must exceed cowl_nose_length')
    # Positive downstream Mach > 1 is required by this supersonic design model.
    max_ratio = (1+(gamma-1)/2*mach*mach)/(1+(gamma-1)/2)
    if gas_model == 'constant_gamma':
        require(temperature_ratio < max_ratio, 'Target requires sonic or subsonic downstream flow')
    target_stage = math.exp(math.log(temperature_ratio)/stages)
    if gas_model == 'nasa7':
        flow = []
        m, theta, temperature = mach, 0.0, upstream_temperature
        for i in range(stages):
            beta, delta, q = thermal_stage(m, temperature, target_stage)
            ray = theta+beta
            theta += delta
            require(0 < delta and theta < math.pi/2 and ray < math.pi/2,
                    'No forward-facing ramp / shock geometry')
            downstream = float(q.downstream_mach)
            require(downstream > 1, 'Downstream flow is not supersonic')
            next_temperature = temperature*float(q.static_temperature_ratio)
            flow.append(dict(stage=i+1, mach_in=m, mach_out=downstream,
                             delta_deg=math.degrees(delta), angle_deg=math.degrees(theta),
                             beta_deg=math.degrees(beta), ray_angle_deg=math.degrees(ray),
                             temperature_ratio=float(q.static_temperature_ratio),
                             pressure_ratio=float(q.static_pressure_ratio), density_ratio=float(q.static_density_ratio),
                             total_pressure_ratio=float(q.total_pressure_ratio),
                             temperature_in_K=temperature, temperature_out_K=next_temperature))
            m, temperature = downstream, next_temperature
    else:
        lo, hi = 1.0, mach*mach
        p, rho, _ = normal_state(hi, gamma)
        require(target_stage < p/rho, 'Temperature rise exceeds available normal shock strength')
        for _ in range(100):
            mid = (lo+hi)/2
            p, rho, _ = normal_state(mid, gamma)
            if p/rho < target_stage:
                lo = mid
            else:
                hi = mid
        normal_squared = (lo+hi)/2
        pressure, density, recovery = normal_state(normal_squared, gamma)
        flow = []
        m, theta = mach, 0.0
        for i in range(stages):
            require(normal_squared < m*m, f'Stage {i+1}: insufficient upstream Mach')
            beta = math.asin(math.sqrt(normal_squared)/m)
            require(beta < weak_limit(m, gamma), f'Stage {i+1}: target is outside attached weak branch')
            delta = turning(beta, m, gamma)
            ray = theta+beta
            theta += delta
            require(0 < delta and theta < math.pi/2 and ray < math.pi/2,
                    f'Stage {i+1}: no forward-facing ramp / shock geometry')
            mn_out = math.sqrt((1+(gamma-1)/2*normal_squared)
                               / (gamma*normal_squared-(gamma-1)/2))
            downstream = mn_out/math.sin(beta-delta)
            require(downstream > 1, f'Stage {i+1}: downstream flow is not supersonic')
            flow.append(dict(stage=i+1, mach_in=m, mach_out=downstream,
                             delta_deg=math.degrees(delta), angle_deg=math.degrees(theta),
                             beta_deg=math.degrees(beta), ray_angle_deg=math.degrees(ray),
                             temperature_ratio=pressure/density, pressure_ratio=pressure,
                             density_ratio=density, total_pressure_ratio=recovery))
            m = downstream
    # Set lip x = 1, then intersect each ramp with the NEXT shock through the lip.
    lip = [1.0, math.tan(math.radians(flow[0]['ray_angle_deg']))]
    vertices = [[0.0, 0.0]]
    for prev, following in zip(flow, flow[1:]):
        x, y = vertices[-1]
        slope = math.tan(math.radians(prev['angle_deg']))
        shock_slope = math.tan(math.radians(following['ray_angle_deg']))
        dx = (lip[1]-y-(1-x)*shock_slope)/(slope-shock_slope)
        require(0 < dx < 1-x, 'Common-focus geometry has a nonpositive ramp length')
        vertices.append([x+dx, y+dx*slope])
    x, y = vertices[-1]
    vertices.append([1.0, y+(1-x)*math.tan(math.radians(flow[-1]['angle_deg']))])
    gap = lip[1]-vertices[-1][1]
    require(gap > 0, 'Ramp closes the inlet')
    scale = height/(gap if gap_height is not None else lip[1])
    vertices = [[x*scale, y*scale] for x, y in vertices]
    lip = [v*scale for v in lip]
    gap *= scale
    end = vertices[-1]
    outlet_x = lip[0]+duct_length
    ramps = [dict(angle_deg=f['angle_deg'], length=vertices[i+1][0]-vertices[i][0])
             for i, f in enumerate(flow)]
    for f, origin in zip(flow, vertices):
        f['origin_m'] = origin
        f['focus_m'] = lip
    walls = dict(lower=vertices+[[outlet_x, end[1]]],
                 cowl_inner=[lip, [outlet_x, lip[1]]],
                 cowl_outer=[lip, [lip[0]+cowl_nose_length, lip[1]+cowl_thickness],
                             [outlet_x, lip[1]+cowl_thickness]])
    total_pressure = math.prod(f['total_pressure_ratio'] for f in flow)
    temperature = math.prod(f['temperature_ratio'] for f in flow)
    pressure_total = math.prod(f['pressure_ratio'] for f in flow)
    density_total = math.prod(f['density_ratio'] for f in flow)
    if gas_model == 'nasa7':
        gas = nasa7_gas()
        t = upstream_temperature*temperature
        h0 = thermal_energy(mach, upstream_temperature)
        speed = math.sqrt(2*(h0-float(gas.standard_enthalpy(t))))
        expected_m = speed/float(gas.speed_of_sound(t))
        speed_ratio = m*float(gas.speed_of_sound(t))/(mach*float(gas.speed_of_sound(upstream_temperature)))
    else:
        expected_m = math.sqrt(((1+(gamma-1)/2*mach*mach)/temperature_ratio-1)*2/(gamma-1))
        speed_ratio = m/mach*math.sqrt(temperature)
    mass_ratio = density_total*speed_ratio*math.cos(theta)*gap/lip[1]
    focus_error = max(abs(lip[1]-f['origin_m'][1]-(lip[0]-f['origin_m'][0])
                         *math.tan(math.radians(f['ray_angle_deg']))) for f in flow)
    require(abs(m-expected_m) < 1e-9*max(1, expected_m), 'Energy consistency check failed')
    require(abs(mass_ratio-1) < 1e-9, 'Capture mass conservation check failed')
    require(focus_error < 1e-9*max(1, scale), 'Shock focus consistency check failed')
    require(all(math.isfinite(v) for point in sum(walls.values(), []) for v in point),
            'Nonfinite geometry')
    result = dict(schema_version=1, model='calorically_perfect_sequential_weak_shocks',
                design_rule='equal_shock_strength_and_common_cowl_focus',
                parameters=dict(mach=mach, temperature_ratio=temperature_ratio, stages=stages,
                                gamma=gamma, height_definition=height_kind, height_m=height,
                                duct_length=duct_length, cowl_thickness=cowl_thickness,
                                cowl_nose_length=cowl_nose_length),
                geometry=dict(ramps=ramps, ramp_vertices_m=vertices, cowl_lip_m=lip,
                              gap_height_m=gap, capture_height_m=lip[1], ramp_length_m=lip[0],
                              outlet_x_m=outlet_x, cowl_radius_m=0.0, walls_m=walls),
                stage_states=flow,
                performance=dict(mach_after_ramps=m, temperature_ratio=temperature,
                                 pressure_ratio=pressure_total, density_ratio=density_total,
                                 total_pressure_recovery=total_pressure),
                checks=dict(energy_mach_error=abs(m-expected_m),
                            shock_focus_error_m=focus_error, mass_flux_ratio=mass_ratio),
                limitations=['Performance is before internal reflections/expansions, not at the duct exit.',
                             'Horizontal internal walls and sharp cowl dimensions are prescribed, not optimized.',
                             'No viscosity, real-gas effects, combustion, or CFD validation.'])
    result['parameters']['gas_model'] = gas_model
    if gas_model == 'nasa7':
        result['model'] = 'frozen_nasa7_sequential_weak_shocks'
        result['design_rule'] = 'equal_temperature_ratios_and_common_cowl_focus'
        result['parameters'].update(upstream_temperature=upstream_temperature, gamma=None)
        result['limitations'] = [
            'Frozen NASA7 composition; no dissociation, chemistry, viscosity or CFD validation.',
            'Equal temperature ratios are prescribed, not a total-pressure optimum.',
            'Performance is before internal reflections/expansions, not at the duct exit.',
            'Mesh export retains the constant-gamma SU2 solver settings; only geometry uses NASA7.']
    return result



def mesh_input(result, temperature, pressure):
    """Explicit dimensional freestream inputs; no implied atmosphere."""
    positive(temperature, 'temperature')
    positive(pressure, 'pressure')
    g, p = result['geometry'], result['parameters']
    raw = dict(kind='intake', mach=p['mach'], gamma=p['gamma'] or 1.4, temperature=temperature,
                pressure=pressure, ramps=g['ramps'], cowl_lip_height=g['cowl_lip_m'][1],
                cowl_thickness=p['cowl_thickness'], cowl_nose_length=p['cowl_nose_length'],
                outlet_x=g['outlet_x_m'],
                top_height_factor=max(3.0, 2+p['cowl_thickness']/g['capture_height_m']),
                output='mesh')
    if p.get('gas_model') == 'nasa7':
        require(temperature == p['upstream_temperature'], 'Mesh temperature must match NASA7 design temperature')
        raw['geometry_design'] = dict(gas_model='nasa7', upstream_temperature=temperature,
                                      solver_gas_model='constant_gamma')
        # Preserve the thermal endpoint; never re-solve its cowl shock with constant gamma.
        if 'ramp_end_point_m' in g:
            raw['duct_gap_height'] = g['duct_gap_height_m']
    elif 'ramp_end' in p:
        raw['ramp_end'] = p['ramp_end']
    if 'duct_gap_height' in p:
        raw['duct_gap_height'] = p['duct_gap_height']
    return raw


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    g, p = result['geometry'], result['parameters']
    fig, ax = plt.subplots(figsize=(13, 6), layout='constrained')
    lower = g['walls_m']['lower']
    x, y = zip(*lower)
    ax.fill_between(x, y, -0.12*g['capture_height_m'], color='#E6E9EB')
    ax.plot(x, y, color='#29333B', lw=2, label='Ramp / lower wall')
    outer = g['walls_m']['cowl_outer']
    ax.add_patch(Polygon(outer+[g['walls_m']['cowl_inner'][-1]], color='#29333B'))
    for i, f in enumerate(result['stage_states']):
        o, c = f['origin_m'], f['focus_m']
        ax.plot([o[0],c[0]], [o[1],c[1]], color='#246A9B', ls='--', lw=1.1,
                label='Theoretical shocks' if i == 0 else None)
    if 'cowl_shock' in result:
        shock = result['cowl_shock']
        a, b = shock['origin_m'], shock['wall_intersection_m']
        ax.plot([a[0], b[0]], [a[1], b[1]], '--', color='#C66B20', label='Cowl shock theory')
    for i, (a,b) in enumerate(zip(g['ramp_vertices_m'],g['ramp_vertices_m'][1:])):
        ax.annotate(str(i+1), ((a[0]+b[0])/2,(a[1]+b[1])/2), xytext=(0,-16),
                    textcoords='offset points',ha='center',fontsize=9)
    c = g['cowl_lip_m']
    ax.scatter(*c, marker='D', color='#C66B20', zorder=5)
    ax.annotate('Cowl lip', c, xytext=(-65,30), textcoords='offset points',
                arrowprops=dict(arrowstyle='-',color='#29333B'))
    dim_x = (g.get('ramp_end_point_m', c)[0]+g['outlet_x_m'])/2
    ax.plot([dim_x,dim_x],[lower[-1][1],c[1]],color='#29333B',lw=1)
    tick = .006*g['ramp_length_m']
    for yy in (lower[-1][1],c[1]):
        ax.plot([dim_x-tick,dim_x+tick],[yy,yy],color='#29333B',lw=1)
    gap_label = f"Duct gap {g.get('duct_gap_height_m', g['gap_height_m']):.4g} m"
    if 'ramp_end_point_m' in g:
        ax.annotate(gap_label, (dim_x, (c[1]+lower[-1][1])/2),
                    xytext=(dim_x, c[1]+p['cowl_thickness']+.06*g['capture_height_m']),
                    ha='center', va='bottom', fontsize=10,
                    arrowprops=dict(arrowstyle='-', color='#29333B'))
    else:
        ax.annotate(gap_label, (dim_x, (c[1]+lower[-1][1])/2),
                    xytext=(8,0), textcoords='offset points', va='center', fontsize=10)
    ax.set_aspect('equal',adjustable='box')
    ax.set(xlabel='x [m]',ylabel='y [m]')
    ax.margins(x=.06,y=.30)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(loc='upper left',frameon=False)
    location = 'cowl' if p.get('temperature_location') == 'after_cowl' else 'ramps'
    fig.suptitle(f"{p['stages']}-ramp intake | M = {p['mach']:g} | T after {location} / T in = {p['temperature_ratio']:g}",fontsize=15)
    ax.set_title(f"{p.get('gas_model', 'constant_gamma')} | Ramp length {g['ramp_length_m']:.4f} m | Capture height {g['capture_height_m']:.4f} m | Equal x/y scale",fontsize=11)
    fig.supxlabel('Sharp cowl; prescribed horizontal duct. Dashed lines are shock theory, not CFD.',fontsize=10)
    for extension in ('png','svg'):
        fig.savefig(output/f'geometry.{extension}',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mach', type=float, required=True)
    parser.add_argument('--temperature-ratio', type=float, required=True)
    parser.add_argument('--temperature-location', choices=('after_ramps', 'after_cowl'), default='after_ramps',
                        help='Static-temperature target location; ratio is relative to freestream')
    parser.add_argument('--ramp-end', choices=('lip_x', 'lip_shock'),
                        help='Default: lip_shock for after_cowl, otherwise lip_x')
    parser.add_argument('--stages', type=int, required=True, choices=range(1, MAX_STAGES+1))
    heights = parser.add_mutually_exclusive_group(required=True)
    heights.add_argument('--gap-height',type=float, help='Vertical lip-to-ramp-end gap [m]')
    heights.add_argument('--capture-height',type=float, help='Lip height above ramp origin [m]')
    parser.add_argument('--gas-model', choices=('constant_gamma', 'nasa7'), default='constant_gamma')
    parser.add_argument('--gamma',type=float, help='Constant-gamma model only; default 1.4')
    parser.add_argument('--duct-gap-height',type=float, help='Fixed parallel duct height [m]; excludes --ramp-end')
    parser.add_argument('--duct-length',type=float,default=1.0)
    parser.add_argument('--cowl-thickness',type=float,default=0.02)
    parser.add_argument('--cowl-nose-length',type=float,default=0.1)
    parser.add_argument('--temperature',type=float,help='Freestream static K; with --pressure exports mesh input')
    parser.add_argument('--pressure',type=float,help='Freestream static Pa; requires --temperature')
    parser.add_argument('--output',type=Path,required=True,help='New output directory (never overwritten)')
    parser.add_argument('--plot',action='store_true',help='Also render PNG/SVG with Matplotlib')
    args = parser.parse_args()
    try:
        require(not args.output.exists(), 'Output already exists; choose a new directory')
        require((args.pressure is None and (args.temperature is None or args.gas_model == 'nasa7'))
                or (args.temperature is not None and args.pressure is not None),
                'Supply both --temperature and --pressure to export mesh input')
        result = design(args.mach, args.temperature_ratio, args.stages,
                        gap_height=args.gap_height, capture_height=args.capture_height,
                        gamma=args.gamma, duct_length=args.duct_length,
                        cowl_thickness=args.cowl_thickness, cowl_nose_length=args.cowl_nose_length,
                        temperature_location=args.temperature_location, ramp_end=args.ramp_end,
                        gas_model=args.gas_model, upstream_temperature=args.temperature,
                        duct_gap_height=args.duct_gap_height)
        mesh = mesh_input(result,args.temperature,args.pressure) if args.pressure is not None else None
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'design.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if mesh is not None:
            (args.output/'intake-input.json').write_text(json.dumps(mesh,indent=2,allow_nan=False)+'\n')
        if args.plot:
            plot(result,args.output)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2,f'Error: {exc}\n')
    g = result['geometry']
    if args.gas_model == 'nasa7' and mesh is not None:
        print('NASA7 geometry exported; intake-input.json still uses the constant-gamma SU2 solver model.')
    print(f"Saved {args.output}/design.json; ramp length={g['ramp_length_m']:.6f} m, gap={g['gap_height_m']:.6f} m")
    for f, ramp in zip(result['stage_states'], g['ramps']):
        print(f"Stage {f['stage']}: angle={ramp['angle_deg']:.6f} deg, length={ramp['length']:.6f} m, M={f['mach_out']:.6f}")


if __name__ == '__main__':
    main()
