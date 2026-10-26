"""Experimental, geometry-derived meshing policy for straight-sided 2-D intakes.

No flow-case coordinates or Gmsh entity tags are prescribed here. Dimensions are
metres. This policy controls geometric quality; it does not certify a CFD result.
"""
import math

import numpy as np

from generate_airfoil_mesh import cell_blocks, distribution

WALLS = {"lower", "cowl"}
POLICY_DEFAULTS = dict(name="geometry_v1", leading_size_ratio=0.005,
                       outer_size_ratio=2.0, size_growth=0.2)


def policy_settings(raw):
    if not isinstance(raw, dict) or set(raw) - set(POLICY_DEFAULTS):
        raise ValueError("Invalid mesh_policy fields")
    policy = POLICY_DEFAULTS | raw
    if policy["name"] not in ("geometry_v1", "geometry_v2"):
        raise ValueError("Unknown mesh_policy.name")
    for key in ("leading_size_ratio", "outer_size_ratio", "size_growth"):
        v = policy[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid mesh_policy.{key}")
    if not (0.001 <= policy["leading_size_ratio"] <= 0.1
            and 1 <= policy["outer_size_ratio"] <= 8 and 0.05 <= policy["size_growth"] <= 0.5):
        raise ValueError("mesh_policy controls outside experimental supported range")
    return policy


def layer_rows(layer):
    """Complete geometric layers within the requested maximum thickness."""
    h, r, thickness = (layer[k] for k in ("first_height", "growth_ratio", "thickness"))
    heights, total = [], 0.0
    while total + h <= thickness * (1 + 1e-12):
        heights.append(h)
        total += h
        h *= r
        if len(heights) > 2000:
            raise ValueError("Too many wall layers")
    if len(heights) < 2:
        raise ValueError("At least two complete wall layers required")
    return np.r_[0., np.cumsum(heights)]


def boundary_plan(poly, names, layer, skip_markers=()):
    """Split open edges at layer ends; return directed progression constraints.

    Oblique junctions use the wall-normal projection. Both wall ends of an
    outlet are handled independently. Collinear splits leave the domain intact.
    """
    rows = layer_rows(layer)
    height = float(rows[-1])
    result, markers, constraints = [], [], []
    for i, name in enumerate(names):
        a, b = poly[i], poly[(i+1) % len(poly)]
        delta = b-a
        length = np.linalg.norm(delta)
        tangent = delta/length
        ends = []
        if name not in WALLS and name not in skip_markers:
            if names[i-1] in WALLS:
                ends.append((0, a-poly[i-1]))
            if names[(i+1) % len(names)] in WALLS:
                ends.append((1, poly[(i+2) % len(poly)]-b))
        distances = {}
        for end, direction in ends:
            direction = direction/np.linalg.norm(direction)
            projection = abs(tangent[0]*direction[1]-tangent[1]*direction[0])
            if projection <= 1e-8:
                raise ValueError("Wall-layer termination is tangential to the adjacent boundary")
            distances[end] = (height/projection, projection)
        if sum(v[0] for v in distances.values()) >= length*(1-1e-8):
            raise ValueError(f"Wall-layer termination segments overlap on {name}")
        cuts = [0.] + ([distances[0][0]] if 0 in distances else [])
        cuts += ([length-distances[1][0]] if 1 in distances else []) + [length]
        for j, start in enumerate(cuts[:-1]):
            index = len(result)
            result.append(a+start*tangent)
            markers.append(name)
            end = 0 if j == 0 and 0 in distances else 1 if j == len(cuts)-2 and 1 in distances else None
            if end is not None:
                constraints.append(dict(edge=index, marker=name, wall_point=(a if end == 0 else b).tolist(),
                                        direction=(tangent if end == 0 else -tangent).tolist(),
                                        projection=float(distances[end][1]),
                                        count=len(rows), progression=layer["growth_ratio"]**(1 if end == 0 else -1)))
    return np.array(result), markers, constraints


def slip_partition(poly, names, layer, policy):
    """Split the fluid at a slip-to-viscous wall junction along its inward normal.

    The partition is internal, not a new physical boundary. This avoids forcing
    wall-normal rows onto a nearly tangential upstream slip edge. At present the
    intake geometry has at most one such junction.
    """
    tips = [i for i, name in enumerate(names) if name in WALLS and names[i-1] == "upstream"]
    if not tips:
        return poly, names, None
    if len(tips) != 1:
        raise ValueError("Only one slip-to-wall junction is supported")
    i = tips[0]
    origin = poly[i].copy()
    upstream = poly[i-1]-origin
    upstream_length = np.linalg.norm(upstream)
    upstream /= upstream_length
    tangent = poly[(i+1) % len(poly)]-origin
    normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    scale = np.linalg.norm(np.ptp(poly, axis=0))
    hits = []
    for j, a in enumerate(poly):
        edge = poly[(j+1) % len(poly)]-a
        determinant = cross(normal, edge)
        if abs(determinant) <= 1e-12*np.linalg.norm(edge):
            continue
        distance = cross(a-origin, edge)/determinant
        fraction = cross(a-origin, normal)/determinant
        if distance > 1e-10*scale and -1e-10 <= fraction <= 1+1e-10:
            hits.append((distance, j, float(np.clip(fraction, 0, 1))))
    if not hits:
        raise ValueError("Slip partition does not reach the domain boundary")
    distance, j, fraction = min(hits)
    if names[j] in WALLS or names[j] == "upstream":
        raise ValueError("Slip partition intersects another wall")
    rows = layer_rows(layer)
    if distance <= 2*rows[-1]:
        raise ValueError("Insufficient space for the slip-junction layer partition")
    widths = layer_rows(dict(first_height=policy["leading_size_ratio"]*layer["thickness"],
                             growth_ratio=layer["growth_ratio"], thickness=rows[-1]))
    if widths[-1] >= upstream_length/2:
        raise ValueError("Insufficient upstream length for the slip-junction block")
    hit = origin+distance*normal
    if fraction <= 1e-10:
        hit = poly[j].copy()
    elif fraction >= 1-1e-10:
        hit = poly[(j+1) % len(poly)].copy()
    else:
        poly = np.insert(poly, j+1, hit, axis=0)
        names = names[:j+1]+[names[j]]+names[j+1:]
    tip = int(np.argmin(np.linalg.norm(poly-origin, axis=1)))
    base = origin+widths[-1]*upstream
    poly = np.insert(poly, tip, base, axis=0)
    names = names[:tip]+["upstream"]+names[tip:]
    return poly, names, dict(origin=origin.tolist(), end=hit.tolist(), normal=normal.tolist(),
                             layer_end=(origin+rows[-1]*normal).tolist(),
                             block_base=base.tolist(), block_top=(base+rows[-1]*normal).tolist(),
                             block_width_nodes=len(widths), block_width_m=float(widths[-1]))


def partition_size_function(size, partition, layer, policy):
    """Limit triangle heights beside the thin first wall column at the partition."""
    a, b = np.array(partition["origin"]), np.array(partition["layer_end"])
    delta = b-a
    tip_size = layer["thickness"]*policy["leading_size_ratio"]
    def capped(point):
        point = np.asarray(point)
        fraction = np.clip((point-a)@delta/(delta@delta), 0, 1)
        distance = np.linalg.norm(point-a-fraction*delta)
        return min(size(point), tip_size+policy["size_growth"]*distance)
    return capped


def partition_constraints(partition, layer, policy):
    """Node checks for all four sides of the structured connector block."""
    base, origin = np.array(partition["block_base"]), np.array(partition["origin"])
    upstream = (base-origin)/np.linalg.norm(base-origin)
    width_layer = dict(first_height=policy["leading_size_ratio"]*layer["thickness"],
                       growth_ratio=layer["growth_ratio"], thickness=float(layer_rows(layer)[-1]))
    return [dict(marker="internal_layer_partition", scope="internal", wall_point=partition["origin"],
                 direction=partition["normal"], projection=1.),
            dict(marker="internal_block_outer", scope="internal", wall_point=partition["block_base"],
                 direction=partition["normal"], projection=1.),
            dict(marker="upstream", wall_point=partition["origin"], direction=upstream.tolist(),
                 projection=1., layer=width_layer, axis="tangent"),
            dict(marker="internal_block_top", scope="internal", wall_point=partition["layer_end"],
                 direction=upstream.tolist(), projection=1., layer=width_layer, axis="tangent")]


def size_function(poly, names, layer, policy):
    """Smooth size caps tied to wall distance and topology-derived leading tips."""
    walls = np.array([[poly[i], poly[(i+1) % len(poly)]] for i, n in enumerate(names) if n in WALLS])
    starts, deltas = walls[:, 0], walls[:, 1]-walls[:, 0]
    lengths2 = np.sum(deltas*deltas, axis=1)
    tips = []
    for i, point in enumerate(poly):
        before, after = names[i-1] in WALLS, names[i] in WALLS
        incoming, outgoing = point-poly[i-1], poly[(i+1) % len(poly)]-point
        turn = math.atan2(incoming[0]*outgoing[1]-incoming[1]*outgoing[0], incoming@outgoing)
        # Positive-x flow: wall starts, plus convex solid corners needing a fan.
        if (after and not before and outgoing[0] > 0
                or before and not after and incoming[0] < 0
                or before and after and turn < -math.pi/6):
            tips.append(point)
    tips = np.array(tips).reshape(-1, 2)
    rows = layer_rows(layer)
    outer_size = policy["outer_size_ratio"]*(rows[-1]-rows[-2])
    leading_size = policy["leading_size_ratio"]*layer["thickness"]
    growth = policy["size_growth"]

    def size(point):
        point = np.asarray(point)
        t = np.clip(np.sum((point-starts)*deltas, axis=1)/lengths2, 0, 1)
        distance = float(np.linalg.norm(point-starts-t[:, None]*deltas, axis=1).min())
        cap = outer_size+growth*max(0., distance-rows[-1])
        if len(tips):
            cap = min(cap, leading_size+growth*float(np.linalg.norm(tips-point, axis=1).min()))
        return cap
    return size


def termination_quality(points, markers, constraints, layer):
    """Verify exported boundary nodes, not just the requested transfinite counts."""
    records = []
    for c in constraints:
        target_layer = c.get("layer", layer)
        rows = layer_rows(target_layer)
        candidates = points if c.get("scope") == "internal" else points[np.unique(markers[c["marker"]])]
        origin, direction = np.array(c["wall_point"]), np.array(c["direction"])
        along = (candidates-origin)@direction
        perpendicular = np.abs((candidates-origin)[:, 0]*direction[1]-(candidates-origin)[:, 1]*direction[0])
        scale = max(rows[-1]/c["projection"], np.linalg.norm(origin), 1.)
        tol = 1e-9*scale
        observed = np.sort(along[(perpendicular < tol) & (along >= -tol)
                                 & (along <= rows[-1]/c["projection"]+tol)])
        expected = rows/c["projection"]
        if len(observed) != len(expected):
            raise ValueError(f"Wall-layer termination node count mismatch: {c['marker']}")
        error = float(np.max(np.abs(observed-expected)))
        # Compare in the wall-normal direction: oblique intersections amplify
        # tiny along-boundary errors. Allow 0.2% of the first normal layer, far
        # below the separate factor-of-two wall-height rejection criterion.
        normal_error = error*c["projection"]
        if normal_error > 0.002*target_layer["first_height"]:
            raise ValueError(f"Wall-layer termination spacing mismatch: {c['marker']}")
        record = dict(marker=c["marker"], wall_point=c["wall_point"], scope=c.get("scope", "boundary"),
                      nodes=len(observed), max_position_error_m=error)
        key = "max_tangential_error_over_first_spacing" if c.get("axis") == "tangent" else "max_normal_error_over_first_height"
        record[key] = normal_error/target_layer["first_height"]
        records.append(record)
    rows = layer_rows(layer)
    return dict(complete_layers=len(rows)-1, normal_thickness_m=float(rows[-1]), boundaries=records)


def transition_quality(points, cells):
    """Triangle altitude / adjoining quad normal thickness at every mixed edge.

    These are geometric diagnostics, not a residual-error estimator. Interfaces
    at fan sides are included and can be located from the reported coordinates.
    """
    blocks = cell_blocks(cells)
    owners = {}
    for n, block in blocks.items():
        for cell in block:
            for i in range(n):
                edge = tuple(sorted((int(cell[i]), int(cell[(i+1) % n]))))
                owners.setdefault(edge, []).append((n, cell, i))
    records = []
    for edge, pair in owners.items():
        if len(pair) != 2 or {v[0] for v in pair} != {3, 4}:
            continue
        (_, tri, ti), (_, quad, qi) = sorted(pair, key=lambda v: v[0])
        a, b = points[list(edge)]
        tangent = (b-a)/np.linalg.norm(b-a)
        normal = np.array([-tangent[1], tangent[0]])
        altitude = abs(float((points[tri[(ti+2) % 3]]-a)@normal))
        opposite = points[quad[[(qi+2) % 4, (qi+3) % 4]]].mean(axis=0)
        height = abs(float((opposite-(a+b)/2)@normal))
        if height <= 0 or not math.isfinite(height):
            raise ValueError("Invalid quad thickness at triangle interface")
        records.append(dict(midpoint_m=((a+b)/2).tolist(), quad_height_m=height,
                            triangle_altitude_m=altitude, ratio=altitude/height))
    return dict(interfaces=len(records), altitude_ratio=distribution([r["ratio"] for r in records]) if records else None,
                largest=sorted(records, key=lambda r: r["ratio"], reverse=True)[:20],
                note="All quad/triangle interfaces, including fan sides; diagnostic only, no universal pass threshold.")


def policy_assessment(transition, policy):
    """Experimental policy postcondition, separate from CFD acceptance criteria.

    A factor of two allows unstructured-cell geometry to differ from the target
    size. This is a policy guardrail, not a universal mesh-quality threshold.
    """
    limit = 2*policy["outer_size_ratio"]
    measured = transition["altitude_ratio"]["maximum"] if transition["interfaces"] else None
    passed = measured is not None and math.isfinite(measured) and measured <= limit
    return dict(status="mesh_geometry_passed" if passed else "review_required",
                maximum_altitude_ratio=measured, policy_ratio_limit=limit,
                reasons=[] if passed else ["Quad/triangle transition exceeds this experimental policy's size allowance; inspect layer_transition.largest and revise the junction topology or sizing before running RANS."],
                validated_combustor_inlet=False)


def initial_wall_spacing(mach, pressure, temperature, reference_length,
                         gamma=1.4, gas_constant=287.05, target_y_plus=.5):
    """Cold-freestream flat-plate seed, not a compressible wall-friction model.

    Cf=0.0577/Re_x**(1/5): NASA Turbulence Guide, Eq. 3.7 (incompressible
    turbulent reference). Wall properties are provisionally set to freestream
    values. The reference x is not the distance to each actual leading edge.
    Recompute maximum y+ from a valid RANS pilot before accepting the spacing.
    """
    values=(mach,pressure,temperature,reference_length,gamma,gas_constant,target_y_plus)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
           or v<=0 for v in values) or mach<=1 or gamma<=1 or target_y_plus>1:
        raise ValueError('Invalid initial wall-spacing inputs')
    viscosity=1.716e-5*(temperature/273.15)**1.5*(273.15+110.4)/(temperature+110.4)
    density=pressure/(gas_constant*temperature)
    velocity=mach*math.sqrt(gamma*gas_constant*temperature)
    reynolds=density*velocity*reference_length/viscosity
    cf=.0577/reynolds**.2
    shear=.5*cf*density*velocity**2
    friction_velocity=math.sqrt(shear/density)
    height=target_y_plus*viscosity/(density*friction_velocity)
    if not all(math.isfinite(v) and v>0 for v in (viscosity,density,velocity,reynolds,cf,shear,height)):
        raise ValueError('Nonfinite initial wall-spacing estimate')
    return dict(method='freestream_flat_plate_v1',provisional=True,
                reference_length_m=reference_length,reference_reynolds=reynolds,
                reference_density_kg_m3=density,reference_viscosity_pa_s=viscosity,
                reference_velocity_m_s=velocity,reference_skin_friction=cf,
                reference_wall_shear_pa=shear,reference_friction_velocity_m_s=friction_velocity,
                target_y_plus=target_y_plus,first_height_m=height,
                wall_properties_assumed_from_freestream=True,requires_pilot_yplus_check=True,
                validated_wall_resolution=False,usable_as_combustor_inlet=False,
                source='https://www.nasa.gov/wp-content/uploads/2025/09/turbulence-guide-v4-01.pdf',
                note='Incompressible fully turbulent flat-plate reference only. This does not predict leading-edge, shock, separation, compressibility or wall-temperature effects. Height is a wall-to-first-node distance, not twice a cell-centre distance. Measure wall y+ and profiles after a valid RANS pilot; do not infer grid accuracy from this estimate.')


def wall_spacing_feedback(first_height, wall_max_yplus, checks):
    """Propose a bounded first-height reduction from measured y+, without edits.

    y+ scales with distance only while wall friction and viscosity are frozen.
    This is a provisional remeshing seed, not a convergence or accuracy claim.
    """
    if isinstance(first_height, bool) or not math.isfinite(first_height) or first_height <= 0:
        raise ValueError("Invalid current first height")
    if not wall_max_yplus or any(isinstance(v, bool) or not math.isfinite(v) or v < 0
                                for v in wall_max_yplus.values()):
        raise ValueError("Invalid measured wall y-plus")
    maximum = max(wall_max_yplus.values())
    required = ("solver_success", "mesh_provenance", "settings_provenance", "physics_provenance",
                "physical_conditions", "nonphysical_points", "no_slip")
    failed = [name for name in required if not checks.get(name, False)]
    result = dict(provisional=True, current_first_height_m=first_height,
                  wall_max_y_plus=wall_max_yplus, measured_max_y_plus=maximum,
                  acceptance_y_plus=1., trial_target_y_plus=0.5, minimum_height_factor=0.25,
                  suggested_first_height_m=None, input_patch=None, source_check_failures=failed,
                  remove_input_keys=[],
                  note="Frozen-wall-state y+ scaling; no automatic coarsening. Regenerate and remeasure y+; this does not resolve boundary-layer thickness or flow convergence.")
    if failed or maximum == 0:
        return result | dict(status="withheld", reason="Source checks failed" if failed else "No positive wall-friction signal")
    if maximum <= result["acceptance_y_plus"]:
        return result | dict(status="retain_height", suggested_first_height_m=first_height, input_patch={})
    raw_factor = result["trial_target_y_plus"]/maximum
    factor = max(result["minimum_height_factor"], raw_factor)
    height = first_height*factor
    return result | dict(status="refine_height", unclipped_first_height_m=first_height*raw_factor,
                         reduction_limited=raw_factor < result["minimum_height_factor"],
                         remove_input_keys=['initial_wall_spacing'],
                         suggested_first_height_m=height, input_patch=dict(boundary_layer=dict(first_height=height)))
