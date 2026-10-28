#!/usr/bin/env python3
"""Multi-snapshot residual localization and experimental refinement selection.

Nodal residual ranking is not an error estimator. Coordinates/IDs are meaningful
only with the recorded mesh hash. Shear is a kinematic diagnostic, not a model.
"""
import argparse
import heapq
import json
from pathlib import Path

import numpy as np

from intake_adapt import solution, digest, sensors
from intake_euler import content_sha256, read_csv, read_volume, require, polygon, volume_output
from generate_airfoil_mesh import cell_blocks

RESIDUALS = {'Residual_Density':'rms[Rho]', 'Residual_Energy':'rms[RhoE]', 'Residual_Nu_Tilde':'rms[nu]'}
WALL_RESIDUAL_FIELDS = {'density':'Residual_Density', 'energy':'Residual_Energy', 'sa':'Residual_Nu_Tilde'}


def wall_residual_fields(fields):
    """Keep the established SA-only policy unless explicitly overridden."""
    if fields is None:
        return ('sa',)
    require(isinstance(fields,(tuple,list)) and bool(fields)
            and all(isinstance(k,str) and k in WALL_RESIDUAL_FIELDS for k in fields)
            and len(set(fields))==len(fields), 'Invalid wall residual fields')
    # Canonical order makes field order irrelevant to plan provenance.
    return tuple(k for k in WALL_RESIDUAL_FIELDS if k in fields)


def vtk_fields(path, points, wanted):
    """Read selected point arrays from SU2 volume output (VTU, or legacy ASCII VTK, plain or gzipped)."""
    return read_volume(path, points, wanted, rtol=0, atol=1e-6)


def shear_indicator(points, cells, flow):
    """h/a times planar deviatoric strain magnitude; zero for rigid rotation."""
    result={}
    for n,c in cell_blocks(cells).items():
        q=points[c];dx=q-q.mean(axis=1)[:,None,:]
        velocity=np.stack((flow['u'][c],flow['v'][c]),axis=-1)
        gradient=np.linalg.solve(dx.swapaxes(1,2)@dx,dx.swapaxes(1,2)@(velocity-velocity.mean(axis=1)[:,None,:]))
        d=.5*(gradient[:,0,0]-gradient[:,1,1])
        off=.5*(gradient[:,0,1]+gradient[:,1,0])
        strain=2*np.hypot(d,off)
        area=.5*np.sum(q[:,:,0]*np.roll(q[:,:,1],-1,axis=1)-q[:,:,1]*np.roll(q[:,:,0],-1,axis=1),axis=1)
        require(np.all(area>0), 'Nonpositive shear cell')
        result[n]=strain*np.sqrt(area)/flow['sound_speed'][c].mean(axis=1)
    return result


def largest_mask(values, fraction=.01):
    require(0<fraction<=1 and np.isfinite(values).all(), 'Invalid ranking input')
    mask=np.zeros(len(values),bool)
    count=max(1,int(np.ceil(fraction*len(values))))
    ids=np.argsort(np.abs(values),kind='stable')[-count:]
    mask[ids]=np.abs(values[ids])>0
    return mask


def nodal_dual_areas(points, cells):
    """2D median-dual areas, matching SU2 8.5.0's geometric construction.

    SetControlVolume uses arithmetic cell centers, edge midpoints, and the
    triangle areas accumulated at each endpoint (CEdge::GetVolume). Only
    counterclockwise triangles/convex quadrilaterals are supported here.
    This reconstructs geometry; it does not read solver-internal volumes.
    """
    require(points.ndim==2 and points.shape[1]==2 and np.isfinite(points).all(), 'Invalid dual-area points')
    blocks=cell_blocks(cells)
    require(set(blocks)<={3,4} and bool(blocks), 'Unsupported dual-area cells')
    area=np.zeros(len(points));total=0.
    for block in blocks.values():
        q=points[block];q=q-q[:,:1]  # avoid cancellation in tiny translated cells
        center=q.mean(axis=1)
        def cross(a,b):return a[:,0]*b[:,1]-a[:,1]*b[:,0]
        signed=.5*np.sum(q[:,:,0]*np.roll(q[:,:,1],-1,axis=1)-q[:,:,1]*np.roll(q[:,:,0],-1,axis=1),axis=1)
        require(np.all(signed>0), 'Nonpositive dual-area cell')
        total+=signed.sum()
        for i in range(block.shape[1]):
            j=(i+1)%block.shape[1];mid=.5*(q[:,i]+q[:,j])
            left=.5*cross(mid-q[:,i],center-q[:,i])
            right=.5*cross(center-q[:,j],mid-q[:,j])
            require(np.all(left>0) and np.all(right>0), 'Invalid median-dual subtriangle')
            np.add.at(area,block[:,i],left);np.add.at(area,block[:,j],right)
    require(np.all(area>0) and np.isclose(area.sum(),total,rtol=1e-12,atol=0), 'Invalid total dual area')
    return area


def residual_norms(fields, areas, mask=None):
    """Raw and area-normalized diagnostics of integrated nodal residuals.

    The area-weighted RMS of R/A is sqrt(sum(R**2/A)/sum(A)).
    A nodal mask selects whole dual areas, not exactly clipped physical cells.
    These are diagnostics, not truncation-error estimates or acceptance gates.
    """
    areas=np.asarray(areas,dtype=float)
    require(areas.ndim==1 and len(areas)>0 and np.isfinite(areas).all()
            and np.all(areas>0), 'Invalid residual norm areas')
    selected=np.ones(len(areas),bool) if mask is None else np.asarray(mask)
    require(selected.shape==areas.shape and selected.dtype==np.dtype(bool), 'Invalid residual norm mask')
    require(isinstance(fields,dict) and bool(fields), 'Missing residual norm fields')
    count=int(selected.sum());area=float(areas[selected].sum());output={}
    for key,raw in fields.items():
        raw=np.asarray(raw,dtype=float)
        require(raw.shape==areas.shape and np.isfinite(raw).all(), 'Invalid residual norm field')
        values=raw[selected]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            squared=float(np.sum(values*values))
            absolute=float(np.sum(np.abs(values)))
            weighted=float(np.sum(values*values/areas[selected]))
            rate_rms=float(np.sqrt(weighted/area)) if count else None
        require(np.isfinite([area,squared,absolute,weighted]).all()
                and (rate_rms is None or np.isfinite(rate_rms)), 'Nonfinite residual norm')
        output[key]=dict(raw_squared_sum=squared,raw_absolute_sum=absolute,
                        raw_rms=float(np.sqrt(squared/count)) if count else None,
                        per_area_weighted_rms=rate_rms)
    return dict(nodes=count,dual_area_m2=area,fields=output)


def extrema_residual_diagnostics(points, flow, fields, areas, fraction=.01):
    """Locate state extrema even when their raw residuals are not highly ranked."""
    require(areas.shape==(len(points),) and np.all(areas>0), 'Invalid residual areas')
    ranked={}
    for key in RESIDUALS:
        raw=np.abs(fields[key]);rate=raw/areas
        require(raw.shape==areas.shape and np.isfinite(rate).all(), 'Invalid residual density')
        ranked[key]=(raw,rate,largest_mask(raw,fraction),largest_mask(rate,fraction))
    output={}
    for key in ('pressure','temperature','mach'):
        output[key]={}
        for name,j in [('min',int(np.argmin(flow[key]))),('max',int(np.argmax(flow[key])))]:
            ranks={}
            for residual,(raw,rate,rawhot,ratehot) in ranked.items():
                ranks[residual]=dict(raw_magnitude=float(raw[j]),per_area_magnitude=float(rate[j]),
                    raw_descending_fraction=float(np.mean(raw>=raw[j])),
                    per_area_descending_fraction=float(np.mean(rate>=rate[j])),
                    in_raw_top_fraction=bool(rawhot[j]),in_per_area_top_fraction=bool(ratehot[j]))
            output[key][name]=dict(point_id=j,xy_m=points[j].tolist(),value=float(flow[key][j]),
                                  dual_area_m2=float(areas[j]),residuals=ranks)
    return output


def wall_projection(locations, settings):
    """Closest physical straight wall segment, never an artificial fluid split."""
    poly,names=polygon(settings)
    walls=[i for i,n in enumerate(names) if n in ('lower','cowl')]
    a=poly[walls];d=poly[(np.array(walls)+1)%len(poly)]-a
    r=locations[:,None,:]-a[None,:,:]
    t=np.clip(np.sum(r*d[None,:,:],axis=2)/np.sum(d*d,axis=1),0,1)
    projected=a[None,:,:]+t[:,:,None]*d[None,:,:]
    distances=np.linalg.norm(locations[:,None,:]-projected,axis=2)
    j=distances.argmin(axis=1)
    return projected[np.arange(len(j)),j],distances[np.arange(len(j)),j],np.array([names[walls[k]] for k in j])


def residual_wall_candidates(points, markers, settings, votes, minimum_hits, fields=None):
    """Project recurring residuals onto physical walls, retaining field reasons.

    Each field is ranked separately before this function; incompatible units
    are never summed. The search radius is a mesh heuristic, not a BL estimate.
    """
    fields=wall_residual_fields(fields)
    require(isinstance(minimum_hits,int) and not isinstance(minimum_hits,bool)
            and minimum_hits>=1, 'Invalid residual hit count')
    radius=5*settings['boundary_layer']['thickness']
    require(np.isfinite(radius) and radius>0, 'Invalid wall search radius')
    by_field={}
    for field in fields:
        key=WALL_RESIDUAL_FIELDS[field]
        require(key in votes, 'Missing wall residual votes')
        hits=np.asarray(votes[key])
        require(hits.shape==(len(points),) and np.issubdtype(hits.dtype,np.integer)
                and np.all(hits>=0), 'Invalid wall residual votes')
        locations=points[hits>=minimum_hits]
        projected,distance,names=wall_projection(locations,settings)
        by_field[field]={}
        for name in ('lower','cowl'):
            targets=projected[(names==name)&(distance<=radius)]
            edges=np.asarray(markers.get(name,[]),int).reshape(-1,2)
            require(not len(targets) or len(edges)>0, 'Projected wall lacks mesh edges')
            chosen=np.zeros(len(edges),bool)
            if len(targets):
                a=points[edges[:,0]];delta=points[edges[:,1]]-a
                length2=np.sum(delta*delta,axis=1)
                require(np.all(length2>0), 'Degenerate projected wall edge')
                for chunk in np.array_split(targets,max(1,len(targets)//100+1)):
                    offset=chunk[:,None,:]-a[None,:,:]
                    t=np.clip(np.sum(offset*delta[None,:,:],axis=2)/length2,0,1)
                    dist=np.linalg.norm(offset-t[:,:,None]*delta[None,:,:],axis=2)
                    chosen[np.argmin(dist,axis=1)]=True
            by_field[field][name]=edges[chosen].tolist()
    combined={}
    for name in ('lower','cowl'):
        keys={tuple(sorted(e)) for selection in by_field.values() for e in selection[name]}
        combined[name]=[list(map(int,e)) for e in markers.get(name,[]) if tuple(sorted(e)) in keys]
    return combined,by_field


def wall_reversal(points, cells, markers, flow, friction):
    """Joint negative skin friction and first off-wall tangential velocity.

    Signs are diagnostic only; retain magnitudes rather than certifying separation.
    Reference tangent follows positive x for the supported intake geometries.
    """
    owners={tuple(sorted(e)):q for q in cells[4] for e in zip(q,np.roll(q,-1))}
    output={}
    for name in ('lower','cowl'):
        found=[]
        for edge in markers.get(name,[]):
            require(tuple(sorted(edge)) in owners, 'Wall edge lacks a quadrilateral')
            q=owners[tuple(sorted(edge))];off=[j for j in q if j not in edge]
            tangent=points[edge[1]]-points[edge[0]];tangent/=np.linalg.norm(tangent)
            if tangent[0]<0:tangent=-tangent
            cf=float(friction[edge,:2].mean(axis=0)@tangent)
            velocity=float(np.array([flow['u'][off].mean(),flow['v'][off].mean()])@tangent)
            if cf<0 and velocity<0:
                found.append(dict(edge=list(map(int,edge)),midpoint_m=points[edge].mean(axis=0).tolist(),
                                  skin_friction_t=cf,off_wall_velocity=velocity))
        output[name]=found
    return output


def include_wall_reversals(points, markers, residual_edges, reversals, buffer_m):
    """Union residual seeds with observed reversal and a wall-arclength buffer.

    Every joint-sign reversal is retained, including tiny values. The buffer is
    an experimental mesh search distance, not an estimate of separation length.
    Graphs are separate per marker, so nearby disconnected walls never connect.
    """
    require(np.isfinite(buffer_m) and buffer_m >= 0, 'Invalid reversal buffer')
    selected={};reasons={}
    for name in ('lower','cowl'):
        edges=np.asarray(markers.get(name,[]),int).reshape(-1,2)
        keys=[tuple(sorted(map(int,e))) for e in edges]
        valid=set(keys)
        residual={tuple(sorted(map(int,e))) for e in residual_edges.get(name,[])}
        reversed_edges={tuple(sorted(map(int,r['edge']))) for rs in reversals for r in rs.get(name,[])}
        require((residual|reversed_edges)<=valid, 'Candidate is not on the named wall')
        adjacent={};lengths={}
        for a,b in keys:
            length=float(np.linalg.norm(points[b]-points[a]))
            require(length>0, 'Zero length wall edge')
            lengths[a,b]=length
            adjacent.setdefault(a,[]).append((b,length))
            adjacent.setdefault(b,[]).append((a,length))
        distance={j:0. for edge in reversed_edges for j in edge}
        queue=[(0.,j) for j in distance];heapq.heapify(queue)
        while queue:
            d,j=heapq.heappop(queue)
            if d>distance[j] or d>buffer_m:continue
            for k,length in adjacent[j]:
                nd=d+length
                if nd<=buffer_m and nd<distance.get(k,np.inf):
                    distance[k]=nd;heapq.heappush(queue,(nd,k))
        selected[name]=[];reasons[name]=[]
        for edge,key in zip(edges,keys):
            tags=[]
            if key in residual:tags.append('recurring_sa_residual')
            if key in reversed_edges:tags.append('observed_joint_sign_reversal')
            elif min(distance.get(j,np.inf) for j in key)+.5*lengths[key]<=buffer_m:
                tags.append('reversal_arclength_buffer')
            if tags:
                selected[name].append(edge.tolist())
                reasons[name].append(dict(edge=edge.tolist(),reasons=tags))
    return selected,reasons


def wall_coverage(points, markers, selected, reversal):
    """Audit an existing plan on the same mesh, retaining uncovered locations."""
    missing={};coverage={}
    for name in ('lower','cowl'):
        valid={tuple(sorted(map(int,e))) for e in markers.get(name,[])}
        chosen={tuple(sorted(map(int,e))) for e in selected.get(name,[])}
        observed=reversal.get(name,[])
        require(chosen<=valid and all(tuple(sorted(r['edge'])) in valid for r in observed),
                'Coverage edge is not on the named wall')
        missing[name]=[r for r in observed if tuple(sorted(r['edge'])) not in chosen]
        def length(records):
            return float(sum(np.linalg.norm(points[r['edge'][1]]-points[r['edge'][0]]) for r in records))
        total,uncovered=length(observed),length(missing[name])
        coverage[name]=dict(observed_edges=len(observed),uncovered_edges=len(missing[name]),
            observed_arclength_m=total,uncovered_arclength_m=uncovered,
            uncovered_arclength_fraction=uncovered/total if total else 0.)
    return dict(walls=coverage,uncovered_reversal=missing,
                all_observed_reversals_covered=not any(missing.values()))


def audit_plan(case, result, plan_path):
    """Check a frozen plan against a later valid solution without changing it."""
    plan=json.loads(plan_path.read_text())
    s,p,c,m,data,flow,record=solution(case,result)
    require(s.get('mode')=='rans','Requires RANS')
    require(plan['mesh_sha256']==record['mesh_sha256'] and
            plan['physics_sha256']==record['physics_sha256'],'Coverage plan mesh or physics differs')
    fields=vtk_fields(volume_output(result/'second'),p,{'Skin_Friction_Coefficient'})
    reverse=wall_reversal(p,c,m,flow,fields['Skin_Friction_Coefficient'])
    coverage=wall_coverage(p,m,plan['wall_edges'],reverse)
    return dict(**coverage,provisional=True,plan_sha256=digest(plan_path),result=str(result),
        source={**{k:record[k] for k in ('mesh_sha256','physics_sha256','config_sha256','binary_sha256')},
                'restart_sha256':digest(result/'second/restart_flow.csv'),
                'vtk_sha256':content_sha256(volume_output(result/'second'))},
        note='Coverage of observed joint-sign reversals only. The plan is unchanged. Tiny negative values are retained; coverage does not certify physical separation, mesh adequacy or convergence.')


def diagnose(case, results, fraction=.01, minimum_hits=2, shear_threshold=.03, reversal_buffer_layers=2.,
             wall_fields=None):
    wall_fields=wall_residual_fields(wall_fields)
    require(len(results)>=2 and 1<=minimum_hits<=len(results), 'Need multiple snapshots and valid hit count')
    require(len({r.resolve() for r in results})==len(results), 'Repeated snapshot')
    require(np.isfinite(shear_threshold) and shear_threshold>0, 'Invalid shear threshold')
    require(np.isfinite(reversal_buffer_layers) and reversal_buffer_layers>=0, 'Invalid reversal buffer')
    snapshots=[];votes=None;shear_votes=None;shock_votes=None;areas=None
    for result in results:
        s,p,c,m,data,flow,record=solution(case,result)
        require(s.get('mode')=='rans' and 'mesh_policy' in s, 'Requires RANS with geometry mesh policy')
        fields=vtk_fields(volume_output(result/'second'),p,set(RESIDUALS)|{'Skin_Friction_Coefficient'})
        history=read_csv(result/'second/history.csv')
        shear=shear_indicator(p,c,flow);shock=sensors(p,c,flow)
        if votes is None:
            areas=nodal_dual_areas(p,c)
            votes={k:np.zeros(len(p),int) for k in RESIDUALS}
            shear_votes={n:np.zeros(len(b),int) for n,b in c.items()}
            shock_votes={n:np.zeros(len(b),int) for n,b in c.items()}
        on_quads=np.zeros(len(p),bool);on_quads[np.unique(c[4])]=True
        residual={}
        for key,hkey in RESIDUALS.items():
            value=fields[key];sq=value*value;total=sq.sum()
            rms=float(np.log10(np.sqrt(np.mean(sq)))) if total else -np.inf
            require(np.isfinite(rms) and abs(rms-history[hkey][-1])<1e-5, 'Residual field/history mismatch')
            hot=largest_mask(value,fraction);votes[key]+=hot
            j=int(np.argmax(sq))
            residual[key]=dict(log_rms=rms,top_fraction_squared_share=float(sq[hot].sum()/total),
                quad_nodes_squared_share=float(sq[on_quads].sum()/total),maximum_at_m=p[j].tolist())
        for n in c:
            shear_votes[n]+=shear[n]>=shear_threshold
            shock_votes[n]+=shock[n]['selected']
        snapshots.append(dict(result=str(result),residuals=residual,
            state_extrema=extrema_residual_diagnostics(p,flow,fields,areas,fraction),
            wall_reversal=wall_reversal(p,c,m,flow,fields['Skin_Friction_Coefficient']),
            source={**{k:record[k] for k in ('mesh_sha256','physics_sha256','config_sha256','binary_sha256')},
                    'restart_sha256':digest(result/'second/restart_flow.csv'), 'vtk_sha256':content_sha256(volume_output(result/'second'))}))
    persistent=np.logical_or.reduce([v>=minimum_hits for v in votes.values()])
    near_persistent=np.any(persistent[c[3]],axis=1)
    shear_any=shear_votes[3]>0
    # Residual recurrence plus an independently observed strain feature. The
    # union of observed strain allows modest movement between checkpoints.
    internal=near_persistent & shear_any
    # A diagnostic search radius derived from the existing mesh layer, not a
    # claim that this is the physical boundary-layer thickness.
    radius=5*s['boundary_layer']['thickness']
    residual_wall_edges,by_field=residual_wall_candidates(p,m,s,votes,minimum_hits,wall_fields)
    reversal_buffer=reversal_buffer_layers*s['boundary_layer']['thickness']
    wall_edges,wall_reasons=include_wall_reversals(p,m,residual_wall_edges,
        [snapshot['wall_reversal'] for snapshot in snapshots],reversal_buffer)
    for name,records in wall_reasons.items():
        field_edges={field:{tuple(sorted(e)) for e in selected[name]} for field,selected in by_field.items()}
        for row in records:
            if 'recurring_sa_residual' in row['reasons']:
                row['reasons']=[f'recurring_{field}_residual' for field,keys in field_edges.items()
                                if tuple(sorted(row['edge'])) in keys]+[
                                tag for tag in row['reasons'] if tag!='recurring_sa_residual']
    selected_wall_set={tuple(sorted(e)) for es in wall_edges.values() for e in es}
    missed_reversal={name:[r for r in rs if tuple(sorted(r['edge'])) not in selected_wall_set]
                     for name,rs in snapshots[-1]['wall_reversal'].items()}
    recurrence={key:dict(nodes_at_least_minimum=int((v>=minimum_hits).sum()),
                            node_ids=np.flatnonzero(v>=minimum_hits).tolist()) for key,v in votes.items()}
    return dict(provisional=True,mesh_sha256=digest(case/'mesh.su2'),physics_sha256=record['physics_sha256'],
        thresholds=dict(top_node_fraction=fraction,minimum_hits=minimum_hits,shear=shear_threshold,wall_search_radius_m=radius,
                        reversal_buffer_layers=reversal_buffer_layers,reversal_buffer_m=reversal_buffer,
                        wall_residual_fields=list(wall_fields)),
        snapshots=snapshots,recurrence=recurrence,internal_triangle_ids=np.flatnonzero(internal).tolist(),
        dual_area_diagnostic=dict(total_m2=float(areas.sum()),minimum_m2=float(areas.min()),
            method='2D median dual: arithmetic cell centers, edge midpoints, endpoint subtriangles; SU2 8.5.0 SetControlVolume geometry. Reconstructed, not read back from solver.',
            note='Per-area residual rankings at state extrema are supplementary diagnostics only. They do not alter refinement selection or certify truncation error, positivity, or convergence.'),
        wall_edges=wall_edges,wall_edge_reasons=wall_reasons,residual_only_wall_edges=residual_wall_edges,
        residual_wall_edges_by_field=by_field,
        latest_reversal_outside_wall_selection=missed_reversal,counts=dict(internal_triangles=int(internal.sum()),persistent_residual_nodes=int(persistent.sum()),
        shear_any_triangles=int(shear_any.sum()),wall_edges=sum(map(len,wall_edges.values())),
        residual_only_wall_edges=sum(map(len,residual_wall_edges.values())),
        latest_reversal_edges_outside_selection=sum(map(len,missed_reversal.values()))),
        note='Snapshot recurrence is not temporal convergence or truncation error. Strain also detects anisotropic compression. Wall candidates include all joint-sign reversals in the supplied snapshots without a recurrence requirement, including tiny negative values. Buffers follow connected wall edges and use requested mesh-layer thickness, not physical boundary-layer thickness. Last supplied snapshot is called latest. Tangential refinement quality checks remain required.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case',type=Path);ap.add_argument('results',type=Path,nargs='+')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reversal-buffer-layers',type=float,default=2.)
    ap.add_argument('--wall-residual-fields',nargs='+',choices=list(WALL_RESIDUAL_FIELDS),
                    help='Experimental wall candidates from separately ranked residual fields; default: sa')
    ap.add_argument('--audit-plan',type=Path,help='Audit a frozen plan against exactly one solution; do not regenerate candidates')
    a=ap.parse_args()
    if a.audit_plan:
        require(len(a.results)==1,'Coverage audit requires exactly one result')
        require(a.wall_residual_fields is None,'Frozen-plan audit cannot change wall residual fields')
        report=audit_plan(a.case,a.results[0],a.audit_plan)
    else:
        report=diagnose(a.case,a.results,reversal_buffer_layers=a.reversal_buffer_layers,
                        wall_fields=a.wall_residual_fields)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report.get('counts',report.get('walls')),indent=2))


if __name__=='__main__':main()
