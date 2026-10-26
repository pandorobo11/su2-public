#!/usr/bin/env python3
"""Prepare a positive, same-physics RANS initial guess on another mesh.

Conservative variables are interpolated with convex weights, but global
conservation on an arbitrary remesh is not guaranteed. This is initialization,
not a converged solution or a same-mesh restart.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from generate_airfoil_mesh import read_su2
from intake_adapt import solution,digest
from intake_euler import require,physics_signature,polygon,primitives,read_csv,plot_triangles
from intake_convergence import check_completed_block
from intake_regions import nodal_dual_areas

FIELDS=('Density','Momentum_x','Momentum_y','Energy','Nu_Tilde')


def interpolate(points,cells,markers,data,target):
    from matplotlib.tri import Triangulation
    triangles=plot_triangles(cells)
    require(np.isfinite(target).all(),'Nonfinite target coordinates')
    source_values=np.column_stack([data[k] for k in FIELDS])
    require(source_values.shape==(len(points),len(FIELDS)) and np.isfinite(source_values).all(),'Invalid source fields')
    direct={tuple(xy):i for i,xy in enumerate(points)}
    ids=np.array([direct.get(tuple(xy),-1) for xy in target])
    result=np.empty((len(target),len(FIELDS)));matched=ids>=0
    result[matched]=source_values[ids[matched]]
    pending=np.flatnonzero(~matched);boundary_fallback=0
    if len(pending):
        found=Triangulation(points[:,0],points[:,1],triangles).get_trifinder()(target[pending,0],target[pending,1])
        inside=found>=0;selected=pending[inside];tri=triangles[found[inside]]
        q=points[tri];v1=q[:,1]-q[:,0];v2=q[:,2]-q[:,0];r=target[selected]-q[:,0]
        cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
        det=cross(v1,v2);require(np.all(det>0),'Nonpositive interpolation triangle')
        b=cross(r,v2)/det;c=cross(v1,r)/det
        weights=np.column_stack((1-b-c,b,c))
        require(np.all(weights>=-1e-10),'Nonconvex interpolation weights')
        weights=np.maximum(weights,0);weights/=weights.sum(axis=1)[:,None]
        result[selected]=np.einsum('ij,ijk->ik',weights,source_values[tri])
        # Exact boundary midpoints can fall just outside a floating-point
        # triangle locator. Only project onto an existing physical edge within
        # roundoff-scale distance; never extrapolate into the fluid exterior.
        missing=pending[~inside]
        edges=np.vstack(list(markers.values()));a=points[edges[:,0]];d=points[edges[:,1]]-a
        length2=np.sum(d*d,axis=1)
        tol=64*np.finfo(float).eps*max(float(np.ptp(points,axis=0).max()),1.)
        for j in missing:
            t=np.clip(np.sum((target[j]-a)*d,axis=1)/length2,0,1)
            dist=np.linalg.norm(a+t[:,None]*d-target[j],axis=1);k=int(np.argmin(dist))
            require(dist[k]<=tol,'Target point is outside source domain')
            result[j]=(1-t[k])*source_values[edges[k,0]]+t[k]*source_values[edges[k,1]]
            boundary_fallback+=1
    require(np.isfinite(result).all(),'Nonfinite interpolated fields')
    return {k:result[:,i] for i,k in enumerate(FIELDS)},dict(exact_source_nodes=int(matched.sum()),
        interpolated_nodes=int((~matched).sum()),roundoff_boundary_projections=boundary_fallback)


def prepare(source_case,source_result,target_case,output):
    source_case=source_case.resolve();source_result=source_result.resolve();target_case=target_case.resolve();output=output.resolve()
    require(not output.exists(),'Output already exists')
    s,p,c,m,data,flow,record=solution(source_case,source_result)
    require(s.get('mode')=='rans' and record['second_order'],'Requires second-order RANS source')
    check_completed_block(read_csv(source_result/'second/history.csv'),record['iterations'])
    target_settings=json.loads((target_case/'settings.json').read_text())
    require(physics_signature(s)==physics_signature(target_settings),'Initial guess physical conditions differ')
    require(np.array_equal(polygon(s)[0],polygon(target_settings)[0]),'Initial guess geometry differs')
    tp,tc,tm=read_su2(target_case/'mesh.su2')
    require(set(m)==set(tm),'Boundary markers differ')
    require(np.all(data['Nu_Tilde']>=0),'Negative standard-SA source')
    transferred,counts=interpolate(p,c,m,data,tp)
    state=primitives(transferred,target_settings)
    require(np.all(transferred['Nu_Tilde']>=0),'Negative transferred standard-SA state')
    source_area=nodal_dual_areas(p,c);target_area=nodal_dual_areas(tp,tc)
    require(np.isclose(source_area.sum(),target_area.sum(),rtol=1e-12,atol=0),'Domain area differs')
    integrals={}
    for k in FIELDS:
        old=float(source_area@data[k]);new=float(target_area@transferred[k]);scale=float(source_area@np.abs(data[k]))
        integrals[k]=dict(source=old,target=new,change_normalized_by_absolute_source=abs(new-old)/scale if scale else (0. if new==old else None))
    output.mkdir(parents=True)
    csv=output/'restart_flow.csv'
    with csv.open('x') as f:
        f.write(','.join('"'+k+'"' for k in ('PointID','x','y')+FIELDS)+'\n')
        for j,xy in enumerate(tp):
            f.write(str(j)+','+','.join(format(float(v),'.17g') for v in [*xy,*[transferred[k][j] for k in FIELDS]])+'\n')
    report=dict(kind='same_physics_mesh_initial_guess',provisional=True,**counts,
        source=dict(case=str(source_case),result=str(source_result),mesh_sha256=record['mesh_sha256'],
            run_sha256=digest(source_result/'second/run.json'),restart_sha256=digest(source_result/'second/restart_flow.csv'),
            binary_sha256=record['binary_sha256']),
        target=dict(case=str(target_case),mesh_sha256=digest(target_case/'mesh.su2'),settings_sha256=digest(target_case/'settings.json')),
        physics_sha256=record['physics_sha256'],numerics=record.get('numerics',{}),csv_sha256=digest(csv),
        minimum_density=float(state['density'].min()),minimum_pressure=float(state['pressure'].min()),
        minimum_temperature=float(state['temperature'].min()),area_integrals=integrals,
        usable_as_combustor_inlet=False,
        note='Convex interpolation of stored conservative flow variables and SA scalar, using a triangulation of the source cells. Boundary roundoff projection is recorded. Positive initialization only: global integral drift is reported, not corrected; recompute residuals and convergence on the target mesh. Use ASCII restart input, not a copied binary from another mesh.')
    (output/'initial_guess.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def validate_guess(case,folder):
    report=json.loads((folder/'initial_guess.json').read_text())
    require(report.get('kind')=='same_physics_mesh_initial_guess','Unknown initial guess kind')
    s=json.loads((case/'settings.json').read_text())
    require(report['target']['mesh_sha256']==digest(case/'mesh.su2'),'Initial guess target mesh differs')
    require(report['target']['settings_sha256']==digest(case/'settings.json'),'Initial guess target settings differ')
    require(report['physics_sha256']==physics_signature(s),'Initial guess physical conditions differ')
    require(report['csv_sha256']==digest(folder/'restart_flow.csv'),'Initial guess data provenance differs')
    p,_,_=read_su2(case/'mesh.su2');data=read_csv(folder/'restart_flow.csv')
    require(list(data)==['PointID','x','y',*FIELDS],'Initial guess column layout differs')
    require(np.array_equal(data['PointID'],np.arange(len(p))),'Initial guess point IDs differ')
    require(np.array_equal(np.column_stack([data['x'],data['y']]),p),'Initial guess coordinates differ')
    primitives(data,s);require(np.all(data['Nu_Tilde']>=0),'Negative standard-SA initial guess')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source_case','source_result','target_case','output'):parser.add_argument(key,type=Path)
    a=parser.parse_args();print(json.dumps(prepare(a.source_case,a.source_result,a.target_case,a.output),indent=2))


if __name__=='__main__':main()
