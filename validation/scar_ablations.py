"""Predeclared, one-mechanism detector ablations on frozen segmented contours.

This is a development experiment, not a hyperparameter optimizer. Ground truth
is consulted only after a detector returns its candidate list and prediction.
"""
import copy
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev

from pombe_tracker.config import Config
from pombe_tracker.detection import BirthScarDetector
from pombe_tracker.geometry import compute_pca_axis, get_contour_endpoints
from pombe_tracker.poles import determine_new_pole_from_neighbors
from validation.scar_batch import metrics, classify, write_csv


def spline_geometry(contour, smooth_factor, n_points=300, arclength=False):
    """Same spline, analytic derivatives, optionally uniform smoothed arc length."""
    c=np.asarray(contour,float)
    tck,_=splprep([c[:,1],c[:,0]],s=smooth_factor,per=True)
    dense=np.linspace(0,1,max(3000,10*n_points)+1)
    xy=np.array(splev(dense,tck)).T
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    u=(np.interp(np.linspace(0,arc[-1],n_points,endpoint=False),arc,dense)
       if arclength else np.linspace(0,1,n_points,endpoint=False))
    x,y=splev(u,tck);dx,dy=splev(u,tck,der=1);ddx,ddy=splev(u,tck,der=2)
    k=(dx*ddy-dy*ddx)/np.maximum((dx*dx+dy*dy)**1.5,1e-10)
    if np.mean(k)<0:k=-k
    return np.c_[y,x],k


def periodic_geometry(contour,smooth_factor,n_points=300):
    return spline_geometry(contour,smooth_factor,n_points,arclength=True)


class AblationDetector(BirthScarDetector):
    def __init__(self,cfg,mechanism):
        super().__init__(cfg);self.mechanism=mechanism

    def _local_curvature_baseline(self,k):
        if self.mechanism=='raw_curvature': return np.zeros_like(k)
        if self.mechanism=='smooth_cap_baseline': return gaussian_filter1d(k,12,mode='wrap')
        return super()._local_curvature_baseline(k)

    def _local_curvature_excess(self,k):
        if self.mechanism=='raw_curvature': return np.maximum(k,0)
        if self.mechanism=='smooth_cap_baseline':
            return np.maximum(k-gaussian_filter1d(k,12,mode='wrap'),0)
        if self.mechanism=='concave_excess':
            return np.maximum(self._local_curvature_baseline(k)-k,0)
        if self.mechanism=='absolute_excess':
            return np.abs(k-self._local_curvature_baseline(k))
        excess=super()._local_curvature_excess(k)
        if self.mechanism=='normalized_excess':
            scale=np.array([np.median(np.abs(k[(i+np.arange(-25,26))%len(k)]-
                        np.median(k[(i+np.arange(-25,26))%len(k)]))) for i in range(len(k))])
            # A cell-wide floor prevents dividing by an arbitrarily flat side.
            floor=max(float(np.median(np.abs(k-np.median(k)))),1e-4)
            excess=excess/np.maximum(scale,floor)
        return excess

    def _collect_windowed_candidates(self,*args,**kwargs):
        candidates=super()._collect_windowed_candidates(*args,**kwargs)
        pts,k,center,axis,normal,width,u=args[:7]
        excess=self._local_curvature_excess(k)
        for c in candidates:
            if self.mechanism=='width_diagnostic':
                c['score']/=c['width_weight'];c['width_weight']=1.
            if self.mechanism=='weak_recency_prior':
                old_weight = c['recency_weight']
                c['recency_weight'] = 1.0 + .2 * old_weight
                c['score'] *= c['recency_weight'] / max(old_weight,1e-12)
            if self.mechanism=='no_recency_prior':
                c['score']/=max(c['recency_weight'],1e-12);c['recency_weight']=1.
            if self.mechanism=='require_two_features':
                i,j=c['indices'];strength=[excess[i],excess[j]]
                balance=2*min(strength)/max(sum(strength),1e-12)
                c['score']*=balance;c['bilateral_weight']=balance
            if self.mechanism=='window_excess_centroid':
                near=(np.abs(u-c['window_center']) <= self.cfg.SCAR_CURVATURE_WINDOW/2)
                near &= (u>=self.cfg.SCAR_CAP_EXCLUSION)&(u<=1-self.cfg.SCAR_CAP_EXCLUSION)
                side=(pts-center)@normal
                weights=np.zeros(len(pts))
                for sign in (-1,1):
                    members=np.flatnonzero(near&(side*sign>0))
                    if len(members):weights[members]=excess[members]/len(members)
                if weights.sum()>0:
                    position=float(np.sum(weights*u)/weights.sum())
                    indices=[]
                    for sign in (-1,1):
                        members=np.flatnonzero(side*sign>0)
                        indices.append(int(members[np.argmin(np.abs(u[members]-position))]))
                    c['original_points']=c['points']
                    c['localization_method']='excess-weighted cross-section; score held fixed'
                    c['indices']=tuple(indices);c['points']=tuple(pts[indices])
                    c['width']=float(np.linalg.norm(pts[indices[0]]-pts[indices[1]]))
            if self.mechanism=='one_feature_anchor':
                i,j=c['indices'];strong=i if excess[i]>=excess[j] else j
                side=(pts-center)@normal
                partners=np.flatnonzero(side*side[strong]<0)
                other=int(partners[np.argmin(np.abs(u[partners]-u[strong]))])
                c['indices']=(strong,other);c['points']=(pts[strong],pts[other])
                c['width']=float(np.linalg.norm(pts[strong]-pts[other]))
        return [c for c in candidates if c['score'] > 1e-10]

    def _select_best_candidate(self,candidates,new_pole_point):
        # Score ablations use a score-only control: a hard hint would mask them.
        if self.mechanism not in ('legacy','weak_neighbor_hint'): new_pole_point=None
        if self.mechanism=='weak_neighbor_hint':
            if new_pole_point is not None:
                mid=np.array([np.mean(c['points'],axis=0) for c in candidates])
                distances=np.linalg.norm(mid-new_pole_point,axis=1)
                weights=1+.2*(1-distances/max(float(distances.max()),1e-6))
                for c,w in zip(candidates,weights):
                    c['hint_weight']=float(w);c['selection_score']=float(c['score']*w)
                return candidates[int(np.argmax([c['score']*w for c,w in zip(candidates,weights)]))]
            return max(candidates,key=lambda c:c['score'])
        return super()._select_best_candidate(candidates,new_pole_point)


ABLATIONS = {
    'legacy': {},
    'ignore_neighbor_hint': {},
    'weak_neighbor_hint': {},
    'raw_curvature': {},
    'normalized_excess': {},
    'concave_excess': {},
    'absolute_excess': {},
    'no_cap_exclusion': {'SCAR_CAP_EXCLUSION':0.0},
    'smooth_cap_baseline': {'SCAR_CAP_EXCLUSION':0.0},
    'require_two_features': {},
    'one_feature_anchor': {},
    'window_excess_centroid': {},
    'no_recency_prior': {},
    'weak_recency_prior': {},
    'width_diagnostic': {},
    'analytic_derivatives': {},
    'periodic_arclength': {},
    'smoothing_half': {'SMOOTH_FACTOR':20.0},
    'smoothing_double': {'SMOOTH_FACTOR':80.0},
    'sampling_double': {'N_CONTOUR_POINTS':600, 'SCAR_N_WINDOWS':25, 'SCAR_BASELINE_SAMPLES':50},
}


def hint_for_case(case,cfg):
    r=case['result'];d=r['debug_info']
    center,axis=compute_pca_axis(d['smooth_pts'])
    poles=get_contour_endpoints(r['contour'],center,axis)
    idx,confidence=determine_new_pole_from_neighbors(r['neighbors'],poles,cfg)
    return poles[idx] if idx is not None and confidence in ('high','medium') else None


def evaluate_cases(cases, mechanism, overrides=None):
    cfg=Config()
    for k,v in (overrides or {}).items():setattr(cfg,k,v)
    detector=AblationDetector(cfg,mechanism)
    records=[];newcases=[]
    for case in cases:
        rec=copy.deepcopy(case['record']);r=copy.deepcopy(case['result'])
        if r is None:
            records.append(rec);newcases.append(case);continue
        hint=hint_for_case(case,cfg)
        if mechanism in ('analytic_derivatives','periodic_arclength'):
            geometry = periodic_geometry if mechanism == 'periodic_arclength' else spline_geometry
            with patch('pombe_tracker.detection.compute_smoothed_curvature',geometry):
                pair,dbg=detector.detect(r['contour'],hint)
        else: pair,dbg=detector.detect(r['contour'],hint)
        if mechanism != 'legacy':
            dbg['selection_rule'] = ('score × bounded hint (1–1.2)' if mechanism == 'weak_neighbor_hint'
                                     else 'maximum score; neighbor hint disabled')
        mp=None if pair is None else np.mean(pair,axis=0)
        errors=[float(np.linalg.norm(np.mean(detector._map_to_original(c['points'],r['contour']),axis=0)-case['gt']))
                for c in dbg['scar_candidates']]
        error=None if mp is None else float(np.linalg.norm(mp-case['gt']))
        oracle=min(errors) if errors else None
        rec.update(error_px=error,oracle_error_px=oracle,detected=mp is not None,
                   failure=classify(error,oracle,has_candidate=bool(errors)),
                   pred_y=None if mp is None else float(mp[0]),pred_x=None if mp is None else float(mp[1]),
                   n_candidates=len(errors))
        records.append(rec)
        dbg['curvature_excess']=detector._local_curvature_excess(dbg['kappa'])
        r.update(debug_info=dbg,scar_midpoint=mp,scar_points=pair)
        newcases.append(dict(case,record=rec,result=r))
    return records,newcases


def run_ablations(baseline_dir,out_dir):
    baseline_dir=Path(baseline_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    with (baseline_dir/'cases.pkl').open('rb') as f:cases=pickle.load(f)
    import csv
    with (baseline_dir/'per_cell.csv').open() as f:allrows=list(csv.DictReader(f))
    case_keys={(c['record']['video_id'],int(c['record']['frame']),c['record']['cell_name']) for c in cases}
    missing=[r for r in allrows if (r['video_id'],int(r['frame']),r['cell_name']) not in case_keys]
    for r in missing:
        r.update(gt_positive=r['gt_positive']=='True',detected=False,error_px=None,oracle_error_px=None)
    combined=[]
    for name,overrides in ABLATIONS.items():
        records,_=evaluate_cases(cases,name,overrides)
        records+=missing
        write_csv(out/f'{name}_per_cell.csv',records)
        m=metrics(records);combined.append(dict(ablation=name,**m))
        print(name,m,flush=True)
    write_csv(out/'summary.csv',combined)
    (out/'summary.json').write_text(json.dumps(combined,indent=2))
    return combined


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default='validation/runs/scar_v2/baseline')
    p.add_argument('--out',default='validation/runs/scar_v2/ablations')
    a=p.parse_args();run_ablations(a.baseline,a.out)
