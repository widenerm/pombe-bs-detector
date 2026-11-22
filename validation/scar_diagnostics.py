"""Inspectable candidate diagnostics and failure gallery, using saved crops."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pombe_tracker.detection import BirthScarDetector


def plot_case(case, cfg):
    from pombe_tracker.visualization import plot_scar_diagnostics
    rec=case['record']
    title=f"{rec['video_id']} / {rec['frame']} / {rec['cell_name']} | {rec['failure']} | error={rec['error_px']}"
    if case['result'] is None:
        fig,ax=plt.subplots(figsize=(7,7));origin=case['origin'];crop=case['crop']
        ax.imshow(crop,cmap='gray',extent=(origin[1],origin[1]+crop.shape[1],origin[0]+crop.shape[0],origin[0]))
        for c in case.get('nearby_contours',[]):ax.plot(c[:,1],c[:,0],color='cyan',lw=.7)
        ax.plot(case['gt'][1],case['gt'][0],'r+',ms=12)
        ax.set_xlim(origin[1],origin[1]+crop.shape[1]);ax.set_ylim(origin[0]+crop.shape[0],origin[0])
        ax.set_title(title,fontsize=8);return fig
    return plot_scar_diagnostics(case['crop'], case['result'], cfg, case['gt'], case['origin'], title)


def gallery(cases,out,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sections=[]
    for case in sorted(cases,key=lambda c:float('inf') if c['record']['error_px'] is None else c['record']['error_px'],reverse=True):
        rec=case['record'];name=f"{rec['video_id']}_{rec['frame']}_{rec['cell_name']}"
        fig=plot_case(case,cfg);fig.savefig(out/f'{name}.png',dpi=110);plt.close(fig)
        candidates=[]
        for i,c in enumerate(case['result']['debug_info']['scar_candidates'] if case['result'] is not None else []):
            candidate = {k:np.asarray(v).tolist() if isinstance(v,(np.ndarray,tuple)) else v for k,v in c.items()}
            candidate.setdefault('candidate_id', i)
            candidates.append(candidate)
        result=case['result']
        details=dict(candidates=candidates,ground_truth=case['gt'],record=rec,
                     contour=None if result is None else result['contour'],
                     debug=None if result is None else {k:v for k,v in result['debug_info'].items() if k!='scar_candidates'})
        (out/f'{name}.json').write_text(json.dumps(details,indent=2,default=lambda v:np.asarray(v).tolist()))
        sections.append(f'<h2>{html.escape(name)}: {html.escape(rec["failure"])}</h2><a href="{name}.json">Exact candidate scores / coordinates</a><img width="1400" src="{name}.png">')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Scar diagnostics</title>'+''.join(sections))
