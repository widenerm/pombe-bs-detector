"""Frame-only scar validation. No tracker or temporal stabilization is run.

GT coordinates are used ONLY to associate a label after independent detection.
Precision at 5 px is conditional on the annotated cells; unannotated cells are
not negatives. Missing predictions remain in the recall denominator.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from skimage.measure import regionprops, find_contours

from pombe_tracker.config import Config
from pombe_tracker.detection import BirthScarDetector
from pombe_tracker.pipeline import CellProcessor, filter_valid_cells, prepare_cell_info


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def metrics(rows):
    positives = [r for r in rows if r['gt_positive']]
    errors = [r['error_px'] for r in positives if r['error_px'] is not None]
    detected = sum(r['detected'] for r in positives)
    hits = sum(e <= 5 for e in errors)
    return dict(n=len(positives), detected=detected, within_5=hits,
                detection_recall=detected / len(positives) if positives else None,
                recall_5px=hits / len(positives) if positives else None,
                precision_5px_annotated=hits / detected if detected else None,
                population_detection_precision=None,
                fraction_within_5px=hits / len(positives) if positives else None,
                median_error_px=float(np.median(errors)) if errors else None,
                mae_px=float(np.mean(errors)) if errors else None,
                p90_error_px=float(np.percentile(errors, 90)) if errors else None,
                oracle_within_5=sum(r.get('oracle_error_px') is not None and
                                    r['oracle_error_px'] <= 5 for r in positives),
                failures={k: sum(r['failure'] == k for r in positives) for k in
                          ['segmentation failure', 'cell-association failure',
                           'no-candidate failure', 'wrong-candidate selection', 'localization error']})


def classify(error, oracle, association='ok', has_candidate=True):
    if association != 'ok':
        return association
    if not has_candidate:
        return 'no-candidate failure'
    if error is not None and error <= 5:
        return 'success'
    if oracle is not None and oracle <= 5:
        return 'wrong-candidate selection'
    return 'localization error'


def geometry_diagnostics(result, point, cap=.12):
    """Contour-only cross-section oracle; diagnostic only, never a prediction."""
    d=result['debug_info'];center=d['center'];axis=d['axis']
    normal=np.array([-axis[1],axis[0]])
    c=np.asarray(result['contour'])
    proj=(c-center)@axis;trans=(c-center)@normal
    low,high=float(proj.min()),float(proj.max())
    gt_position=float((point-center)@axis)
    gt_u=(gt_position-low)/max(high-low,1e-10)
    mids=[];positions=[]
    # Include the exact annotated longitudinal position only to measure the
    # transverse centering discrepancy, never to create detector candidates.
    for pos in np.r_[np.linspace(low,high,501),gt_position]:
        b=np.roll(proj,-1);tb=np.roll(trans,-1)
        cross=((proj<=pos)&(b>pos))|((b<=pos)&(proj>pos))
        if cross.sum()<2:continue
        ys=trans[cross]+(pos-proj[cross])/(b[cross]-proj[cross])*(tb[cross]-trans[cross])
        mids.append(center+pos*axis+.5*(ys.min()+ys.max())*normal)
        positions.append((pos-low)/max(high-low,1e-10))
    mids=np.asarray(mids);positions=np.asarray(positions)
    distances=np.linalg.norm(mids-point,axis=1) if len(mids) else np.array([])
    eligible=(positions>=cap)&(positions<=1-cap)
    steps=np.linalg.norm(np.roll(d['smooth_pts'],-1,axis=0)-d['smooth_pts'],axis=1)
    return dict(gt_long_norm=gt_u,gt_nearest_pole_fraction=min(gt_u,1-gt_u),
                gt_in_excluded_cap=bool(gt_u<cap or gt_u>1-cap),
                cross_section_oracle_px=float(distances.min()) if len(distances) else None,
                cap_cross_section_oracle_px=float(distances[eligible].min()) if eligible.any() else None,
                contour_step_cv=float(steps.std()/max(steps.mean(),1e-10)),
                contour_step_mean_px=float(steps.mean()))


def associate(labels, valid, gt):
    """Containment only; never pick the prediction nearest to the GT scar."""
    p = np.array([float(gt['gt_scar_mid_y_px']), float(gt['gt_scar_mid_x_px'])])
    y, x = np.rint(p).astype(int)
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return None, 'cell-association failure'
    label = int(labels[y, x])
    if label == 0 or label not in valid:
        return label, 'segmentation failure'
    return label, 'ok'


def validate_inputs(h5_dir, gt_path):
    with open(gt_path, newline='') as f:
        gt = [r for r in csv.DictReader(f) if r['scorer_initials'].upper() != 'EXAMPLE']
    keys = [(r['video_id'], int(r['frame']), r['cell_name']) for r in gt]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate annotation keys')
    files = sorted(Path(h5_dir).glob('*.h5'))
    frame_sets = {}
    for p in files:
        with h5py.File(p) as f:
            nums = f['frame_numbers'][:]
            if len(nums) != len(set(nums)) or len(nums) != len(f['frames']):
                raise ValueError(f'Invalid frame_numbers in {p}')
            frame_sets[p.stem] = set(map(int, nums))
    for video, frame, _ in keys:
        if frame not in frame_sets.get(video, set()):
            raise ValueError(f'Annotation frame absent from packages: {video}/{frame}')
    return files, gt


def segmentation_signature(provenance, allow_runtime_variation=False):
    """Compare model/settings exactly, optionally across paths/devices.

    The original v2 caches used Cellpose 4.0.9's default bfloat16 precision
    before the dtype field was recorded. No other legacy dtype is inferred.
    """
    signature = dict(provenance)
    if 'dtype' not in signature:
        if signature.get('cellpose') == '4.0.9':
            signature['dtype'] = 'bfloat16'
        else:
            signature['dtype'] = 'unknown'
    if allow_runtime_variation:
        signature.pop('model', None)
        signature.pop('device', None)
        # CUDA wheel suffixes differ from macOS, but the release must match.
        if 'torch' in signature:
            signature['torch'] = signature['torch'].split('+')[0]
    return signature


def run_batch(h5_dir, gt_path, cache_dir, out_dir, overrides=None,
              gallery=False, segment_missing=False, segmentation_batch_size=1, cache_wait_seconds=0, anchors_path=None,
              segmentation_precision='bfloat16', allow_runtime_variation=False,
              expected_model_sha256=None):
    """Run all packages using verified full-frame masks, saving exhaustive rows."""
    if segmentation_precision not in ('bfloat16', 'float32'):
        raise ValueError('Segmentation precision must be bfloat16 or float32')
    cfg = Config()
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            raise ValueError(f'Unknown configuration: {k}')
        setattr(cfg, k, v)
    files, gt = validate_inputs(h5_dir, gt_path)
    source_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [*Path(__file__).resolve().parents[1].joinpath('pombe_tracker').glob('*.py'),Path(__file__)]}
    anchors = {}
    if anchors_path is not None:
        with open(anchors_path) as f:
            for a in csv.DictReader(f):
                key=(a['video_id'],int(a['frame']),a['cell_name'])
                if key in anchors: raise ValueError('Duplicate cell anchor')
                anchors[key]=a
        if any((g['video_id'],int(g['frame']),g['cell_name']) not in anchors for g in gt):
            raise ValueError('Missing independent cell anchor')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    processor = CellProcessor(cfg)
    rows, predictions, cases = [], [], []
    segmenter = None
    runtime_provenance = None
    provenance_seen = None
    provenance_variants = []
    cache_manifest = []
    for path in files:
        video = path.stem
        with h5py.File(path) as f:
            for i, number in enumerate(f['frame_numbers'][:]):
                number = int(number)
                image = f['frames'][i]
                cache = Path(cache_dir) / video / f'{number:04d}.npz'
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if cache_wait_seconds and not segment_missing:
                    import time
                    deadline = time.monotonic() + cache_wait_seconds
                    while not cache.exists() and time.monotonic() < deadline:
                        time.sleep(2)
                    if cache.exists(): time.sleep(1)  # writer may still be flushing
                if not cache.exists() and segment_missing:
                    if segmenter is None:
                        from pombe_tracker.segmentation import CellposeSegmenter
                        from importlib.metadata import version
                        if segmentation_precision == 'bfloat16':
                            segmenter = CellposeSegmenter(cfg)
                        else:
                            from cellpose import models, core
                            from types import SimpleNamespace
                            segmenter = SimpleNamespace(model=models.CellposeModel(
                                gpu=core.use_gpu(), use_bfloat16=False))
                        model = Path(segmenter.model.pretrained_model)
                        provenance = dict(cellpose=version('cellpose'), torch=version('torch'),
                                          model=str(model), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                                          diameter=cfg.CELLPOSE_DIAMETER, channels=[0, 0],
                                          batch_size=segmentation_batch_size, device=str(segmenter.model.device),
                                          dtype=segmentation_precision)
                        if expected_model_sha256 and provenance['model_sha256'] != expected_model_sha256:
                            raise ValueError('Downloaded Cellpose weights differ from the pinned audit model')
                        if provenance_seen is not None and segmentation_signature(provenance, allow_runtime_variation) != segmentation_signature(provenance_seen, allow_runtime_variation):
                            raise ValueError('Runtime model/settings differ from existing cache; use a separate cache')
                        runtime_provenance = provenance
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    mask, _, _ = segmenter.model.eval(image, diameter=cfg.CELLPOSE_DIAMETER,
                                                       channels=[0, 0], batch_size=segmentation_batch_size)
                    temporary_cache = cache.with_suffix('.partial.npz')
                    np.savez_compressed(temporary_cache, labels=mask, image_sha256=digest,
                                        provenance=json.dumps(runtime_provenance, sort_keys=True))
                    temporary_cache.replace(cache)
                with np.load(cache, allow_pickle=False) as z:
                    if str(z['image_sha256']) != digest:
                        raise ValueError(f'Stale image cache: {cache}')
                    provenance = json.loads(str(z['provenance']))
                    if provenance['diameter'] != cfg.CELLPOSE_DIAMETER:
                        raise ValueError(f'Stale segmentation settings: {cache}')
                    signature = segmentation_signature(provenance, allow_runtime_variation)
                    if signature['dtype'] not in ('unknown', segmentation_precision):
                        raise ValueError(f'Cached numeric precision differs from requested precision: {cache}')
                    if expected_model_sha256 and provenance.get('model_sha256') != expected_model_sha256:
                        raise ValueError(f'Cached model differs from pinned audit model: {cache}')
                    if provenance_seen is not None and segmentation_signature(provenance_seen, allow_runtime_variation) != signature:
                        raise ValueError('Mixed segmentation model provenance')
                    provenance_seen = provenance
                    if provenance not in provenance_variants:
                        provenance_variants.append(provenance)
                    cache_manifest.append(dict(video_id=video, frame=number,
                        runtime_variant=provenance_variants.index(provenance),
                        mask_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
                    labels = z['labels']
                if labels.shape != image.shape:
                    raise ValueError(f'Mask shape mismatch: {cache}')
                valid = filter_valid_cells(regionprops(labels), image.shape, cfg)
                info = prepare_cell_info(list(valid.values()), labels, cfg)
                results = {lab: processor.process_cell(r, labels, image, info) for lab, r in valid.items()}
                for lab, result in results.items():
                    if result is None: continue
                    mp = result.get('scar_midpoint')
                    predictions.append(dict(video_id=video, frame=number, label=lab,
                                            pred_y=None if mp is None else float(mp[0]),
                                            pred_x=None if mp is None else float(mp[1]),
                                            detected=mp is not None, area=result['area']))
                used = set()
                for g in [g for g in gt if g['video_id'] == video and int(g['frame']) == number]:
                    positive = bool(g['gt_scar_mid_x_px'].strip() and g['gt_scar_mid_y_px'].strip())
                    rec = dict(video_id=video, frame=number, cell_name=g['cell_name'],
                               gt_positive=positive, notes=g['notes'], label=None, detected=False,
                               error_px=None, oracle_error_px=None, failure='cell-association failure')
                    if not positive:
                        rec['failure'] = 'unscored: no independent cell anchor'
                        rows.append(rec); continue
                    label, status = associate(labels, valid, g)
                    rec['gt_containment_label'] = label
                    anchor = anchors.get((video,number,g['cell_name']))
                    if anchor is not None:
                        anchor_label, anchor_status = associate(labels,valid,
                            {'gt_scar_mid_y_px':anchor['anchor_y'],'gt_scar_mid_x_px':anchor['anchor_x']})
                        rec['anchor_label'] = anchor_label
                        rec['association_method'] = 'independent historical pole midpoint'
                        if anchor_status != 'ok':
                            status = 'segmentation failure'
                        elif label and label != anchor_label:
                            status = 'cell-association failure'
                        label = anchor_label
                    else:
                        rec['association_method'] = 'GT containment (conditional)'
                    if label in used: status = 'cell-association failure'
                    used.add(label)
                    result = results.get(label)
                    if result is None and status == 'ok': status = 'segmentation failure'
                    point = np.array([float(g['gt_scar_mid_y_px']), float(g['gt_scar_mid_x_px'])])
                    rec.update(label=label, gt_y=float(point[0]), gt_x=float(point[1]))
                    if result is not None and status == 'ok':
                        dbg = result['debug_info']; contour = result['contour']
                        mp = result.get('scar_midpoint')
                        errors = [float(np.linalg.norm(np.mean(processor.detector._map_to_original(c['points'], contour),axis=0)-point))
                                  for c in dbg['scar_candidates']]
                        error = float(np.linalg.norm(mp-point)) if mp is not None else None
                        rec.update(detected=mp is not None, error_px=error,
                                   oracle_error_px=min(errors) if errors else None,
                                   pred_y=float(mp[0]) if mp is not None else None,
                                   pred_x=float(mp[1]) if mp is not None else None,
                                   n_candidates=len(errors), seg_quality=result['seg_quality'],
                                   area_ratio=result['area']/float(g['gt_area_px2']),
                                   pole_method=dbg['pole_method'], pole_confidence=dbg['pole_confidence'])
                        rec.update(geometry_diagnostics(result,point,cfg.SCAR_CAP_EXCLUSION))
                        if errors:
                            oracle_id=int(np.argmin(errors))
                            rec['oracle_candidate_id']=oracle_id
                            oracle_score=dbg['scar_candidates'][oracle_id]['score']
                            rec['oracle_score_rank']=1+sum(c['score']>oracle_score for c in dbg['scar_candidates'])
                        result['cell_name'] = g['cell_name']
                        # Save cropped raw pixels only in ignored run outputs, not report files.
                        lo=np.maximum(0,np.floor(contour.min(axis=0)).astype(int)-25)
                        hi=np.minimum(image.shape,np.ceil(contour.max(axis=0)).astype(int)+26)
                        cases.append(dict(record=rec,result=result,gt=point,
                                          crop=image[lo[0]:hi[0],lo[1]:hi[1]],origin=lo))
                    if result is None or status != 'ok':
                        lo=np.maximum(0,np.floor(point).astype(int)-150)
                        hi=np.minimum(image.shape,np.ceil(point).astype(int)+151)
                        nearby = [r['contour'] for r in results.values() if r is not None
                                  and np.linalg.norm(np.asarray(r['centroid'])-point) < 250]
                        cases.append(dict(record=rec,result=None,gt=point,
                                          crop=image[lo[0]:hi[0],lo[1]:hi[1]],origin=lo,
                                          nearby_contours=nearby))
                    rec['failure'] = classify(rec['error_px'],rec['oracle_error_px'],status,
                                              bool(rec.get('n_candidates')))
                    rows.append(rec)
                # Incremental diagnostics while a separate segmentation job fills the cache.
                import pickle
                with (out/'cases_progress.pkl').open('wb') as progress:
                    pickle.dump(cases, progress)
                write_csv(out/'per_cell_progress.csv', rows)
                print(video, number, 'processed', len(results), flush=True)
    write_csv(out/'per_cell.csv',rows)
    write_csv(out/'predictions_all_cells.csv',predictions)
    write_csv(out/'outside_5px.csv',[r for r in rows if r['gt_positive'] and r['failure']!='success'])
    summary={'combined':metrics(rows),'per_video':{p.stem:metrics([r for r in rows if r['video_id']==p.stem]) for p in files}}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    write_csv(out/'cache_manifest.csv', cache_manifest)
    (out/'provenance.json').write_text(json.dumps(dict(segmentation=provenance_seen,
        segmentation_runtime_variants=provenance_variants,
        allow_runtime_variation=allow_runtime_variation,overrides=overrides or {},
        gt_sha256=hashlib.sha256(Path(gt_path).read_bytes()).hexdigest(),
        source_sha256=source_hashes,
        anchors_sha256=None if anchors_path is None else hashlib.sha256(Path(anchors_path).read_bytes()).hexdigest(),
        config={k:getattr(cfg,k) for k in dir(cfg) if k.isupper()}),indent=2))
    for video in summary['per_video']:
        write_csv(out/f'{video}_per_cell.csv',[r for r in rows if r['video_id']==video])
        (out/f'{video}_metrics.json').write_text(json.dumps(summary['per_video'][video],indent=2))
    # Private local cache of diagnostics. Never load pickles from other sources.
    import pickle
    with (out/'cases.pkl').open('wb') as f: pickle.dump(cases,f)
    if gallery:
        from validation.scar_diagnostics import gallery as make_gallery
        make_gallery(cases,out/'gallery',cfg)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5-dir',default='validation/h5_for_drive/carmen_package_2')
    p.add_argument('--gt',default='validation/package_for_carmen_2/ground_truth_template_cr_v2.csv')
    p.add_argument('--cache',default='validation/runs/scar_v2/cache')
    p.add_argument('--out',default='validation/runs/scar_v2/baseline')
    p.add_argument('--overrides',default='{}',help='JSON object of configuration overrides')
    p.add_argument('--gallery',action='store_true')
    p.add_argument('--segment-missing',action='store_true')
    p.add_argument('--cache-wait-seconds',type=int,default=0)
    p.add_argument('--anchors',default=None)
    p.add_argument('--precision',choices=['bfloat16','float32'],default='bfloat16')
    p.add_argument('--allow-runtime-variation',action='store_true')
    p.add_argument('--expected-model-sha256',default=None)
    a=p.parse_args()
    print(json.dumps(run_batch(a.h5_dir,a.gt,a.cache,a.out,json.loads(a.overrides),a.gallery,a.segment_missing,cache_wait_seconds=a.cache_wait_seconds,anchors_path=a.anchors,segmentation_precision=a.precision,allow_runtime_variation=a.allow_runtime_variation,expected_model_sha256=a.expected_model_sha256),indent=2))


if __name__=='__main__': main()
