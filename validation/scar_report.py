"""Build auditable metric tables and an exact failure list from completed runs."""
import csv
import json
from pathlib import Path
import numpy as np


def load_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def metric_table(metrics_by_name):
    lines=['| Run | n | Detected | Recall ≤5 px | Precision ≤5 px* | Median px | MAE px | p90 px |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def f(x,percent=False):
        if x is None:return 'N/A'
        return f'{x:.1%}' if percent else f'{x:.2f}'
    for name,m in metrics_by_name.items():
        lines.append(f"| {name} | {m['n']} | {m['detected']} | {f(m['recall_5px'],True)} | {f(m['precision_5px_annotated'],True)} | {f(m['median_error_px'])} | {f(m['mae_px'])} | {f(m['p90_error_px'])} |")
    return '\n'.join(lines)


def create_tables(baseline_dir,comparison_dir,out_dir):
    baseline_dir=Path(baseline_dir);comparison_dir=Path(comparison_dir)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((baseline_dir/'metrics.json').read_text())
    comparison=json.loads((comparison_dir/'metrics.json').read_text())
    text=['# Numerical validation results',metric_table({'legacy':baseline['combined'],'experimental':comparison['combined']}),
          '*Precision is localization precision conditional on annotated cells. Population detection precision is not estimable from this positive-only set. Localization recall and fraction within 5 pixels both divide by all annotated positives; missing predictions count as failures. Error quantiles and MAE use finite matched predictions. Detection recall is reported separately below.*']
    for name,summary in [('legacy',baseline),('experimental',comparison)]:
        text.extend([f'## {name}: per-video performance',metric_table(summary['per_video']),
                     f"Detection recall: {summary['combined']['detection_recall']:.1%}.",
                     f"Candidate oracle within 5 px: {summary['combined']['oracle_within_5']}/{summary['combined']['n']}.",
                     'Failure categories: '+json.dumps(summary['combined']['failures'])])
    for name,folder in [('legacy',baseline_dir),('experimental',comparison_dir)]:
        text.append(f'## Exact cells outside tolerance: {name}')
        failure_lines=['| Video | Original frame | Cell | Error px | Oracle px | Classification |',
                       '|---|---:|---|---:|---:|---|']
        for r in load_rows(folder/'outside_5px.csv'):
            error=f"{float(r['error_px']):.3f}" if r['error_px'] else 'missing'
            oracle=f"{float(r['oracle_error_px']):.3f}" if r['oracle_error_px'] else 'missing'
            failure_lines.append(f"| {r['video_id']} | {r['frame']} | {r['cell_name']} | {error} | {oracle} | {r['failure']} |")
        text.append("\n".join(failure_lines))
    (out/'METRICS.md').write_text('\n\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name,folder in [('legacy',baseline_dir),('experimental',comparison_dir)]:
        rows=load_rows(folder/'per_cell.csv');n=sum(r['gt_positive']=='True' for r in rows)
        errors=sorted(float(r['error_px']) for r in rows if r['error_px'])
        if errors:ax.step(errors,np.arange(1,len(errors)+1)/n,where='post',label=name)
    ax.axvline(5,color='black',ls=':');ax.axhline(.9,color='gray',ls=':')
    ax.set(xlabel='Localization error (pixels)',ylabel='Fraction of all annotated positives',ylim=(0,1.02))
    ax.legend();fig.tight_layout();fig.savefig(out/'localization_cdf.png',dpi=150);plt.close(fig)
    return out/'METRICS.md'


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline');p.add_argument('comparison');p.add_argument('out')
    a=p.parse_args();print(create_tables(a.baseline,a.comparison,a.out))
