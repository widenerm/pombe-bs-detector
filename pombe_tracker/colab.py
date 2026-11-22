"""Notebook helpers: show a cell inline and place a birth scar by clicking on it (Google Colab).

    from pombe_tracker import colab
    colab.review('/content/drive/MyDrive/bsd_runs/my_movie')            # table of cells to check
    colab.show('/content/drive/MyDrive/bsd_runs/my_movie', 'A01')        # lengths over time + the cell
    colab.place_scars('/content/drive/MyDrive/bsd_runs/my_movie', 'A01', '52, 60')

Each click is appended to RUN/corrections/edits.csv (a later click on the same frame replaces an
earlier one when corrections are applied). Nothing is saved to disk except that CSV row.
"""
import base64
import io
import uuid

import numpy as np

from .corrections import add_edit, corrected_measurements, parse_frames
from .review import cell_view, draw_cell_view, review_table, show_cell
from .workflow import Run


def review(run_path, top=None):
    t = review_table(Run(run_path))
    return t if top is None else t.head(top)


def show(run_path, cell, frames=None, max_panels=8):
    import matplotlib.pyplot as plt
    fig = show_cell(Run(run_path), cell, frames, max_panels)
    plt.show()
    plt.close(fig)


def click_to_movie(extent, fx, fy):
    """Click position as a fraction of the displayed image (0-1, from the top-left) -> movie (x, y)."""
    x0, x1, y1, y0 = extent
    return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)


def _png(run, row, src, zoom):
    import matplotlib
    import matplotlib.pyplot as plt
    v = cell_view(run, row, src)
    h, w = v['crop'].shape
    fig = plt.figure(figsize=(w * zoom / 100, h * zoom / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_cell_view(ax, v, row, title=False)
    ax.set_aspect('auto')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), v['extent']


def place_scars(run_path, cell, frames, zoom=4, who=''):
    """Show the cell at each frame; click the birth-scar midpoint to record it as a correction."""
    try:
        from google.colab import output
    except ImportError as e:
        raise RuntimeError('click-to-place works in Google Colab. Elsewhere, read the position off '
                           '`bsd show` (axes are movie pixels) and use `bsd fix RUN --cell C --frames F --scar X Y`.') from e
    from IPython.display import HTML, display
    run = Run(run_path)
    d, _ = corrected_measurements(run, log=lambda *_: None)
    g = d[d.cell_name.astype(str) == str(cell)]
    if g.empty:
        raise ValueError(f'no cell {cell!r}')
    wanted = parse_frames(frames) if isinstance(frames, str) else list(frames)
    if wanted is None:
        raise ValueError('give the frames to correct, e.g. "52" or "50-55"')
    with run.frames() as src:
        for fr in wanted:
            rows = g[g.frame == fr]
            if rows.empty:
                display(HTML(f'<p>{cell} has no frame {fr}.</p>'))
                continue
            row = rows.iloc[0]
            b64, extent = _png(run, row, src, zoom)
            name = f'bsd_scar_{uuid.uuid4().hex[:10]}'

            def _record(fx, fy, fr=fr, extent=extent):
                x, y = click_to_movie(extent, float(fx), float(fy))
                add_edit(run, str(cell), str(fr), scar_x=round(x, 1), scar_y=round(y, 1), who=who,
                         note='scar placed in Colab')
                return f'{x:.1f}, {y:.1f}'

            output.register_callback(name, _record)
            ne, oe = row.get('new_end_length_um'), row.get('old_end_length_um')
            now = f'detector: new {ne:.2f} µm, old {oe:.2f} µm' if np.isfinite(ne) and np.isfinite(oe) else 'detector: no scar'
            display(HTML(f'''
<div style="font:13px sans-serif;margin:8px 0 2px">{cell}, frame {fr} (Fiji {fr + 1}). {now}.
 Click the birth-scar midpoint (white dot = detector's).</div>
<img src="data:image/png;base64,{b64}" style="cursor:crosshair;border:1px solid #888"
 onclick="(async function(e){{const r=e.target.getBoundingClientRect();
   const res=await google.colab.kernel.invokeFunction('{name}',[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height],{{}});
   const t=res.data['text/plain'];
   document.getElementById('{name}_msg').textContent='Saved scar at x, y = '+t.replace(/'/g,'')+
     '. Click again to replace it. Run the growth step to apply.';}})(event)">
<div id="{name}_msg" style="font:13px sans-serif;color:#2a6;margin-bottom:10px"></div>'''))
