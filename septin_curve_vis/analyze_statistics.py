"""Spatially aware statistics for septin intensity versus contour curvature.

The raw measurements contain repeated, autocorrelated positions around each
cell. This script therefore treats cells (not contour points) as independent
units and uses circular shifts within each cell as permutation controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binomtest, rankdata, wilcoxon


METRICS = {
    "signed_curvature": lambda k: k,
    "positive_curvature": lambda k: np.maximum(k, 0),
    "absolute_curvature": lambda k: np.abs(k),
}


def circular_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    return values if sigma == 0 else gaussian_filter1d(values, sigma, mode="wrap")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot=5000):
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1.0)
    return result


def circular_shift_pvalue(x: np.ndarray, y: np.ndarray, observed: float,
                          rng: np.random.Generator, n_perm: int) -> float:
    null = np.empty(n_perm)
    for i in range(n_perm):
        shift = rng.integers(1, len(y))
        null[i] = spearman(x, np.roll(y, shift))
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))


DIVIDING = {"dividing", "divide", "d", "1", "yes", "y", "true"}


def select_cells(summary: pd.DataFrame, exclude: Path | None = None, min_length_um: float | None = None,
                 max_length_um: float | None = None):
    """Cells kept by the optional filters, and a list of plain-language reasons for the report.

    exclude: CSV with a cell_id column. If it also has a `label` column (the division-labelling
    template), only rows labelled dividing are excluded; otherwise every listed cell is.
    """
    keep = pd.Series(True, index=summary.index)
    notes = []
    if exclude is not None:
        ex = pd.read_csv(exclude, dtype=str)
        if "cell_id" not in ex:
            raise ValueError(f"{exclude} has no cell_id column")
        if "label" in ex:
            ex = ex[ex.label.fillna("").str.strip().str.lower().isin(DIVIDING)]
        unknown = set(ex.cell_id) - set(summary.cell_id)
        hit = summary.cell_id.isin(set(ex.cell_id))
        keep &= ~hit
        notes.append(f"excluded {int(hit.sum())} cells listed in `{Path(exclude).name}`"
                     + (f" ({len(unknown)} listed IDs are not in this dataset)" if unknown else ""))
    for bound, op, word in ((min_length_um, "lt", "shorter"), (max_length_um, "gt", "longer")):
        if bound is None:
            continue
        if "length_um" not in summary:
            raise ValueError("cell_summary.csv has no length_um column; rerun `bsd septin measure` to add it")
        hit = getattr(summary.length_um, op)(bound) & keep
        keep &= ~hit
        notes.append(f"excluded {int(hit.sum())} cells {word} than {bound:g} um")
    return summary[keep], notes


def analyze(measurements: pd.DataFrame, summary: pd.DataFrame, out: Path, qc_threshold=0.10,
            primary_sigma=3.0, permutations=1000, title="Septin localization versus curvature",
            notes=(), seed=20260910, fig_dir="out"):
    """Every statistic, table, figure and statistics_report.md for the cells in `summary`, written to `out`.

    Returns the association summary table.
    """
    args = argparse.Namespace(qc_threshold=qc_threshold, primary_sigma=primary_sigma, permutations=permutations)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    clean_ids = set(summary.loc[summary.max_abs_curvature <= args.qc_threshold, "cell_id"])
    clean = measurements[measurements.cell_id.isin(clean_ids)].copy()

    # Smoothing sensitivity: mean per-cell Spearman correlation.
    sigmas = [0, 1, 2, 3, 4, 6, 8]
    sensitivity = []
    for sigma in sigmas:
        for metric_name, transform in METRICS.items():
            rhos = []
            for _, cell in clean.groupby("cell_id", sort=False):
                cell = cell.sort_values("contour_index")
                k = circular_smooth(cell.curvature.to_numpy(), sigma)
                f = circular_smooth(cell.fluorescence_intensity.to_numpy(), sigma)
                rhos.append(spearman(transform(k), f))
            lo, hi = bootstrap_mean(np.asarray(rhos), rng)
            sensitivity.append({"sigma_samples": sigma, "metric": metric_name,
                                "mean_spearman": np.mean(rhos),
                                "ci_low": lo, "ci_high": hi,
                                "median_spearman": np.median(rhos)})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out / "smoothing_sensitivity.csv", index=False)

    # Independent smoothing grid for the primary biological hypothesis. This
    # separates the effect of denoising fluorescence from contour smoothing.
    independent = []
    for curvature_sigma in [0, 1, 2, 3, 4, 6]:
        for fluorescence_sigma in [0, 1, 2, 3, 4, 6]:
            rhos = []
            for _, cell in clean.groupby("cell_id", sort=False):
                cell = cell.sort_values("contour_index")
                k = circular_smooth(cell.curvature.to_numpy(), curvature_sigma)
                f = circular_smooth(cell.fluorescence_intensity.to_numpy(), fluorescence_sigma)
                rhos.append(spearman(np.maximum(k, 0), f))
            independent.append({"curvature_sigma": curvature_sigma,
                                "fluorescence_sigma": fluorescence_sigma,
                                "mean_positive_curvature_spearman": np.mean(rhos)})
    independent = pd.DataFrame(independent)
    independent.to_csv(out / "independent_smoothing_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    for metric, table in sensitivity.groupby("metric"):
        ax.plot(table.sigma_samples, table.mean_spearman, marker="o", label=metric.replace("_", " "))
        ax.fill_between(table.sigma_samples, table.ci_low, table.ci_high, alpha=0.15)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(xlabel="Circular Gaussian smoothing (contour samples)",
           ylabel="Mean within-cell Spearman correlation",
           title="Smoothing sensitivity after curvature QC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig_dir = out if fig_dir == "out" else (Path(fig_dir) if fig_dir else None)
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "smoothing_sensitivity.png")
    plt.close(fig)

    grid = independent.pivot(index="curvature_sigma", columns="fluorescence_sigma",
                             values="mean_positive_curvature_spearman")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.12, vmax=0.12)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set(xlabel="Fluorescence smoothing (samples)",
           ylabel="Curvature smoothing (samples)",
           title="Positive-curvature association vs smoothing")
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            ax.text(j, i, f"{grid.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean within-cell Spearman correlation")
    fig.tight_layout()
    if fig_dir is not None:
        fig.savefig(fig_dir / "independent_smoothing_sensitivity.png")
    plt.close(fig)

    # Primary analysis uses the selected smoothing width. Test each cell with
    # a circular-shift null; then aggregate cell effects without point-level
    # pseudoreplication.
    rows = []
    standardized = {name: [] for name in METRICS}
    intensity_ranks = []
    for cell_id, cell in clean.groupby("cell_id", sort=False):
        cell = cell.sort_values("contour_index")
        k = circular_smooth(cell.curvature.to_numpy(), args.primary_sigma)
        f = circular_smooth(cell.fluorescence_intensity.to_numpy(), args.primary_sigma)
        f_rank = rankdata(f)
        f_rank = (f_rank - f_rank.mean()) / (f_rank.std() + 1e-12)
        intensity_ranks.append(f_rank)
        for metric_name, transform in METRICS.items():
            x = transform(k)
            rho = spearman(x, f)
            p_shift = circular_shift_pvalue(x, f, rho, rng, args.permutations)
            x_rank = rankdata(x)
            x_rank = (x_rank - x_rank.mean()) / (x_rank.std() + 1e-12)
            standardized[metric_name].append(x_rank)
            rows.append({"cell_id": cell_id, "metric": metric_name, "spearman_rho": rho,
                         "circular_shift_p": p_shift, "max_abs_curvature": np.max(np.abs(k))})
    cell_stats = pd.DataFrame(rows)
    cell_stats["fdr_q"] = cell_stats.groupby("metric")["circular_shift_p"].transform(
        lambda x: bh_fdr(x.to_numpy()))
    cell_stats.to_csv(out / "cell_level_associations.csv", index=False)

    # Pooled effect permutation: each permutation independently rotates each
    # cell's intensity rank vector, preserving profile shape and autocorrelation.
    aggregate_rows = []
    for metric_name in METRICS:
        x_vectors = standardized[metric_name]
        observed = float(np.mean([np.mean(x * y) for x, y in zip(x_vectors, intensity_ranks)]))
        null = np.empty(args.permutations)
        for b in range(args.permutations):
            null[b] = np.mean([np.mean(x * np.roll(y, rng.integers(1, len(y))))
                               for x, y in zip(x_vectors, intensity_ranks)])
        p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (args.permutations + 1))
        effects = cell_stats.loc[cell_stats.metric == metric_name, "spearman_rho"].to_numpy()
        ci_low, ci_high = bootstrap_mean(effects, rng)
        w = wilcoxon(effects, zero_method="wilcox", alternative="two-sided", method="auto")
        aggregate_rows.append({
            "metric": metric_name,
            "n_clean_cells": len(effects),
            "mean_within_cell_spearman": effects.mean(),
            "median_within_cell_spearman": np.median(effects),
            "mean_ci_low": ci_low,
            "mean_ci_high": ci_high,
            "circular_shift_permutation_p": p,
            "wilcoxon_p": float(w.pvalue),
            "cells_fdr_significant": int((cell_stats.loc[cell_stats.metric == metric_name, "fdr_q"] < 0.05).sum()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(out / "association_summary.csv", index=False)

    # Enrichment of fluorescence at a cell's strongest positive-curvature sites.
    enrichment = []
    for cell_id, cell in clean.groupby("cell_id", sort=False):
        cell = cell.sort_values("contour_index")
        k = circular_smooth(cell.curvature.to_numpy(), args.primary_sigma)
        f = circular_smooth(cell.fluorescence_intensity.to_numpy(), args.primary_sigma)
        high = f[k >= np.quantile(k, 0.90)]
        mid = f[(k >= np.quantile(k, 0.40)) & (k <= np.quantile(k, 0.60))]
        enrichment.append(float(np.mean(high) - np.mean(mid)))
    enrichment = np.asarray(enrichment)
    enrichment_test = wilcoxon(enrichment, zero_method="wilcox", alternative="two-sided", method="auto")
    sign = binomtest(int(np.sum(enrichment > 0)), len(enrichment), 0.5)
    enrich_lo, enrich_hi = bootstrap_mean(enrichment, rng)

    # Document the analysis in a compact, readable report.
    primary = sensitivity[(sensitivity.sigma_samples == args.primary_sigma) &
                          (sensitivity.metric == "positive_curvature")].iloc[0]
    lines = [
        f"# {title}",
        "",
        *([f"Cell selection: {'; '.join(notes)}.", ""] if notes else []),
        f"Primary dataset: {len(clean_ids)} QC-passing cells of {len(summary)} total. "
        f"Excluded {len(summary) - len(clean_ids)} cells with max |curvature| > {args.qc_threshold:.2f}.",
        "",
        f"Primary smoothing: circular Gaussian sigma = {args.primary_sigma:g} contour samples "
        f"({args.primary_sigma / 300:.1%} of the resampled perimeter).",
        "",
        "## Spatially aware association tests",
        "",
        "```csv",
        aggregate.round(5).to_csv(index=False).strip(),
        "```",
        "",
        "Pooled permutation p-values independently circularly shift fluorescence around each cell, "
        "preserving within-cell spatial autocorrelation and avoiding point-level pseudoreplication.",
        "",
        "## Positive-curvature enrichment",
        "",
        f"Mean fluorescence difference (top 10% positive-curvature locations minus middle 20% curvature locations): "
        f"{enrichment.mean():.2f} a.u. (95% bootstrap CI {enrich_lo:.2f} to {enrich_hi:.2f}); "
        f"Wilcoxon p={enrichment_test.pvalue:.4g}, sign-test p={sign.pvalue:.4g}; "
        f"{np.sum(enrichment > 0)}/{len(enrichment)} cells positive.",
        "",
        "## Smoothing interpretation",
        "",
        f"At the primary width, the mean positive-curvature Spearman association is "
        f"{primary.mean_spearman:.4f} (95% bootstrap CI {primary.ci_low:.4f} to {primary.ci_high:.4f}). "
        "See `smoothing_sensitivity.csv`, `independent_smoothing_sensitivity.csv` and the corresponding PNGs "
        "for how the association changes when curvature and fluorescence are smoothed, together or separately.",
        "",
        "Interpret signed curvature as directional convexity. Positive-only curvature tests the stated "
        "hypothesis that septins enrich at convex features; absolute curvature tests a non-directional relationship.",
        "",
        "## Open question",
        "",
        "Fluorescence is sampled in a narrow band centred on the Cellpose outline. Brightfield outlines sit about "
        "2-4 px outside the membrane, so the band may lie partly outside the cortex. This affects absolute "
        "enrichment values more than within-cell rank correlations; it has not yet been tested.",
    ]
    (out / "statistics_report.md").write_text("\n".join(lines) + "\n")
    return aggregate


def run(results: Path, exclude=None, min_length_um=None, max_length_um=None, qc_threshold=0.10,
        primary_sigma=3.0, permutations=1000, log=print, figures_dir="out"):
    """Statistics for all cells in results/statistics/; with any filter, also for the kept cells in
    results/statistics_filtered/, plus results/statistics_comparison.md side by side."""
    measurements = pd.read_csv(results / "measurements.csv")
    summary = pd.read_csv(results / "cell_summary.csv")
    kw = dict(qc_threshold=qc_threshold, primary_sigma=primary_sigma, permutations=permutations)
    fd = lambda name: "out" if figures_dir == "out" else (Path(figures_dir) / name if figures_dir else None)
    everyone = analyze(measurements, summary, results / "statistics", fig_dir=fd("statistics"), **kw)
    log((results / "statistics" / "statistics_report.md").read_text().split("## Positive")[0].strip())
    if exclude is None and min_length_um is None and max_length_um is None:
        return everyone, None
    kept, notes = select_cells(summary, exclude, min_length_um, max_length_um)
    filtered = analyze(measurements, kept, results / "statistics_filtered", fig_dir=fd("statistics_filtered"),
                       title="Septin localization versus curvature (filtered cells)", notes=notes, **kw)
    cols = ["metric", "n_clean_cells", "mean_within_cell_spearman", "mean_ci_low", "mean_ci_high",
            "circular_shift_permutation_p"]
    both = pd.concat([everyone[cols].assign(cells="all"), filtered[cols].assign(cells="filtered")])
    both = both[["cells"] + cols].sort_values(["metric", "cells"]).round(5)
    text = ["# All cells vs filtered cells", "", f"Filters: {'; '.join(notes)}.", "",
            "```csv", both.to_csv(index=False).strip(), "```", "",
            "Full reports: `statistics/statistics_report.md` (all cells) and "
            "`statistics_filtered/statistics_report.md` (filtered)."]
    (results / "statistics_comparison.md").write_text("\n".join(text) + "\n")
    log("\n".join(text))
    return everyone, filtered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="folder with measurements.csv and cell_summary.csv")
    parser.add_argument("--qc-threshold", type=float, default=0.10)
    parser.add_argument("--primary-sigma", type=float, default=3.0,
                        help="Circular Gaussian smoothing in contour samples")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--exclude", type=Path, help="CSV of cell_id to leave out (e.g. dividing-cell labels)")
    parser.add_argument("--min-length-um", type=float)
    parser.add_argument("--max-length-um", type=float)
    args = parser.parse_args()
    run(args.results, args.exclude, args.min_length_um, args.max_length_um, args.qc_threshold,
        args.primary_sigma, args.permutations)


if __name__ == "__main__":
    main()
