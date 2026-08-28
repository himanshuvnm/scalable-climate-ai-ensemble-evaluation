import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Make heatmap of query-conditioned historical GEFS weights."
    )
    p.add_argument("--meta", required=True)
    p.add_argument("--descriptor", default="")
    p.add_argument("--ell", default="auto",
                   help="'auto' or numeric. Auto uses ell_factor * median distance for each query.")
    p.add_argument("--ell-factor", type=float, default=0.25)
    p.add_argument("--outdir", required=True)
    p.add_argument("--title-tag", default="HGT500 query-conditioned historical weights")
    p.add_argument("--exclude-neighbor", type=int, default=0,
                   help="Exclude candidates with |m-q| <= this value. Default 0 excludes only the query itself.")
    return p.parse_args()


def short_tag(tag):
    if "_pgrb2a" in tag:
        tag = tag.split("_pgrb2a")[0]
    return tag


def load_metadata(meta_path):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    return meta_path, meta


def load_descriptors(meta, descriptor_override):
    if descriptor_override:
        dpath = Path(descriptor_override)
    else:
        if "descriptor_norm_path" not in meta:
            raise KeyError("metadata missing descriptor_norm_path")
        dpath = Path(meta["descriptor_norm_path"])

    if not dpath.exists():
        raise FileNotFoundError(f"Descriptor file not found: {dpath}")

    D = np.load(dpath)
    return dpath, D


def effective_sample_size(w):
    return float(1.0 / np.sum(w ** 2))


def compute_weight_matrix(D, ell_arg, ell_factor, exclude_neighbor):
    M = D.shape[0]
    W = np.full((M, M), np.nan, dtype=float)
    Dist = np.full((M, M), np.nan, dtype=float)

    summary = []

    for q in range(M):
        psi_q = D[q]
        dist_all = np.linalg.norm(D - psi_q[None, :], axis=1)

        mask = np.ones(M, dtype=bool)
        # Exclude the query itself, and optionally nearby chronological neighbors.
        mask[np.abs(np.arange(M) - q) <= exclude_neighbor] = False

        cand = np.where(mask)[0]
        dist = dist_all[cand]

        if ell_arg == "auto":
            ell = float(ell_factor * np.median(dist))
        else:
            ell = float(ell_arg)

        if ell <= 0:
            raise ValueError("ell must be positive")

        score = -dist / ell
        score = score - score.max()
        raw = np.exp(score)
        w = raw / raw.sum()

        W[q, cand] = w
        Dist[q, cand] = dist

        ws = np.sort(w)[::-1]
        summary.append({
            "query_index": q,
            "ell": ell,
            "num_candidates": int(len(cand)),
            "static_weight": float(1.0 / len(cand)),
            "effective_sample_size": effective_sample_size(w),
            "max_weight": float(w.max()),
            "top5_total_weight": float(ws[:5].sum()),
            "top10_total_weight": float(ws[:10].sum()),
            "top20_total_weight": float(ws[:20].sum()),
        })

    return W, Dist, summary


def make_ticks(tags, step=8):
    M = len(tags)
    idx = list(range(0, M, step))
    labels = [short_tag(tags[i]).replace("2026", "") for i in idx]
    return idx, labels


def plot_weight_heatmap(W, tags, out_png, out_pdf, title_tag, ell_factor, exclude_neighbor):
    M = W.shape[0]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    # Square matrix-style heatmap with isoweight contours.
    fig, ax = plt.subplots(figsize=(10.8, 10.5), constrained_layout=True)

    im = ax.imshow(
        W,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_box_aspect(1)

    # Add 5--7 isoweight contours on the unmasked valid weights.
    valid = W[np.isfinite(W)]
    if valid.size > 0:
        # Quantile levels emphasize elevated-weight structure without overcrowding.
        levels = np.quantile(valid, [0.65, 0.75, 0.85, 0.92, 0.96, 0.98])
        levels = np.unique(levels)

        Xg, Yg = np.meshgrid(np.arange(M), np.arange(M))
        W_masked = np.ma.masked_invalid(W)

        if len(levels) >= 2:
            ax.contour(
                Xg,
                Yg,
                W_masked,
                levels=levels,
                colors="white",
                linewidths=0.65,
                alpha=0.65,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Query-conditioned weight $w_m(q)$")

    xticks, xlabels = make_ticks(tags, step=8)
    yticks, ylabels = make_ticks(tags, step=8)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)

    ax.set_xlabel("Historical GEFS case index $m$")
    ax.set_ylabel("Query GEFS case index $q$")

    if exclude_neighbor == 0:
        excl = "query case itself excluded"
    else:
        excl = f"cases with |m-q| <= {exclude_neighbor} excluded"

    ax.set_title(
        f"{title_tag}\n"
        f"Exponential-distance weights with isoweight contours; {excl}; ell-factor = {ell_factor}",
        fontsize=13,
        pad=12,
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary, tags, out_png, out_pdf, title_tag):
    q = np.array([s["query_index"] for s in summary])
    neff = np.array([s["effective_sample_size"] for s in summary])
    maxw = np.array([s["max_weight"] for s in summary])
    top10 = np.array([s["top10_total_weight"] for s in summary])

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.0), sharex=True)

    axes[0].plot(q, neff, linewidth=2)
    axes[0].set_ylabel(r"$N_{\mathrm{eff}}$")
    axes[0].set_title("(a) Effective number of historical cases used")
    axes[0].grid(alpha=0.3)

    axes[1].plot(q, maxw, linewidth=2)
    axes[1].set_ylabel("Max weight")
    axes[1].set_title("(b) Largest single historical-case weight")
    axes[1].grid(alpha=0.3)

    axes[2].plot(q, top10, linewidth=2)
    axes[2].set_ylabel("Top-10 weight")
    axes[2].set_xlabel("Query GEFS case index")
    axes[2].set_title("(c) Total weight carried by top 10 historical cases")
    axes[2].grid(alpha=0.3)

    xticks, xlabels = make_ticks(tags, step=8)
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(xlabels, rotation=90, fontsize=8)

    fig.suptitle(
        f"{title_tag}\nSummary of query-conditioned historical weighting",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(summary, tags, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index",
            "query_tag",
            "ell",
            "num_candidates",
            "static_weight",
            "effective_sample_size",
            "max_weight",
            "top5_total_weight",
            "top10_total_weight",
            "top20_total_weight",
        ])
        for s in summary:
            q = s["query_index"]
            writer.writerow([
                q,
                tags[q],
                s["ell"],
                s["num_candidates"],
                s["static_weight"],
                s["effective_sample_size"],
                s["max_weight"],
                s["top5_total_weight"],
                s["top10_total_weight"],
                s["top20_total_weight"],
            ])


if __name__ == "__main__":
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path, meta = load_metadata(args.meta)
    desc_path, D = load_descriptors(meta, args.descriptor)

    tags = meta["tags"]
    if len(tags) != D.shape[0]:
        raise ValueError(f"number of tags {len(tags)} does not match descriptor rows {D.shape[0]}")

    W, Dist, summary = compute_weight_matrix(
        D,
        ell_arg=args.ell,
        ell_factor=args.ell_factor,
        exclude_neighbor=args.exclude_neighbor,
    )

    ell_tag = f"ellfac{str(args.ell_factor).replace('.', 'p')}_excl{args.exclude_neighbor}"

    np.save(outdir / f"gefs_query_conditioned_weight_matrix_{ell_tag}.npy", W)
    np.save(outdir / f"gefs_query_conditioned_distance_matrix_{ell_tag}.npy", Dist)

    save_summary_csv(
        summary,
        tags,
        outdir / f"gefs_query_conditioned_weight_summary_{ell_tag}.csv",
    )

    plot_weight_heatmap(
        W,
        tags,
        outdir / f"gefs_query_conditioned_weight_heatmap_{ell_tag}.png",
        outdir / f"gefs_query_conditioned_weight_heatmap_{ell_tag}.pdf",
        args.title_tag,
        args.ell_factor,
        args.exclude_neighbor,
    )

    plot_summary(
        summary,
        tags,
        outdir / f"gefs_query_conditioned_weight_summary_{ell_tag}.png",
        outdir / f"gefs_query_conditioned_weight_summary_{ell_tag}.pdf",
        args.title_tag,
    )

    print("saved outputs in:", outdir)
    print("weight heatmap:", outdir / f"gefs_query_conditioned_weight_heatmap_{ell_tag}.png")
    print("summary figure:", outdir / f"gefs_query_conditioned_weight_summary_{ell_tag}.png")
    print("summary csv:", outdir / f"gefs_query_conditioned_weight_summary_{ell_tag}.csv")
