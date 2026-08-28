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
        description="Leave-one-out DA-facing figure for observation-conditioned historical weights."
    )
    p.add_argument("--meta", required=True)
    p.add_argument("--descriptor", default="")
    p.add_argument("--query-index", type=int, required=True)
    p.add_argument("--ell", default="auto",
                   help="'auto' or numeric. Auto uses ell_factor * median non-self distance.")
    p.add_argument("--ell-factor", type=float, default=0.5,
                   help="Used only when --ell auto.")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--outdir", required=True)
    p.add_argument("--title-tag", default="HGT500 observation-conditioned historical weights")
    return p.parse_args()


def short_tag(tag):
    return tag.split("_pgrb2a")[0] if "_pgrb2a" in tag else tag


def effective_sample_size(w):
    return float(1.0 / np.sum(w ** 2))


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
        raise FileNotFoundError(dpath)

    D = np.load(dpath)
    return dpath, D


def compute_leave_one_out_weights(D, q, ell_arg, ell_factor):
    psi_q = D[q]
    dist_all = np.linalg.norm(D - psi_q[None, :], axis=1)

    mask = np.ones(D.shape[0], dtype=bool)
    mask[q] = False

    candidate_idx = np.where(mask)[0]
    dist = dist_all[candidate_idx]

    if ell_arg == "auto":
        ell = float(ell_factor * np.median(dist))
    else:
        ell = float(ell_arg)

    if ell <= 0:
        raise ValueError("ell must be positive")

    # stable softmax-style computation
    score = -dist / ell
    score = score - score.max()
    raw = np.exp(score)
    w = raw / raw.sum()

    return candidate_idx, dist, w, ell, dist_all


def save_topk_csv(path, top_candidate_idx, tags, weights, distances):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "archive_index", "tag", "weight", "distance"])
        for r, k in enumerate(top_candidate_idx, start=1):
            writer.writerow([
                r,
                int(k),
                tags[k],
                float(weights[r-1]),
                float(distances[r-1]),
            ])


def make_figure(path_png, path_pdf, title_tag, tags, q, cand_idx, dist, w, ell, topk):
    M_hist = len(w)
    uniform = 1.0 / M_hist

    order_local = np.argsort(w)[::-1]
    cand_sorted = cand_idx[order_local]
    w_sorted = w[order_local]
    dist_sorted = dist[order_local]

    topk = min(topk, M_hist)
    top_cand = cand_sorted[:topk]
    top_w = w_sorted[:topk]
    top_dist = dist_sorted[:topk]

    neff = effective_sample_size(w)
    cumw = np.cumsum(w_sorted)

    fig = plt.figure(figsize=(15.5, 8.5))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(np.arange(1, M_hist + 1), w_sorted, linewidth=2,
             label="Conditioned historical weights")
    ax1.axhline(uniform, linestyle="--", linewidth=1.5,
                label="Static historical weight")
    ax1.set_xlabel("Historical case rank, sorted by weight")
    ax1.set_ylabel("Weight")
    ax1.set_title("(a) Ranked leave-one-out historical weights")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    y = np.arange(topk)
    labels = [short_tag(tags[i]) for i in top_cand[::-1]]
    ax2.barh(y, top_w[::-1])
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Weight")
    ax2.set_title(f"(b) Top {topk} weighted historical cases")
    ax2.grid(axis="x", alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(np.arange(1, M_hist + 1), cumw, linewidth=2)
    for yline in [0.5, 0.8, 0.9]:
        ax3.axhline(yline, linestyle="--", linewidth=1)
    ax3.set_xlabel("Top-ranked historical cases included")
    ax3.set_ylabel("Cumulative weight")
    ax3.set_ylim(0, 1.02)
    ax3.set_title("(c) Weight concentration")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(dist, w, s=22, alpha=0.85)
    ax4.set_xlabel(r"Descriptor distance $\|\psi_q-\psi_m\|_2$, $m\neq q$")
    ax4.set_ylabel("Weight")
    ax4.set_title("(d) Descriptor distance versus weight")
    ax4.grid(alpha=0.3)

    txt = (
        f"Query tag: {short_tag(tags[q])}\n"
        f"Historical candidates = {M_hist}\n"
        f"Static weight = {uniform:.4f}\n"
        f"Effective sample size = {neff:.2f}\n"
        f"Max weight = {w.max():.4f}\n"
        f"Top-5 total weight = {w_sorted[:5].sum():.3f}\n"
        f"Top-10 total weight = {w_sorted[:10].sum():.3f}\n"
        f"Length scale ell = {ell:.3f}"
    )
    ax4.text(
        0.98, 0.98, txt,
        transform=ax4.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.13)
    )

    fig.suptitle(
        f"{title_tag}\nLeave-one-out query-conditioned weighting for covariance-response construction",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path_png, dpi=250, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)

    return top_cand, top_w, top_dist, neff, w_sorted


if __name__ == "__main__":
    args = parse_args()

    meta_path, meta = load_metadata(args.meta)
    desc_path, D = load_descriptors(meta, args.descriptor)

    tags = meta["tags"]
    if D.shape[0] != len(tags):
        raise ValueError(f"descriptor rows {D.shape[0]} != number of tags {len(tags)}")

    q = args.query_index
    if not (0 <= q < len(tags)):
        raise ValueError(f"query index out of range: {q}")

    cand_idx, dist, w, ell, dist_all = compute_leave_one_out_weights(
        D, q, args.ell, args.ell_factor
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ell_str = f"{ell:.3f}".replace(".", "p")
    stem = f"query{q:03d}_{short_tag(tags[q])}_loo_ell{ell_str}"

    png = outdir / f"{stem}_historical_weights.png"
    pdf = outdir / f"{stem}_historical_weights.pdf"
    csv_path = outdir / f"{stem}_top_cases.csv"
    json_path = outdir / f"{stem}_summary.json"

    top_cand, top_w, top_dist, neff, w_sorted = make_figure(
        png, pdf, args.title_tag, tags, q, cand_idx, dist, w, ell, args.topk
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "archive_index", "tag", "weight", "distance"])
        for r, idx in enumerate(top_cand, start=1):
            loc = np.where(cand_idx == idx)[0][0]
            writer.writerow([r, int(idx), tags[idx], float(w[loc]), float(dist[loc])])

    summary = {
        "query_index": int(q),
        "query_tag": tags[q],
        "metadata_path": str(meta_path),
        "descriptor_path": str(desc_path),
        "leave_one_out": True,
        "historical_candidates": int(len(w)),
        "ell": float(ell),
        "uniform_static_weight": float(1.0 / len(w)),
        "effective_sample_size": float(neff),
        "max_weight": float(w.max()),
        "top5_total_weight": float(w_sorted[:5].sum()),
        "top10_total_weight": float(w_sorted[:10].sum()),
        "top20_total_weight": float(w_sorted[:20].sum()),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print("saved:", png)
    print("saved:", pdf)
    print("saved:", csv_path)
    print("saved:", json_path)
    print(json.dumps(summary, indent=2))
