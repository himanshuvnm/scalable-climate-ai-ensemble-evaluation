#!/usr/bin/env python3
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
sys.path.append(str(PROJ / "code"))

from p4_gefs_obs_conditioned_localized_retrieval import (
    VAR_NAMES,
    NVARS,
    NLAT,
    NLON,
    BLOCK,
    cov_action,
    make_probe,
    local_descriptors_for_probe,
    laplacian_weights,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metadata_path = PROJ / "data_processed/p2_gefs_archive/archive_metadata_20260524_20260624_f006_M128.json"
csv_path = PROJ / "results/p4_obs_conditioned_localized_results_M128_Q128_P8_locsig15_patch4.csv"
out_dir = PROJ / "figures/final_presentation"
out_dir.mkdir(parents=True, exist_ok=True)

print("Using metadata:", metadata_path)
print("Using CSV:", csv_path)
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

with open(metadata_path, "r") as f:
    meta = json.load(f)

tags = meta["tags"]
x_paths = [Path(p) for p in meta["X_paths"]]

if "mean_paths" in meta:
    mean_paths = [Path(p) for p in meta["mean_paths"]]
else:
    mean_paths = [
        PROJ / "data_processed/p2_gefs_archive/mean_states" / f"mean_{tag}.npy"
        for tag in tags
    ]

Z = np.load(meta["descriptor_norm_path"]).astype(np.float32)
M = len(tags)

df = pd.read_csv(csv_path)

best = df[
    (df["method"] == "local_obs_laplacian") &
    (df["topk"] == 32) &
    (np.isclose(df["ell_mult"], 0.5)) &
    (np.isclose(df["beta"], 0.25)) &
    (np.isclose(df["lambda_shrink"], 1.0))
].copy()

static = df[df["method"] == "static_climatological"].copy()

keys = ["query", "probe_id", "source_var", "source_var_idx", "lat_idx", "lon_idx"]

b = best[keys + ["localized_rel_error"]].rename(
    columns={"localized_rel_error": "best_local"}
)
s = static[keys + ["localized_rel_error"]].rename(
    columns={"localized_rel_error": "static_local"}
)

m = pd.merge(b, s, on=keys, how="inner")
m["gain"] = m["static_local"] - m["best_local"]

# Prefer HGT500 for an intuitive atmospheric map. If not available, use best overall.
mh = m[m["source_var"] == "HGT500"].copy()
if len(mh) > 0:
    # Select example: default is max gain; optional environment override.
    import os
    case_query_env = os.environ.get("CASE_QUERY")
    case_probe_env = os.environ.get("CASE_PROBE")
    case_var_env = os.environ.get("CASE_VAR", "HGT500")

    if case_query_env is not None and case_probe_env is not None:
        case_query = int(case_query_env)
        case_probe = int(case_probe_env)
        sel = m[
            (m["query"] == case_query) &
            (m["probe_id"] == case_probe) &
            (m["source_var"] == case_var_env)
        ]
        if len(sel) == 0:
            raise RuntimeError(
                f"No row found for query={case_query}, probe_id={case_probe}, source_var={case_var_env}"
            )
        row = sel.iloc[0]
    else:
        row = m.sort_values("gain", ascending=False).iloc[0]
else:
    row = m.sort_values("gain", ascending=False).iloc[0]

q = int(row["query"])
source_var_idx = int(row["source_var_idx"])
source_var = str(row["source_var"])
lat_idx = int(row["lat_idx"])
lon_idx = int(row["lon_idx"])

print("Selected example:")
print(row)
print("Query tag:", tags[q])

# Load X and mean factors.
print("Loading ensemble anomaly factors...")
Xs = []
for i, p in enumerate(x_paths):
    print(f"  X {i+1:03d}/{M}: {p.name}", flush=True)
    Xs.append(torch.from_numpy(np.load(p).astype(np.float32)).to(device))

Means = []
have_means = all(p.exists() for p in mean_paths)
if have_means:
    print("Loading mean states...")
    for i, p in enumerate(mean_paths):
        print(f"  mean {i+1:03d}/{M}: {p.name}", flush=True)
        Means.append(torch.from_numpy(np.load(p).astype(np.float32)).to(device))
else:
    Means = None

K = Xs[0].shape[0]
v = make_probe(K, source_var_idx, lat_idx, lon_idx, device)

candidate_idx = np.array([i for i in range(M) if i != q], dtype=int)

with torch.no_grad():
    print("Computing oracle action...")
    y_oracle = cov_action(Xs[q], v)

    print("Computing candidate actions...")
    actions = []
    for idx in candidate_idx:
        actions.append(cov_action(Xs[int(idx)], v))
    all_actions = torch.stack(actions, dim=0)

    print("Computing static action...")
    y_static = all_actions.mean(dim=0)

    print("Computing observation-conditioned Laplacian action...")
    Dg = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2).astype(np.float64)
    offdiag = Dg[~np.eye(M, dtype=bool)]
    base_ell_global = np.median(offdiag)
    global_d_candidate = Dg[q, candidate_idx]
    g_unit = global_d_candidate / (base_ell_global + 1e-12)

    L = local_descriptors_for_probe(
        Xs, Means, lat_idx, lon_idx, radius=4, device=device
    )
    Lq = L[q]
    Lcand = L[candidate_idx]
    local_d = np.linalg.norm(Lcand - Lq[None, :], axis=1)

    beta = 0.25
    topk = 32
    ell_mult = 0.5

    combined_d = np.sqrt(g_unit**2 + beta * local_d**2)
    order = np.argsort(combined_d)
    idx_order = order[:topk]
    d_top = combined_d[idx_order]

    base_ell_local = np.median(d_top)
    if base_ell_local <= 0 or not np.isfinite(base_ell_local):
        base_ell_local = 1.0

    w = laplacian_weights(d_top, base_ell_local * ell_mult).astype(np.float32)
    w_t = torch.from_numpy(w).to(device)

    y_local = torch.matmul(w_t, all_actions[idx_order])

# Extract same-variable response maps.
a = source_var_idx * BLOCK
bidx = (source_var_idx + 1) * BLOCK

oracle_map = y_oracle[a:bidx].detach().cpu().numpy().reshape(NLAT, NLON)
static_map = y_static[a:bidx].detach().cpu().numpy().reshape(NLAT, NLON)
local_map = y_local[a:bidx].detach().cpu().numpy().reshape(NLAT, NLON)

static_err = static_map - oracle_map
local_err = local_map - oracle_map
abs_improve = np.abs(static_err) - np.abs(local_err)
# Save full response maps for Kalnay-style observation-centered diagnostic.
bundle_dir = PROJ / "results/obs_centered_cov_response_bundles"
bundle_dir.mkdir(parents=True, exist_ok=True)

bundle_path = bundle_dir / (
    f"gefs_{str(source_var).lower()}_q{int(row['query']):03d}_"
    f"probe{int(row['probe_id'])}_flow_conditioned_response_bundle.npz"
)

# The selected probe is stored by grid index in the P4 table.
# GEFS public grid here is 0.5-degree: lat index 0 = 90N, lon index 0 = 0E.
obs_lat_save = 90.0 - 0.5 * float(row["lat_idx"])
obs_lon_save = 0.5 * float(row["lon_idx"])

np.savez_compressed(
    bundle_path,
    oracle_response=oracle_map.astype(np.float32),
    static_response=static_map.astype(np.float32),
    obs_conditioned_laplacian_response=local_map.astype(np.float32),
    static_error=static_err.astype(np.float32),
    local_error=local_err.astype(np.float32),
    improvement=abs_improve.astype(np.float32),
    obs_lat=np.float32(obs_lat_save),
    obs_lon=np.float32(obs_lon_save),
    local_gain=np.float32(row["gain"]),
    query_label=str(tags[q]),
    var_name=str(source_var),
)

print("SAVED RESPONSE BUNDLE:", bundle_path)
print(f"SAVED OBS LOCATION: lat={obs_lat_save:.3f}, lon={obs_lon_save:.3f}")


# Local lat/lon crop around observation.
lat_half = 55
lon_half = 80

lat0 = max(0, lat_idx - lat_half)
lat1 = min(NLAT, lat_idx + lat_half + 1)
lon_ids = (np.arange(lon_idx - lon_half, lon_idx + lon_half + 1) % NLON).astype(int)

def crop(A):
    return A[lat0:lat1, :][:, lon_ids]

oracle_c = crop(oracle_map)
static_c = crop(static_map)
local_c = crop(local_map)
static_err_c = crop(static_err)
local_err_c = crop(local_err)
improve_c = crop(abs_improve)

# Use longitude offset to avoid wrap issues.
dlon = (np.arange(-lon_half, lon_half + 1) * 0.5)
lat_vals = 90.0 - np.arange(lat0, lat1) * 0.5
obs_lat = 90.0 - lat_idx * 0.5
obs_lon = lon_idx * 0.5

vmax = np.nanpercentile(np.abs(np.concatenate([
    oracle_c.ravel(), static_c.ravel(), local_c.ravel()
])), 99)
err_vmax = np.nanpercentile(np.abs(np.concatenate([
    static_err_c.ravel(), local_err_c.ravel()
])), 99)
imp_vmax = np.nanpercentile(np.abs(improve_c.ravel()), 99)

fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)

top_maps = [
    (oracle_c, "Oracle GEFS response"),
    (static_c, "Static archive response"),
    (local_c, "Obs-conditioned Laplacian response"),
]

for ax, (A, title) in zip(axes[0], top_maps):
    im = ax.imshow(
        A,
        origin="upper",
        extent=[dlon.min(), dlon.max(), lat_vals[-1], lat_vals[0]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.scatter([0], [obs_lat], marker="*", s=110, c="black")
    ax.set_title(title)
    ax.set_xlabel("Longitude offset from observation (deg)")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.78)

bottom_maps = [
    (static_err_c, "Static − oracle error", "RdBu_r", -err_vmax, err_vmax),
    (local_err_c, "Obs-conditioned − oracle error", "RdBu_r", -err_vmax, err_vmax),
    (improve_c, "|static error| − |local error|", "BrBG", -imp_vmax, imp_vmax),
]

for ax, (A, title, cmap, vmin, vmax0) in zip(axes[1], bottom_maps):
    im = ax.imshow(
        A,
        origin="upper",
        extent=[dlon.min(), dlon.max(), lat_vals[-1], lat_vals[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax0,
        aspect="auto",
    )
    ax.scatter([0], [obs_lat], marker="*", s=110, c="black")
    ax.set_title(title)
    ax.set_xlabel("Longitude offset from observation (deg)")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.78)

fig.suptitle(
    f"GEFS covariance response map: {source_var} observation, query {tags[q]}\n"
    f"obs lat={obs_lat:.1f}°, lon={obs_lon:.1f}°E | local gain={row['gain']:.3f}",
    fontsize=14,
)

out_png = out_dir / "gefs_latlon_covariance_response_oracle_static_local.png"
fig.savefig(out_png, dpi=240, bbox_inches="tight")
plt.close(fig)

info_path = out_dir / "gefs_latlon_covariance_response_info.txt"
with open(info_path, "w") as f:
    f.write("Selected GEFS covariance-response map example\n")
    f.write("=" * 80 + "\n")
    f.write(f"metadata: {metadata_path}\n")
    f.write(f"csv: {csv_path}\n")
    f.write(f"query: {q}\n")
    f.write(f"query_tag: {tags[q]}\n")
    f.write(f"source_var: {source_var}\n")
    f.write(f"lat_idx: {lat_idx}\n")
    f.write(f"lon_idx: {lon_idx}\n")
    f.write(f"obs_lat: {obs_lat}\n")
    f.write(f"obs_lon_E: {obs_lon}\n")
    f.write(f"static_local_error: {row['static_local']}\n")
    f.write(f"local_obs_laplacian_error: {row['best_local']}\n")
    f.write(f"gain: {row['gain']}\n")

print("Wrote:")
print(out_png)
print(info_path)
