#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def lon_diff_deg(lon, lon0):
    return ((lon - lon0 + 180.0) % 360.0) - 180.0


def make_default_latlon(nlat, nlon):
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    return lat, lon


def crop_local(field, lat, lon, obs_lat, obs_lon, dlat=18.0, dlon=28.0):
    dlon_arr = lon_diff_deg(lon, obs_lon)
    dlat_arr = lat - obs_lat

    lat_mask = np.abs(dlat_arr) <= dlat
    lon_mask = np.abs(dlon_arr) <= dlon

    sub = field[np.ix_(lat_mask, lon_mask)]
    yy = dlat_arr[lat_mask]
    xx = dlon_arr[lon_mask] * np.cos(np.deg2rad(obs_lat))

    order_x = np.argsort(xx)
    xx = xx[order_x]
    sub = sub[:, order_x]

    order_y = np.argsort(yy)
    yy = yy[order_y]
    sub = sub[order_y, :]

    return sub, xx, yy


def nice_symmetric_limit(*arrays, pct=98.5):
    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays])
    lim = np.percentile(np.abs(vals), pct)
    return max(float(lim), 1e-8)


def draw_panel(ax, arr, x, y, title, lim, cmap="RdBu_r", show_signs=False):
    levels = np.linspace(-lim, lim, 15)

    im = ax.contourf(x, y, arr, levels=levels, cmap=cmap, extend="both")
    ax.contour(x, y, arr, levels=levels, colors="k", linewidths=0.35, alpha=0.42)
    ax.contour(x, y, arr, levels=[0.0], colors="k", linewidths=1.05)

    ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.plot(
        0.0, 0.0,
        marker="*", markersize=12,
        markerfacecolor="black",
        markeredgecolor="white",
        zorder=10,
    )

    # Kalnay-style sign markers: use only for physical response panels,
    # not for error/reduction diagnostic panels.
    if show_signs:
        halo = [pe.withStroke(linewidth=3.0, foreground="white")]
        if np.nanmax(arr) > 0:
            iy, ix = np.unravel_index(np.nanargmax(arr), arr.shape)
            ax.text(
                x[ix], y[iy], "+",
                fontsize=16, weight="bold",
                ha="center", va="center", color="black",
                path_effects=halo, zorder=12
            )
        if np.nanmin(arr) < 0:
            iy, ix = np.unravel_index(np.nanargmin(arr), arr.shape)
            ax.text(
                x[ix], y[iy], "−",
                fontsize=18, weight="bold",
                ha="center", va="center", color="black",
                path_effects=halo, zorder=12
            )

    ax.set_title(title, fontsize=12, pad=7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta x$ from observation", fontsize=10)
    ax.set_ylabel(r"$\Delta y$ from observation", fontsize=10)
    ax.tick_params(labelsize=9)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="gefs_kalnay_style_cov_response_polished")
    ap.add_argument("--dlat", type=float, default=18.0)
    ap.add_argument("--dlon", type=float, default=28.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z = np.load(args.bundle, allow_pickle=True)

    oracle = z["oracle_response"].astype(float)
    static = z["static_response"].astype(float)
    adaptive = z["obs_conditioned_laplacian_response"].astype(float)

    obs_lat = float(z["obs_lat"])
    obs_lon = float(z["obs_lon"])
    gain = float(z["local_gain"])
    query = str(z["query_label"])
    var_name = str(z["var_name"])

    nlat, nlon = oracle.shape
    lat, lon = make_default_latlon(nlat, nlon)

    oracle_c, x, y = crop_local(oracle, lat, lon, obs_lat, obs_lon, args.dlat, args.dlon)
    static_c, _, _ = crop_local(static, lat, lon, obs_lat, obs_lon, args.dlat, args.dlon)
    adaptive_c, _, _ = crop_local(adaptive, lat, lon, obs_lat, obs_lon, args.dlat, args.dlon)

    static_err = static_c - oracle_c
    adaptive_err = adaptive_c - oracle_c
    error_reduction = np.abs(static_err) - np.abs(adaptive_err)

    resp_lim = nice_symmetric_limit(oracle_c, static_c, adaptive_c, pct=98.5)
    err_lim = nice_symmetric_limit(static_err, adaptive_err, pct=98.5)
    red_lim = nice_symmetric_limit(error_reduction, pct=98.5)

    fig = plt.figure(figsize=(14.5, 8.2))
    gs = fig.add_gridspec(
        2, 6,
        width_ratios=[1, 1, 1, 0.045, 0.030, 0.045],
        height_ratios=[1, 1],
        left=0.06, right=0.94, top=0.84, bottom=0.10,
        wspace=0.35, hspace=0.38
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    cax_resp = fig.add_subplot(gs[0, 3])
    cax_top_gap = fig.add_subplot(gs[0, 4])
    cax_top_gap.axis("off")
    cax_top_blank = fig.add_subplot(gs[0, 5])
    cax_top_blank.axis("off")

    cax_err = fig.add_subplot(gs[1, 3])
    cax_gap = fig.add_subplot(gs[1, 4])
    cax_gap.axis("off")
    cax_red = fig.add_subplot(gs[1, 5])

    im_resp = draw_panel(ax00, oracle_c, x, y, "Oracle GEFS response", resp_lim, show_signs=True)
    draw_panel(ax01, static_c, x, y, "Static archive response", resp_lim, show_signs=True)
    draw_panel(ax02, adaptive_c, x, y, "Flow-conditioned archive response", resp_lim, show_signs=True)

    im_err = draw_panel(ax10, static_err, x, y, "Static − oracle error", err_lim, show_signs=False)
    draw_panel(ax11, adaptive_err, x, y, "Flow-conditioned − oracle error", err_lim, show_signs=False)
    im_red = draw_panel(
        ax12,
        error_reduction,
        x,
        y,
        "Error reduction: static − flow-conditioned",
        red_lim,
        cmap="BrBG"
    )

    cb1 = fig.colorbar(im_resp, cax=cax_resp)
    cb1.set_label("Covariance response", fontsize=10)
    cb1.ax.tick_params(labelsize=9)

    cb2 = fig.colorbar(im_err, cax=cax_err)
    cb2.set_label("Error", fontsize=10)
    cb2.ax.tick_params(labelsize=9)

    cb3 = fig.colorbar(im_red, cax=cax_red)
    cb3.set_label("Error reduction", fontsize=10)
    cb3.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Observation-centered {var_name} covariance-response functions\n"
        f"probe at ({obs_lat:.1f}°, {obs_lon:.1f}°E), query {query}, local gain = {gain:.3f}",
        fontsize=15,
        y=0.965
    )

    fig.text(
        0.5, 0.035,
        "Black star marks the observation/probe at local coordinate (0,0). "
        "Responses are computed by matrix-free covariance actions, without forming dense B.",
        ha="center",
        va="center",
        fontsize=10
    )

    png = os.path.join(args.outdir, f"{args.tag}.png")
    pdf = os.path.join(args.outdir, f"{args.tag}.pdf")

    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("WROTE:", png)
    print("WROTE:", pdf)


if __name__ == "__main__":
    main()
