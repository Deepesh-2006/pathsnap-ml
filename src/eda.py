import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip, make_local_projector


def inspect_part(pname: str, pdir: Path) -> list[dict]:
    rows = []
    for tid in list_trips(pdir):
        d = load_trip(pdir, tid)
        trk, nds, arcs, rt = d["track"], d["nodes"], d["arcs"], d["route"]
        n_obs, n_nds, n_arcs = len(trk), len(nds), len(arcs)
        n_rt = len(rt) if rt is not None else 0

        ts = trk[:, 2]
        dt = np.diff(ts) if n_obs > 1 else np.array([])
        non1 = int(np.sum(np.abs(dt - 1.0) > 1e-3)) if len(dt) > 0 else 0
        max_dt = float(np.max(dt)) if len(dt) > 0 else 0.0
        min_dt = float(np.min(dt)) if len(dt) > 0 else 0.0

        proj = make_local_projector(float(trk[:, 0].mean()), float(trk[:, 1].mean()))
        tx, ty = proj(trk[:, 0], trk[:, 1])
        sdist = np.hypot(np.diff(tx), np.diff(ty)) if n_obs > 1 else np.array([0.0])

        u, v = arcs[:, 0], arcs[:, 1]
        adj = csr_matrix(
            (np.ones(n_arcs * 2, dtype=np.int8), (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(n_nds, n_nds),
        )
        n_comp, _ = connected_components(adj, directed=False)
        deg = np.bincount(np.concatenate([u, v]), minlength=n_nds)

        rt_cont = np.nan
        if rt is not None and len(rt) > 1:
            ra = arcs[rt]
            rt_cont = float(np.sum(ra[:-1, 1] == ra[1:, 0]) / (len(rt) - 1))

        rows.append({
            "partition": pname,
            "trip_id": tid,
            "n_obs": n_obs,
            "n_nodes": n_nds,
            "n_arcs": n_arcs,
            "n_route": n_rt,
            "nans_track": int(np.isnan(trk).sum()),
            "nans_nodes": int(np.isnan(nds).sum()),
            "non_1s_gaps": non1,
            "min_dt_s": min_dt,
            "max_dt_s": max_dt,
            "total_track_km": float(np.sum(sdist) / 1000.0),
            "max_gps_jump_m": float(np.max(sdist)),
            "n_components": int(n_comp),
            "isolated_nodes": int(np.sum(deg == 0)),
            "route_continuity": rt_cont,
            "lon_min": float(trk[:, 0].min()),
            "lon_max": float(trk[:, 0].max()),
            "lat_min": float(trk[:, 1].min()),
            "lat_max": float(trk[:, 1].max()),
        })
    return rows


def main():
    df = pd.DataFrame(inspect_part("train", config.TRAIN_DIR) + inspect_part("student_test", config.TEST_DIR))
    df.to_csv(config.OUTPUTS_DIR / "eda_trips.csv", index=False)

    summary = {}
    for p in ("train", "student_test"):
        s = df[df["partition"] == p]
        summary[p] = {
            "num_trips": int(len(s)),
            "total_gps_obs": int(s["n_obs"].sum()),
            "obs_per_trip_min_med_max": [int(s["n_obs"].min()), float(s["n_obs"].median()), int(s["n_obs"].max())],
            "nans_total": int(s["nans_track"].sum() + s["nans_nodes"].sum()),
            "non_1s_fraction": float(s["non_1s_gaps"].sum() / max(s["n_obs"].sum() - len(s), 1)),
            "max_dt_s": float(s["max_dt_s"].max()),
            "min_dt_s": float(s["min_dt_s"].min()),
            "median_nodes": float(s["n_nodes"].median()),
            "median_arcs": float(s["n_arcs"].median()),
            "median_route_len": float(s["n_route"].median()),
            "mean_route_continuity": float(s["route_continuity"].mean()),
        }

    with open(config.OUTPUTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== PathSnap Dataset EDA ===")
    for p in ("train", "student_test"):
        s = df[df["partition"] == p]
        st = summary[p]
        print(f"\n[{p}] trips={st['num_trips']}, total_gps_obs={st['total_gps_obs']:,}, NaNs={st['nans_total']}")
        print(f"  obs/trip (min/med/max): {st['obs_per_trip_min_med_max']}")
        print(f"  non-1s gaps: {st['non_1s_fraction']*100:.2f}% | dt range: [{st['min_dt_s']:.1f}s, {st['max_dt_s']:.1f}s]")
        print(f"  median network: {st['median_nodes']:.0f} nodes, {st['median_arcs']:.0f} arcs | median route: {st['median_route_len']:.0f} arcs")
        print(f"  mean route continuity: {st['mean_route_continuity']:.4f}")
        print(f"  connected components (med/max): {s['n_components'].median():.0f} / {s['n_components'].max()} | isolated nodes (med/max): {s['isolated_nodes'].median():.0f} / {s['isolated_nodes'].max()}")
        print(f"  track km (min/med/max): {s['total_track_km'].min():.2f} / {s['total_track_km'].median():.2f} / {s['total_track_km'].max():.2f}")

    top_gaps = df.nlargest(5, "max_dt_s")[["partition", "trip_id", "n_obs", "max_dt_s", "max_gps_jump_m"]]
    print("\nTop 5 trips by max timestamp gap:")
    print(top_gaps.to_string(index=False))

    plt.figure(figsize=(8, 4))
    for p, c in [("train", "#2b5c8f"), ("student_test", "#d95f02")]:
        plt.hist(df[df["partition"] == p]["n_obs"], bins=15, alpha=0.65, label=p, color=c, edgecolor="white")
    plt.xlabel("GPS observations per trip")
    plt.ylabel("Trip count")
    plt.title("Distribution of Trip Lengths (GPS Observations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "eda_trip_lengths.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(df["n_nodes"], df["n_arcs"], c=df["partition"].map({"train": "#2b5c8f", "student_test": "#d95f02"}), alpha=0.8)
    ax[0].set_xlabel("Road network nodes")
    ax[0].set_ylabel("Road network arcs")
    ax[0].set_title("Road Network Size per Trip")
    ax[1].hist(df["max_dt_s"], bins=20, color="#4c72b0", edgecolor="white")
    ax[1].set_xlabel("Max timestamp gap per trip (s)")
    ax[1].set_ylabel("Trip count")
    ax[1].set_title("Maximum GPS Time Gap per Trip")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "eda_network_and_gaps.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
