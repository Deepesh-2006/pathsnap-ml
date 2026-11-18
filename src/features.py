from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    list_trips,
    load_trip,
    make_local_projector,
    point_to_segments_dist_and_proj,
    segment_bearings,
    angle_diff,
)

FEATURE_COLS = [
    "dist_m",
    "dist_rank",
    "dist_diff_best",
    "abs_heading_diff",
    "cos_heading_diff",
    "arc_len_m",
    "proj_t",
    "gps_speed_mps",
    "dt_prev_s",
    "prev_dist_m",
    "next_dist_m",
    "step_along_arc",
]


def prepare_trip_geometry(trip: dict) -> dict:
    trk, nds, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_local_projector(float(trk[:, 0].mean()), float(trk[:, 1].mean()))
    tx, ty = proj(trk[:, 0], trk[:, 1])
    nx, ny = proj(nds[:, 0], nds[:, 1])

    u, v = arcs[:, 0], arcs[:, 1]
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
    tree = cKDTree(np.column_stack([0.5 * (ax + bx), 0.5 * (ay + by)]))

    n_obs = len(trk)
    ts = trk[:, 2]
    dt_prev = np.ones(n_obs, dtype=np.float64)
    if n_obs > 1:
        dt_prev[1:] = np.maximum(np.diff(ts), 1e-3)
        dt_prev[0] = dt_prev[1]

    hdg = np.zeros(n_obs, dtype=np.float64)
    spd = np.zeros(n_obs, dtype=np.float64)
    has_hdg = np.zeros(n_obs, dtype=bool)

    for i in range(n_obs):
        i0, i1 = max(0, i - 1), min(n_obs - 1, i + 1)
        dx, dy = tx[i1] - tx[i0], ty[i1] - ty[i0]
        step = np.hypot(dx, dy)
        spd[i] = step / max(ts[i1] - ts[i0], 1e-3)
        if step >= 1.0:
            hdg[i] = np.arctan2(dy, dx)
            has_hdg[i] = True

    return {
        "tx": tx,
        "ty": ty,
        "ts": ts,
        "dt_prev": dt_prev,
        "gps_heading": hdg,
        "gps_speed": spd,
        "has_heading": has_hdg,
        "ax": ax,
        "ay": ay,
        "bx": bx,
        "by": by,
        "arc_len": np.hypot(bx - ax, by - ay),
        "arc_bearing": segment_bearings(ax, ay, bx, by),
        "tree": tree,
        "nx": nx,
        "ny": ny,
        "arcs": arcs,
        "proj": proj,
    }


def assign_reference_arcs(trip: dict, geom: dict) -> np.ndarray | None:
    rt = trip.get("route")
    if rt is None or len(rt) == 0:
        return None
    tx, ty = geom["tx"], geom["ty"]
    ax, ay, bx, by = geom["ax"][rt], geom["ay"][rt], geom["bx"][rt], geom["by"][rt]
    ref = np.empty(len(tx), dtype=np.int64)
    for i in range(len(tx)):
        d, _, _, _ = point_to_segments_dist_and_proj(tx[i], ty[i], ax, ay, bx, by)
        ref[i] = rt[int(np.argmin(d))]
    return ref


def extract_trip_candidates(
    trip: dict,
    top_k: int = config.TOP_K,
    search_k: int = 60,
    with_labels: bool = True,
) -> tuple[pd.DataFrame, dict]:
    geom = prepare_trip_geometry(trip)
    tx, ty = geom["tx"], geom["ty"]
    ax, ay, bx, by = geom["ax"], geom["ay"], geom["bx"], geom["by"]
    n_obs, n_arcs = len(tx), len(ax)
    kq, kk = min(search_k, n_arcs), min(top_k, n_arcs)
    ref = assign_reference_arcs(trip, geom) if with_labels else None

    _, idx_mat = geom["tree"].query(np.column_stack([tx, ty]), k=kq)
    if kq == 1:
        idx_mat = idx_mat.reshape(-1, 1)

    obs_ids, cands, feats, lbls = [], [], [], []
    missed = 0

    for i in range(n_obs):
        pool = idx_mat[i]
        dists, tp, _, _ = point_to_segments_dist_and_proj(tx[i], ty[i], ax[pool], ay[pool], bx[pool], by[pool])
        ord_i = np.argsort(dists)[:kk]
        ch, ch_d, ch_t = pool[ord_i], dists[ord_i], tp[ord_i]

        ip, inx = max(0, i - 1), min(n_obs - 1, i + 1)
        pd_m, pt, _, _ = point_to_segments_dist_and_proj(tx[ip], ty[ip], ax[ch], ay[ch], bx[ch], by[ch])
        nd_m, nt, _, _ = point_to_segments_dist_and_proj(tx[inx], ty[inx], ax[ch], ay[ch], bx[ch], by[ch])

        if geom["has_heading"][i]:
            hd = angle_diff(geom["gps_heading"][i], geom["arc_bearing"][ch])
            ahd, chd = np.abs(hd), np.cos(hd)
        else:
            ahd, chd = np.full(kk, np.pi / 2.0), np.zeros(kk)

        al = geom["arc_len"][ch]
        blk = np.column_stack([
            ch_d,
            np.arange(kk, dtype=np.float32),
            ch_d - ch_d[0],
            ahd,
            chd,
            al,
            ch_t,
            np.full(kk, geom["gps_speed"][i]),
            np.full(kk, geom["dt_prev"][i]),
            pd_m,
            nd_m,
            (nt - pt) * al,
        ]).astype(np.float32)

        obs_ids.append(np.full(kk, i, dtype=np.int32))
        cands.append(ch.astype(np.int64))
        feats.append(blk)

        if ref is not None:
            m = (ch == ref[i]).astype(np.int8)
            if m.sum() == 0:
                missed += 1
            lbls.append(m)

    df = pd.DataFrame(np.vstack(feats), columns=FEATURE_COLS)
    df["trip_id"] = trip["trip_id"]
    df["obs_idx"] = np.concatenate(obs_ids)
    df["arc_id"] = np.concatenate(cands)
    if ref is not None:
        df["label"] = np.concatenate(lbls)

    return df, {"n_obs": n_obs, "missed_candidates": missed, "miss_rate": float(missed / max(n_obs, 1)), "ref_arcs": ref, "geom": geom}


def build_partition_dataset(pdir: Path, cache_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    dfs, stats = [], {}
    for tid in list_trips(pdir):
        df_t, meta = extract_trip_candidates(load_trip(pdir, tid), with_labels=True)
        dfs.append(df_t)
        stats[tid] = {"n_obs": meta["n_obs"], "missed_candidates": meta["missed_candidates"], "miss_rate": meta["miss_rate"]}
    full_df = pd.concat(dfs, ignore_index=True)
    if cache_path is not None:
        full_df.to_pickle(cache_path)
    return full_df, stats


if __name__ == "__main__":
    tdf, tst = build_partition_dataset(config.TEST_DIR, config.OUTPUTS_DIR / "test_features.pkl")
    print(f"student_test rows={len(tdf)}, miss={sum(v['missed_candidates'] for v in tst.values()) / sum(v['n_obs'] for v in tst.values()):.4f}")
    trdf, trst = build_partition_dataset(config.TRAIN_DIR, config.OUTPUTS_DIR / "train_features.pkl")
    print(f"train rows={len(trdf)}, miss={sum(v['missed_candidates'] for v in trst.values()) / sum(v['n_obs'] for v in trst.values()):.4f}")
