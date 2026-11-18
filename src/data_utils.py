from pathlib import Path
import numpy as np

R_EARTH = 6371000.0


def list_trips(pdir: Path) -> list[str]:
    return sorted(p.name for p in pdir.iterdir() if p.is_dir())


def load_trip(pdir: Path, tid: str) -> dict:
    tdir = pdir / tid
    trk = np.loadtxt(tdir / f"{tid}.track", dtype=np.float64)
    nds = np.loadtxt(tdir / f"{tid}.nodes", dtype=np.float64)
    arcs = np.loadtxt(tdir / f"{tid}.arcs", dtype=np.int64)

    if trk.ndim == 1:
        trk = trk.reshape(1, -1)
    if nds.ndim == 1:
        nds = nds.reshape(1, -1)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    rpath = tdir / f"{tid}.route"
    rt = None
    if rpath.exists():
        rt = np.loadtxt(rpath, dtype=np.int64)
        if rt.ndim == 0:
            rt = rt.reshape(1)

    return {"trip_id": tid, "track": trk, "nodes": nds, "arcs": arcs, "route": rt}


def make_local_projector(lon0: float, lat0: float):
    m_lat = np.deg2rad(1.0) * R_EARTH
    m_lon = m_lat * np.cos(np.deg2rad(lat0))

    def proj(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (lon - lon0) * m_lon, (lat - lat0) * m_lat

    return proj


def point_to_segments_dist_and_proj(
    px: float, py: float, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    wx, wy = px - ax, py - ay
    t = np.where(l2 > 1e-12, (wx * vx + wy * vy) / np.maximum(l2, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t, qx, qy


def segment_bearings(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    return np.arctan2(by - ay, bx - ax)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi
