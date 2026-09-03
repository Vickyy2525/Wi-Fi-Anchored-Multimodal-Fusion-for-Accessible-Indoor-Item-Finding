"""
IMU 軌跡推算與視覺化 (整合 Wi-Fi 指紋閉環校正與地標回推導航)

修正版：
  - 扣除重力後，使用手機水平面 acc_x / acc_z 積分
  - 旋轉軌跡使「主要前進方向」朝上
  - 新增: Wi-Fi 閉環特徵優化 (Graph Relaxation)，拉直飄移的 IMU 軌跡
  - 新增: Wi-Fi 地標回推導航，輔助找物
"""

from __future__ import annotations

import glob
import os
from typing import Any

from nav_i18n import display_landmark_label, nav_t

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _setup_cjk_font() -> None:
    from matplotlib import font_manager
    candidates = [
        "PingFang TC", "PingFang SC", "Heiti TC",
        "Microsoft JhengHei", "Microsoft YaHei",
        "Noto Sans CJK TC", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def load_imu_data(data_folder: str) -> pd.DataFrame:
    imu_files = sorted(glob.glob(os.path.join(data_folder, "imu_*.csv")))
    if not imu_files:
        return pd.DataFrame()

    frames = []
    for path in imu_files:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        frames.append(df)

    imu = pd.concat(frames, ignore_index=True)
    imu = imu.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    for col in ("acc_x", "acc_y", "acc_z", "gyro_mag", "jerk"):
        if col in imu.columns:
            imu[col] = pd.to_numeric(imu[col], errors="coerce").fillna(0.0)
        else:
            imu[col] = 0.0
    return imu.reset_index(drop=True)


def load_keyframes(data_folder: str) -> pd.DataFrame:
    kf_files = sorted(glob.glob(os.path.join(data_folder, "keyframes_*.csv")))
    if not kf_files:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in kf_files]
    kf = pd.concat(frames, ignore_index=True)
    kf["timestamp"] = pd.to_numeric(kf["timestamp"], errors="coerce")
    return kf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _downsample_imu(imu_df: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    """Keep trajectory responsive: dense phone IMU is often 20k+ rows."""
    n = len(imu_df)
    if n <= max_points:
        return imu_df
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return imu_df.iloc[idx].reset_index(drop=True)


def compute_trajectory(imu_df: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    if imu_df.empty:
        return pd.DataFrame(columns=["timestamp", "x", "y"])

    imu_df = _downsample_imu(imu_df, max_points=max_points)

    gx = float(imu_df["acc_x"].median())
    gz = float(imu_df["acc_z"].median())

    ts = imu_df["timestamp"].to_numpy(dtype=float) / 1000.0
    dt = np.diff(ts, prepend=ts[0])
    dt = np.clip(dt, 0.001, 0.5)
    if len(dt) > 1:
        dt[0] = float(np.median(dt[1:]))

    ax = (imu_df["acc_x"].to_numpy(dtype=float) - gx)
    az = (imu_df["acc_z"].to_numpy(dtype=float) - gz)
    gyro_mag = imu_df["gyro_mag"].to_numpy(dtype=float)
    jerk = imu_df["jerk"].to_numpy(dtype=float)

    lin_mag = np.hypot(ax, az)
    zupt = (gyro_mag < 0.25) & (jerk < 100) & (lin_mag < 0.8)

    n = len(imu_df)
    easts = np.empty(n, dtype=float)
    norths = np.empty(n, dtype=float)
    vx = vy = east = north = 0.0
    for i in range(n):
        if zupt[i]:
            vx = vy = 0.0
        vx = (vx + ax[i] * dt[i]) * 0.90
        vy = (vy + az[i] * dt[i]) * 0.90
        east += vx * dt[i]
        north += vy * dt[i]
        easts[i] = east
        norths[i] = north

    traj = pd.DataFrame({
        "timestamp": imu_df["timestamp"].astype(int).to_numpy(),
        "x": easts,
        "y": norths,
    })
    return _rotate_trajectory_north_up(traj)


def _rotate_trajectory_north_up(traj_df: pd.DataFrame) -> pd.DataFrame:
    if len(traj_df) < 3:
        return traj_df
    pts = traj_df[["x", "y"]].to_numpy(dtype=float)
    centered = pts - pts.mean(axis=0)
    if np.allclose(centered, 0):
        return traj_df
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return traj_df
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(main_dir[0], main_dir[1])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = centered @ rot.T
    out = traj_df.copy()
    out["x"] = rotated[:, 0]
    out["y"] = rotated[:, 1]
    return out


def interpolate_trajectory_position(traj_df: pd.DataFrame, timestamp: int | float | None) -> tuple[float, float] | None:
    if traj_df.empty or timestamp is None:
        return None
    ts = int(timestamp)
    t_arr = traj_df["timestamp"].to_numpy()
    if ts <= t_arr[0]:
        return float(traj_df.iloc[0]["x"]), float(traj_df.iloc[0]["y"])
    if ts >= t_arr[-1]:
        return float(traj_df.iloc[-1]["x"]), float(traj_df.iloc[-1]["y"])

    idx = np.searchsorted(t_arr, ts)
    t0, t1 = t_arr[idx - 1], t_arr[idx]
    r0, r1 = traj_df.iloc[idx - 1], traj_df.iloc[idx]
    ratio = (ts - t0) / max(t1 - t0, 1)
    x = r0["x"] + ratio * (r1["x"] - r0["x"])
    y = r0["y"] + ratio * (r1["y"] - r0["y"])
    return float(x), float(y)


def _add_north_arrow(ax, x=0.92, y=0.92, lang: str = "zh"):
    ax.annotate(
        nav_t("north", lang), xy=(x, y), xycoords="axes fraction", fontsize=11,
        fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9),
    )


def load_wifi_signature_rows(data_folder: str) -> pd.DataFrame:
    wifi_files = sorted(glob.glob(os.path.join(data_folder, "wifi_*.csv")))
    if not wifi_files:
        wifi_files = sorted(glob.glob(os.path.join(data_folder, "wifi.csv")))
    if not wifi_files:
        return pd.DataFrame(columns=["timestamp", "bssid", "level"])
    frames = []
    for path in wifi_files:
        try:
            df = pd.read_csv(path)
            if not {"timestamp", "bssid", "level"}.issubset(df.columns):
                continue
            df = df[["timestamp", "bssid", "level"]].copy()
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["timestamp", "bssid", "level"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["timestamp", "bssid", "level"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def compute_wifi_similarity(fp1: dict, fp2: dict) -> float:
    """計算兩個 Wi-Fi 指紋的餘弦相似度。"""
    if not fp1 or not fp2:
        return 0.0
    bssids = set(fp1.keys()).union(set(fp2.keys()))
    if not bssids:
        return 0.0

    v1 = np.array([max(fp1.get(b, -100), -100) for b in bssids])
    v2 = np.array([max(fp2.get(b, -100), -100) for b in bssids])

    # 轉換為正權重訊號 (-100 -> 0, -30 -> 70)
    v1 = np.clip(v1 + 100, 0, 100)
    v2 = np.clip(v2 + 100, 0, 100)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _map_wifi_scans_to_traj(
    traj_df: pd.DataFrame,
    fps: dict[int, dict],
    max_dt_ms: int = 8000,
) -> list[tuple[int, int]]:
    """Return [(traj_idx, wifi_ts), ...] for scans near the trajectory timeline."""
    if traj_df.empty or not fps:
        return []
    t_arr = traj_df["timestamp"].to_numpy(dtype=np.int64)
    nodes: list[tuple[int, int]] = []
    for wts in sorted(fps.keys()):
        idx = int(np.searchsorted(t_arr, wts))
        if idx <= 0:
            ti = 0
        elif idx >= len(t_arr):
            ti = len(t_arr) - 1
        else:
            ti = idx if abs(int(t_arr[idx]) - wts) < abs(int(t_arr[idx - 1]) - wts) else idx - 1
        if abs(int(t_arr[ti]) - int(wts)) > max_dt_ms:
            continue
        nodes.append((ti, int(wts)))
    return nodes


def _smooth_path(P: np.ndarray, window: int = 21) -> np.ndarray:
    """Light moving-average smoother for display (keeps endpoints)."""
    n = len(P)
    if n < 5 or window < 3:
        return P
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return P
    kernel = np.ones(w, dtype=float) / w
    sm = np.empty_like(P)
    for d in range(P.shape[1]):
        pad = w // 2
        ext = np.pad(P[:, d], (pad, pad), mode="edge")
        sm[:, d] = np.convolve(ext, kernel, mode="valid")
    sm[0] = P[0]
    sm[-1] = P[-1]
    return sm


def _wifi_mds_layout(sim: np.ndarray, imu_scan_xy: np.ndarray) -> np.ndarray:
    """Classical MDS on (1 - similarity); scale/align to IMU scan cloud."""
    m = sim.shape[0]
    if m == 1:
        return imu_scan_xy.copy()
    D = np.clip(1.0 - sim, 0.02, 1.0)
    np.fill_diagonal(D, 0.0)
    # Mix a little IMU geometry so MDS doesn't fully collapse when all APs look alike
    imu_d = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            d = float(np.linalg.norm(imu_scan_xy[i] - imu_scan_xy[j]))
            imu_d[i, j] = imu_d[j, i] = d
    med_imu = float(np.median(imu_d[imu_d > 0])) if np.any(imu_d > 0) else 1.0
    med_w = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
    scale = med_imu / max(med_w, 1e-6)
    # Wi-Fi dominates when fingerprints differ; IMU distance keeps spread when they don't
    alpha = float(os.getenv("WIFI_MDS_ALPHA", "0.65"))
    Dist = alpha * (D * scale) + (1.0 - alpha) * imu_d

    # Classical MDS
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J @ (Dist ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order][:2], 0, None)
    evecs = evecs[:, order][:, :2]
    Y = evecs * np.sqrt(evals + 1e-12)

    # Kabsch align Y → imu_scan_xy
    mu_y, mu_i = Y.mean(0), imu_scan_xy.mean(0)
    A = Y - mu_y
    B2 = imu_scan_xy - mu_i
    H = A.T @ B2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale_r = float(np.linalg.norm(B2) / max(np.linalg.norm(A), 1e-9))
    return (A @ R) * scale_r + mu_i


def _collapse_wifi_detours(
    P: np.ndarray,
    idxs: list[int],
    ts_list: list[int],
    fps: dict[int, dict],
    sim_thr: float,
) -> tuple[np.ndarray, int]:
    """If Wi-Fi says two scans are the same place but IMU wandered, rubber-band the detour."""
    P = P.copy()
    collapsed = 0
    m = len(idxs)
    for a in range(m):
        for b in range(a + 1, m):
            if abs(ts_list[b] - ts_list[a]) < 15000:
                continue
            sim = compute_wifi_similarity(fps[ts_list[a]], fps[ts_list[b]])
            if sim < sim_thr:
                continue
            ia, ib = idxs[a], idxs[b]
            if ib - ia < 8:
                continue
            seg = P[ia : ib + 1]
            chord = float(np.linalg.norm(seg[-1] - seg[0]))
            path_len = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
            if path_len < max(2.2 * max(chord, 0.15), 1.5):
                continue
            # Blend strength grows with similarity (Wi-Fi wins on "same place")
            blend = 0.35 + 0.55 * min(1.0, (sim - sim_thr) / max(1e-6, 1.0 - sim_thr))
            n_seg = ib - ia
            for k in range(ia, ib + 1):
                alpha = (k - ia) / max(n_seg, 1)
                target = (1.0 - alpha) * P[ia] + alpha * P[ib]
                P[k] = (1.0 - blend) * P[k] + blend * target
            collapsed += 1
    return P, collapsed


def optimize_trajectory_with_wifi(traj_df: pd.DataFrame, data_folder: str) -> tuple[pd.DataFrame, dict]:
    """IMU + Wi-Fi complementary fusion.

    Wi-Fi decides *where places are* (MDS + same-place detour collapse).
    IMU only keeps local shape between consecutive scans, then gets warped onto
    those Wi-Fi anchors — so the drawn path should visibly differ from raw IMU.
    """
    if traj_df.empty:
        return traj_df, {"status": "empty_traj"}

    wifi_raw = load_wifi_signature_rows(data_folder)
    if wifi_raw.empty:
        out = traj_df.copy()
        P = _smooth_path(out[["x", "y"]].to_numpy(dtype=float), window=31)
        out["x"], out["y"] = P[:, 0], P[:, 1]
        return out, {"status": "no_wifi_data_smoothed_imu"}

    fps: dict[int, dict] = {}
    for ts, grp in wifi_raw.groupby("timestamp"):
        fps[int(ts)] = dict(zip(grp["bssid"], grp["level"]))
    if len(fps) < 2:
        out = traj_df.copy()
        P = _smooth_path(out[["x", "y"]].to_numpy(dtype=float), window=31)
        out["x"], out["y"] = P[:, 0], P[:, 1]
        return out, {"status": "insufficient_wifi_scans", "wifi_scans": len(fps)}

    scan_nodes = _map_wifi_scans_to_traj(traj_df, fps)
    if len(scan_nodes) < 2:
        out = traj_df.copy()
        P = _smooth_path(out[["x", "y"]].to_numpy(dtype=float), window=31)
        out["x"], out["y"] = P[:, 0], P[:, 1]
        return out, {"status": "wifi_scans_not_on_traj", "wifi_scans": len(fps)}

    P0 = traj_df[["x", "y"]].to_numpy(dtype=float).copy()
    P = P0.copy()
    idxs = [i for i, _ in scan_nodes]
    ts_list = [ts for _, ts in scan_nodes]
    m = len(scan_nodes)
    # Slightly lower default so same-room revisits still trigger collapse
    sim_thr = float(os.getenv("WIFI_FUSION_SIM_THRESHOLD", "0.70"))

    # 1) Collapse IMU detours when Wi-Fi says "same place"
    P, n_collapse = _collapse_wifi_detours(P, idxs, ts_list, fps, sim_thr)

    # 2) Wi-Fi MDS anchors for scan nodes
    sim_mat = np.eye(m, dtype=float)
    for a in range(m):
        for b in range(a + 1, m):
            s = compute_wifi_similarity(fps[ts_list[a]], fps[ts_list[b]])
            sim_mat[a, b] = sim_mat[b, a] = s
    imu_scan_xy = P[idxs].copy()
    wifi_xy = _wifi_mds_layout(sim_mat, imu_scan_xy)

    # 3) Warp each IMU segment so endpoints land on Wi-Fi anchors (piecewise similarity)
    out = P.copy()
    # Before first scan: translate with nearest anchor offset
    out[: idxs[0] + 1] = P[: idxs[0] + 1] + (wifi_xy[0] - P[idxs[0]])
    for s in range(m - 1):
        ia, ib = idxs[s], idxs[s + 1]
        src = P[ia : ib + 1].copy()
        if len(src) < 2:
            out[ia] = wifi_xy[s]
            continue
        src0, src1 = src[0].copy(), src[-1].copy()
        dst0, dst1 = wifi_xy[s], wifi_xy[s + 1]
        v_src = src1 - src0
        v_dst = dst1 - dst0
        len_src = float(np.linalg.norm(v_src))
        len_dst = float(np.linalg.norm(v_dst))
        if len_src < 1e-9:
            for k in range(len(src)):
                a = k / max(len(src) - 1, 1)
                out[ia + k] = (1 - a) * dst0 + a * dst1
            continue
        scale = len_dst / len_src if len_dst > 1e-9 else 1.0
        ang = float(np.arctan2(v_dst[1], v_dst[0]) - np.arctan2(v_src[1], v_src[0]))
        c, sng = np.cos(ang), np.sin(ang)
        R = np.array([[c, -sng], [sng, c]], dtype=float)
        rel = (src - src0) @ R.T * scale
        out[ia : ib + 1] = rel + dst0
    if idxs[-1] < len(out) - 1:
        out[idxs[-1] :] = P[idxs[-1] :] + (wifi_xy[-1] - P[idxs[-1]])

    # 4) Soft same-place pull (high sim → share one point)
    lc_edges = 0
    for a in range(m):
        for b in range(a + 1, m):
            if abs(ts_list[b] - ts_list[a]) < 15000:
                continue
            sim = sim_mat[a, b]
            if sim < sim_thr:
                continue
            ia, ib = idxs[a], idxs[b]
            mid = 0.5 * (out[ia] + out[ib])
            w = 0.5 + 0.45 * min(1.0, (sim - sim_thr) / max(1e-6, 1.0 - sim_thr))
            out[ia] = (1 - w) * out[ia] + w * mid
            out[ib] = (1 - w) * out[ib] + w * mid
            lc_edges += 1

    out = _smooth_path(out, window=21)
    delta = np.linalg.norm(out - P0, axis=1)
    out_df = traj_df.copy()
    out_df["x"] = out[:, 0]
    out_df["y"] = out[:, 1]
    return out_df, {
        "status": "optimized",
        "mode": "wifi_skeleton_imu_warp",
        "edges": lc_edges,
        "wifi_scans": m,
        "detours_collapsed": n_collapse,
        "sim_threshold": sim_thr,
        "mean_displace": float(delta.mean()),
        "max_displace": float(delta.max()),
        "pathlen_imu": float(np.linalg.norm(np.diff(P0, axis=0), axis=1).sum()),
        "pathlen_fused": float(np.linalg.norm(np.diff(out, axis=0), axis=1).sum()),
    }



def generate_wifi_backtrack_guide(
    target_timestamp: int | None,
    data_folder: str,
    landmarks: list[dict] | None,
    traj_df: pd.DataFrame | None = None,
    lang: str = "zh",
) -> str:
    """用軌跡方向 + Wi-Fi 指紋相似度，說明如何走回地標（門／家電等）。"""
    if not target_timestamp or not landmarks:
        return ""

    # Prefer primary landmarks (door / large appliances), else any non-weak, else any
    primary = [lm for lm in landmarks if str(lm.get("role", "")).lower() == "primary"]
    if not primary:
        primary = [lm for lm in landmarks if lm.get("confidence", "high") != "weak"]
    pool = primary if primary else list(landmarks)
    if not pool:
        return ""

    wifi_raw = load_wifi_signature_rows(data_folder)
    fps: dict[int, dict] = {}
    if not wifi_raw.empty:
        for ts, grp in wifi_raw.groupby("timestamp"):
            fps[int(ts)] = dict(zip(grp["bssid"], grp["level"]))

    wifi_ts_arr = np.array(sorted(fps.keys()), dtype=np.int64) if fps else np.array([], dtype=np.int64)

    def nearest_fp(ts: int | None) -> tuple[dict, float]:
        if ts is None or wifi_ts_arr.size == 0:
            return {}, 0.0
        idx = int(np.abs(wifi_ts_arr - int(ts)).argmin())
        dt = abs(int(wifi_ts_arr[idx]) - int(ts))
        if dt > 20000:
            return {}, 0.0
        return fps[int(wifi_ts_arr[idx])], float(dt)

    fp_target, _ = nearest_fp(int(target_timestamp))

    # Score landmarks by path distance + wifi similarity
    best = None
    best_score = -1e9
    details: dict[str, Any] = {}
    for lm in pool:
        lm_ts = lm.get("timestamp")
        path_dist = None
        if traj_df is not None and not getattr(traj_df, "empty", True):
            from landmarks import trajectory_arc_length_to

            a = trajectory_arc_length_to(traj_df, target_timestamp)
            b = trajectory_arc_length_to(traj_df, lm_ts)
            if a is not None and b is not None:
                path_dist = abs(a - b)

        fp_lm, _ = nearest_fp(int(lm_ts) if lm_ts is not None else None)
        sim = compute_wifi_similarity(fp_target, fp_lm) if fp_target and fp_lm else 0.0
        # Higher wifi sim + closer on path is better
        score = sim * 10.0 - (0.15 * path_dist if path_dist is not None else 0.0)
        if score > best_score:
            best_score = score
            best = lm
            details = {"sim": sim, "path_dist": path_dist}

    if best is None:
        return ""

    label = display_landmark_label(best, lang)
    sim = float(details.get("sim") or 0.0)
    path_dist = details.get("path_dist")

    direction = ""
    if traj_df is not None and not getattr(traj_df, "empty", True):
        from landmarks import trajectory_arc_length_to

        t_arc = trajectory_arc_length_to(traj_df, target_timestamp)
        l_arc = trajectory_arc_length_to(traj_df, best.get("timestamp"))
        if t_arc is not None and l_arc is not None:
            if t_arc > l_arc + 0.2:
                direction = nav_t("dir_back_to_start", lang)
            elif l_arc > t_arc + 0.2:
                direction = nav_t("dir_forward_to_end", lang)
            else:
                direction = nav_t("dir_nearby", lang)

    if not direction:
        direction = nav_t("dir_default", lang)

    quote = f"「{label}」" if lang == "zh" else f'"{label}"'
    parts = [nav_t("guide_back_prefix", lang) + direction + quote]
    if path_dist is not None:
        parts.append(nav_t("path_units", lang, n=path_dist))
    if sim >= 0.55:
        parts.append(nav_t("wifi_sim_high", lang, pct=sim * 100))
    elif fps:
        parts.append(nav_t("wifi_sim_hint", lang))
    parts.append(nav_t("after_landmark", lang, label=label))
    sep = "；" if lang == "zh" else "; "
    end = "。" if lang == "zh" else "."
    return sep.join(parts) + end


def load_wifi_anchor_timestamps(data_folder: str) -> list[int]:
    wifi = load_wifi_signature_rows(data_folder)
    if wifi.empty:
        return []
    return sorted({int(t) for t in wifi["timestamp"].tolist()})


def _add_scale_bar(ax, length: float = 2.0, lang: str = "zh") -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = max(x1 - x0, 1e-6)
    span_y = max(y1 - y0, 1e-6)
    bar = min(length, span_x * 0.2)
    bx = x0 + span_x * 0.08
    by = y0 + span_y * 0.08
    ax.plot([bx, bx + bar], [by, by], color="#222", linewidth=3, solid_capstyle="butt", zorder=8)
    ax.plot([bx, bx], [by - span_y * 0.01, by + span_y * 0.01], color="#222", linewidth=2, zorder=8)
    ax.plot([bx + bar, bx + bar], [by - span_y * 0.01, by + span_y * 0.01], color="#222", linewidth=2, zorder=8)
    ax.text(bx + bar / 2, by + span_y * 0.025, nav_t("scale_units", lang, n=bar), ha="center", va="bottom", fontsize=9, color="#222")


def _path_progress_points(traj_df: pd.DataFrame, fractions=(0.25, 0.5, 0.75)) -> list[tuple[float, float, str]]:
    xs = traj_df["x"].to_numpy(dtype=float)
    ys = traj_df["y"].to_numpy(dtype=float)
    if len(xs) < 2:
        return []
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return []
    out = []
    for f in fractions:
        target = total * f
        idx = int(np.searchsorted(cum, target))
        idx = min(max(idx, 0), len(xs) - 1)
        out.append((float(xs[idx]), float(ys[idx]), f"{int(f*100)}%"))
    return out


def plot_trajectory_map(
    traj_df: pd.DataFrame,
    keyframes_df: pd.DataFrame | None = None,
    candidates: list[dict] | None = None,
    query: str = "",
    save_path: str = "trajectory_map.png",
    landmarks: list[dict] | None = None,
    nearby_objects: list[dict] | None = None,
    relative_description: str = "",
    wifi_fusion_note: str = "",
    wifi_timestamps: list[int] | None = None,
    bound_wifi_timestamp: int | None = None,
    bound_wifi_fingerprint: dict | None = None,
    imu_ghost_df: pd.DataFrame | None = None,
    lang: str = "zh",
) -> str | None:
    if traj_df.empty:
        return None

    _setup_cjk_font()
    fig, ax = plt.subplots(figsize=(11, 10))

    xs = traj_df["x"].to_numpy(dtype=float)
    ys = traj_df["y"].to_numpy(dtype=float)

    # Ghost: raw IMU (so fusion difference is visible)
    if imu_ghost_df is not None and not getattr(imu_ghost_df, "empty", True):
        gx = imu_ghost_df["x"].to_numpy(dtype=float)
        gy = imu_ghost_df["y"].to_numpy(dtype=float)
        ax.plot(
            gx, gy, color="#CFD8DC", linewidth=1.2, alpha=0.55, linestyle="--",
            label=nav_t("imu_ghost", lang), zorder=1,
        )

    # Path: Wi-Fi skeleton + IMU warp
    ax.plot(xs, ys, color="#546E7A", linewidth=2.4, alpha=0.9, label=nav_t("wifi_path", lang), zorder=2)

    step = max(len(xs) // 8, 1)
    for i in range(step, len(xs) - 1, step * 2):
        if abs(xs[i] - xs[i - 1]) + abs(ys[i] - ys[i - 1]) < 1e-6:
            continue
        ax.annotate(
            "",
            xy=(xs[i], ys[i]),
            xytext=(xs[i - 1], ys[i - 1]),
            arrowprops=dict(arrowstyle="->", color="#8A93A6", lw=1.0, alpha=0.55),
            zorder=3,
        )

    for px, py, lab in _path_progress_points(traj_df, fractions=(0.5,)):
        ax.scatter([px], [py], c="#FFE082", s=40, marker="o", edgecolors="#F9A825", zorder=4, alpha=0.9)
        ax.annotate(nav_t("midpoint", lang), (px, py), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#F9A825")

    ax.scatter(xs[0], ys[0], c="#43A047", s=120, zorder=6, edgecolors="white", linewidths=1.4, label=nav_t("start", lang))
    ax.annotate(nav_t("start", lang), (xs[0], ys[0]), textcoords="offset points", xytext=(8, 8), fontsize=10, color="#2E7D32", fontweight="bold")
    ax.scatter(xs[-1], ys[-1], c="#90A4AE", s=90, zorder=6, edgecolors="white", linewidths=1.2, label=nav_t("end", lang))
    ax.annotate(nav_t("end", lang), (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 8), fontsize=10, color="#607D8B")

    if wifi_timestamps:
        wxy = []
        for ts in wifi_timestamps:
            # Skip the bound scan here; draw it louder below
            if bound_wifi_timestamp is not None and int(ts) == int(bound_wifi_timestamp):
                continue
            pos = interpolate_trajectory_position(traj_df, ts)
            if pos:
                wxy.append(pos)
        if wxy:
            wx, wy = zip(*wxy)
            ax.scatter(
                wx, wy, c="#CE93D8", s=45, marker="^", zorder=4, alpha=0.45,
                edgecolors="white", linewidths=0.5, label=nav_t("wifi_scan", lang),
            )

    landmark_legend_done = set()
    lm_positions: list[tuple[dict, tuple[float, float]]] = []
    if landmarks:
        for lm in landmarks:
            ts = lm.get("timestamp")
            pos = interpolate_trajectory_position(traj_df, ts)
            if not pos:
                continue
            lm_positions.append((lm, pos))
            label = display_landmark_label(lm, lang)
            weak = lm.get("confidence") == "weak"
            color = "#78909C" if weak else "#455A64"
            marker = lm.get("marker") or "s"
            leg = nav_t("landmark_prefix", lang, label=label)
            ax.scatter(
                pos[0], pos[1],
                c=color, s=160 if not weak else 100,
                marker=marker, zorder=6,
                edgecolors="white",
                linewidths=1.2,
                alpha=0.5 if weak else 0.9,
                label=leg if label not in landmark_legend_done else None,
            )
            landmark_legend_done.add(label)
            ax.annotate(
                label, pos, textcoords="offset points", xytext=(8, 8),
                fontsize=10, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9),
            )

    if nearby_objects:
        labeled = False
        for obj in nearby_objects:
            ts = obj.get("timestamp") or obj.get("keyframe_timestamp") or obj.get("wifi_timestamp")
            pos = interpolate_trajectory_position(traj_df, ts)
            if not pos:
                continue
            name = obj.get("name") or obj.get("object") or "?"
            ax.scatter(
                pos[0], pos[1], c="#90A4AE", s=50, marker="s", zorder=5, alpha=0.6,
                label=nav_t("nearby_objects", lang) if not labeled else None,
            )
            labeled = True
            ax.annotate(str(name), pos, textcoords="offset points", xytext=(6, -10), fontsize=8, color="#607D8B")

    primary_pos = None
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda c: (
                0 if c.get("bound") else 1,
                -float(c.get("score") or 0),
                int(c.get("rank") or 99),
            ),
        )
        primary_drawn = False
        secondary_labeled = False
        for cand in ranked:
            ts = cand.get("keyframe_timestamp")
            if ts is None and cand.get("filename"):
                try:
                    ts = int(str(cand["filename"]).split("_")[1].split(".")[0])
                except Exception:
                    ts = None
            pos = interpolate_trajectory_position(traj_df, ts)
            if not pos:
                continue
            rank = cand.get("rank", "?")
            if not primary_drawn and cand.get("bound"):
                primary_pos = pos
                ax.scatter(
                    pos[0], pos[1], c="#FF1744", s=560, marker="*", zorder=9,
                    edgecolors="#212121", linewidths=1.6, label=nav_t("search_here", lang),
                )
                ax.annotate(
                    nav_t("search_here_query", lang, query=query),
                    pos, textcoords="offset points", xytext=(14, -16),
                    fontsize=12, color="#B71C1C", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE", edgecolor="#FF1744", alpha=0.98),
                )
                primary_drawn = True
            else:
                ax.scatter(
                    pos[0], pos[1], c="#FFCC80", s=70, marker="D", zorder=7, alpha=0.7,
                    edgecolors="#FB8C00", label=nav_t("other_candidates", lang) if not secondary_labeled else None,
                )
                secondary_labeled = True
                ax.annotate(
                    f"#{rank}",
                    pos, textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#EF6C00", alpha=0.85,
                )

    if primary_pos and lm_positions:
        nearest_lm, nearest_xy = min(
            lm_positions,
            key=lambda item: (primary_pos[0] - item[1][0]) ** 2 + (primary_pos[1] - item[1][1]) ** 2,
        )
        # Straight cue + along-path return highlight
        ax.plot(
            [primary_pos[0], nearest_xy[0]],
            [primary_pos[1], nearest_xy[1]],
            linestyle="--", color="#2E7D32", linewidth=2.4, alpha=0.9, zorder=5,
            label=nav_t("back_to_landmark_dir", lang),
        )
        ax.annotate(
            "",
            xy=nearest_xy,
            xytext=primary_pos,
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.2, alpha=0.9),
            zorder=6,
        )
        mx = (primary_pos[0] + nearest_xy[0]) / 2
        my = (primary_pos[1] + nearest_xy[1]) / 2
        ax.annotate(
            nav_t("walk_back_to", lang, label=display_landmark_label(nearest_lm, lang)),
            (mx, my), fontsize=11, color="#1B5E20", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32", alpha=0.96),
        )

        # Highlight trajectory segment between object time and landmark time
        obj_ts = None
        if candidates:
            for cand in candidates:
                if cand.get("bound"):
                    obj_ts = cand.get("keyframe_timestamp")
                    break
        lm_ts = nearest_lm.get("timestamp")
        if obj_ts is not None and lm_ts is not None:
            t0, t1 = sorted([int(obj_ts), int(lm_ts)])
            mask = (traj_df["timestamp"] >= t0) & (traj_df["timestamp"] <= t1)
            seg = traj_df.loc[mask]
            if len(seg) >= 2:
                ax.plot(
                    seg["x"], seg["y"],
                    color="#43A047", linewidth=3.2, alpha=0.85, zorder=4,
                    label=nav_t("back_path_segment", lang),
                )

    # ComSoc highlight: object ↔ time-aligned Wi-Fi fingerprint binding
    if primary_pos and bound_wifi_timestamp is not None:
        wifi_pos = interpolate_trajectory_position(traj_df, bound_wifi_timestamp)
        if wifi_pos:
            n_ap = len(bound_wifi_fingerprint or {})
            ax.scatter(
                [wifi_pos[0]], [wifi_pos[1]],
                c="#6A1B9A", s=220, marker="^", zorder=8,
                edgecolors="#FFD54F", linewidths=2.0,
                label=nav_t("wifi_fingerprint_bind", lang),
            )
            ax.plot(
                [primary_pos[0], wifi_pos[0]],
                [primary_pos[1], wifi_pos[1]],
                linestyle=":", color="#6A1B9A", linewidth=2.2, alpha=0.9, zorder=7,
                label=nav_t("item_wifi_bind", lang),
            )
            ax.annotate(
                nav_t("wifi_fp_ap", lang, n=n_ap),
                wifi_pos, textcoords="offset points", xytext=(10, 12),
                fontsize=10, color="#4A148C", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#6A1B9A", alpha=0.95),
            )

    title = nav_t("map_title", lang, query=query) if query else nav_t("map_title_default", lang)
    ax.set_title(title, fontsize=16, pad=12, fontweight="bold", color="#212121")
    ax.set_xlabel(nav_t("axis_lr", lang), fontsize=10, color="#607D8B")
    ax.set_ylabel(nav_t("axis_fwd", lang), fontsize=10, color="#607D8B")
    ax.grid(True, alpha=0.25, linestyle=":")
    # markerscale shrinks oversized star in legend; labelspacing fixes cramped rows
    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
        labelspacing=1.15,
        handlelength=2.0,
        handletextpad=0.8,
        borderpad=0.9,
        borderaxespad=0.8,
        markerscale=0.55,
    )
    ax.set_aspect("equal", adjustable="datalim")
    _add_north_arrow(ax, lang=lang)
    _add_scale_bar(ax, lang=lang)

    guide = (relative_description or "").strip().split("\n")[0]
    footer = guide if guide else nav_t("map_footer_default", lang)
    fig.text(0.5, 0.015, footer, ha="center", fontsize=11, color="#B71C1C", fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_trajectory_map(
    data_folder: str,
    search_result: dict[str, Any] | None = None,
    save_path: str = "trajectory_map.png",
    landmarks: list[dict] | None = None,
    relative_description: str = "",
    traj_df: pd.DataFrame | None = None,
    lang: str = "zh",
) -> str | None:
    if traj_df is None or getattr(traj_df, "empty", True):
        print("[traj] 載入 IMU…")
        imu = load_imu_data(data_folder)
        if imu.empty:
            return None
        print(f"[traj] 推算軌跡（IMU {len(imu)} 列，上限降採樣）…")
        traj = compute_trajectory(imu)
    else:
        traj = traj_df
        print(f"[traj] 重用既有軌跡（{len(traj)} 點）")

    keyframes = load_keyframes(data_folder)
    query = (search_result or {}).get("query", "")
    candidates = (search_result or {}).get("candidates", [])
    nearby = (search_result or {}).get("nearby_objects") or []
    lms = landmarks if landmarks is not None else (search_result or {}).get("landmarks") or []
    rel = relative_description or (search_result or {}).get("relative_description") or ""

    target_ts = None
    bound = [c for c in candidates if c.get("bound")]
    if bound:
        best = max(bound, key=lambda c: c.get("score") or 0)
        target_ts = best.get("keyframe_timestamp")
    elif candidates:
        target_ts = candidates[0].get("keyframe_timestamp")

    imu_ghost = traj.copy()
    print("[traj] Wi-Fi 骨架 + IMU 段融合…")
    traj, fusion_report = optimize_trajectory_with_wifi(traj, data_folder)
    if fusion_report.get("status") == "optimized":
        fusion_note = (
            f"Wi-Fi 骨架+IMU：{fusion_report.get('wifi_scans')} 掃描、"
            f"收束繞路 {fusion_report.get('detours_collapsed', 0)} 段、"
            f"同地約束 {fusion_report.get('edges')}；"
            f"相對純 IMU 平均位移 {fusion_report.get('mean_displace', 0):.2f}、"
            f"路徑長 {fusion_report.get('pathlen_imu', 0):.1f}→{fusion_report.get('pathlen_fused', 0):.1f}。"
        )
    else:
        fusion_note = f"融合狀態：{fusion_report.get('status')}"
    print(f"[traj] {fusion_note}")

    wifi_guide = generate_wifi_backtrack_guide(target_ts, data_folder, lms, traj_df=traj, lang=lang)
    if wifi_guide:
        # Put return-to-landmark guide first for UI
        rel = (wifi_guide + ("\n" + rel if rel else "")).strip()
        if isinstance(search_result, dict):
            search_result["relative_description"] = rel
            search_result["wifi_backtrack_guide"] = wifi_guide

    if isinstance(search_result, dict):
        search_result["wifi_fusion_report"] = fusion_report
        search_result["relative_description"] = rel or search_result.get("relative_description")

    wifi_ts = load_wifi_anchor_timestamps(data_folder)

    bound_wifi_ts = None
    bound_wifi_fp = None
    if bound:
        best = max(bound, key=lambda c: c.get("score") or 0)
        bound_wifi_ts = best.get("wifi_timestamp")
        bound_wifi_fp = best.get("wifi_fingerprint")
    wifi_bind_payload = (search_result or {}).get("wifi_binding")
    if isinstance(wifi_bind_payload, dict):
        bound_wifi_ts = bound_wifi_ts or wifi_bind_payload.get("wifi_timestamp")
        bound_wifi_fp = bound_wifi_fp or wifi_bind_payload.get("wifi_fingerprint")

    print("[traj] 繪製導航軌跡圖…")
    return plot_trajectory_map(
        traj_df=traj,
        keyframes_df=keyframes,
        candidates=candidates,
        query=query,
        save_path=save_path,
        landmarks=lms,
        nearby_objects=nearby,
        relative_description=rel,
        wifi_fusion_note=fusion_note,
        wifi_timestamps=wifi_ts,
        bound_wifi_timestamp=int(bound_wifi_ts) if bound_wifi_ts is not None else None,
        bound_wifi_fingerprint=bound_wifi_fp if isinstance(bound_wifi_fp, dict) else None,
        imu_ghost_df=imu_ghost,
        lang=lang,
    )
