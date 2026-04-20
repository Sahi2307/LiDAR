import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
import json
import os
import gc

# ── paths ────────────────────────────────────────────────────────────────────────────
RAW_LAS    = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\laser_left.las'
HAG_LAS    = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\_temp_hag.las'
OUT_LAS    = 'poles_wires_with_downguy.las'
OUT_QML    = 'poles_wires_with_downguy.qml'
OUT_SPANS  = 'span_report_with_downguy.csv'

# ── global shift (from poles.ipynb — keeps arithmetic stable) ──────────────────────
LOCAL_SHIFT = np.array([356534.91, 4661602.80, 215.93], dtype=np.float64)

# ── load ────────────────────────────────────────────────────────────────────────────
print(f'Loading {RAW_LAS} ...')
las = laspy.read(RAW_LAS)
raw_pt_fmt = las.header.point_format
raw_version = las.header.version
raw_scales = las.header.scales
raw_offsets = las.header.offsets

points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
local  = (points - LOCAL_SHIFT).astype(np.float32)
print(f'  {len(local):,} points loaded')
del points, las
gc.collect()

# ── spatial indices ───────────────────────────────────────────────────────────
print('Building decimated 2-D tree...')
decimated   = local[::5]
tree_dec_2d = cKDTree(decimated[:, :2])

print('Building full 2-D tree...')
tree_full_2d = cKDTree(local[:, :2])
print('Building full 3-D tree...')
tree_full_3d = cKDTree(local)

print('Spatial indices built.')

# Stage 1 — HAG (Reading pre-generated file)
print(f'Reading HAG from {HAG_LAS} ...')
las_hag = laspy.read(HAG_LAS)
hag     = np.array(las_hag.HeightAboveGround).astype(np.float32)
del las_hag
gc.collect()
print(f'  HAG range: {hag.min():.1f} – {hag.max():.1f} m')
print(f'  Points with HAG > 4 m: {(hag > 4).sum():,}  (pole/wire candidates)')

# Stage 2 — Eigenvalue pre-screen
EIGEN_RADIUS   = 1.2
MIN_NEIGHBOURS = 8
MIN_LINEARITY  = 0.75
MAX_PLANARITY  = 0.25

def extract_signatures(pts_3d):
    if len(pts_3d) < 5: return 0, 0, 0, 0
    pts_c = pts_3d - pts_3d.mean(axis=0)
    cov   = np.cov(pts_c.T) + np.eye(3) * 1e-6
    evals, evecs = np.linalg.eigh(cov)
    idx  = evals.argsort()[::-1]
    L1, L2, L3 = evals[idx]
    linearity  = (L1 - L2) / L1 if L1 > 0 else 0
    planarity  = (L2 - L3) / L1 if L1 > 0 else 0
    z_dir      = abs(evecs[2, idx[0]])
    return linearity, planarity, 0, z_dir

print('Running eigenvalue pre-screen on decimated cloud ...')
pole_cand_idx = []
for i in range(0, len(decimated), 10):
    nbr_idx  = tree_dec_2d.query_ball_point(decimated[i, :2], r=EIGEN_RADIUS)
    if len(nbr_idx) > MIN_NEIGHBOURS:
        lin, pla, _, _ = extract_signatures(decimated[nbr_idx])
        if lin > MIN_LINEARITY and pla < MAX_PLANARITY:
            pole_cand_idx.append(i)

candidate_pts = decimated[pole_cand_idx]
print(f'  {len(candidate_pts):,} eigen-candidate points')

# Stage 3 — DBSCAN
DBSCAN_EPS, DBSCAN_MIN_SAMP = 2.0, 10
MIN_HEIGHT, MAX_WIDTH = 4.0, 4.0
print('DBSCAN clustering ...')
db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMP, n_jobs=-1).fit(candidate_pts)
labels = db.labels_

initial_clusters = []
for lbl in set(labels):
    if lbl == -1: continue
    cl = candidate_pts[labels == lbl]
    extent = cl.max(axis=0) - cl.min(axis=0)
    if extent[2] > MIN_HEIGHT and max(extent[0], extent[1]) < MAX_WIDTH:
        initial_clusters.append(cl)
print(f'  {len(initial_clusters)} clusters pass initial filter')

# Stage 4 — 3-D expansion
EXPAND_XY_RADIUS, EXPAND_Z_BUFFER, TIGHT_RADIUS, MAX_SPREAD = 2.5, 6.0, 1.2, 1.5
expanded_clusters = []
for cl in initial_clusters:
    cxy, max_z, min_z = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    dist_sq = ((local[:, 0] - cxy[0])**2 + (local[:, 1] - cxy[1])**2)
    mask = (dist_sq < EXPAND_XY_RADIUS**2) & (local[:, 2] >= min_z) & (local[:, 2] <= max_z + EXPAND_Z_BUFFER)
    full = local[mask]
    if len(full) == 0: continue
    if np.std(full[:, :2]) < MAX_SPREAD:
        expanded_clusters.append(full)
    else:
        tight = np.linalg.norm(full[:, :2] - cxy, axis=1) < TIGHT_RADIUS
        expanded_clusters.append(full[tight])
print(f'Expansion complete: {len(expanded_clusters)} full structures')

# Stage 5 — Tree rejection
MIN_MEDIAN_HAG, MIN_HIGH_FRAC, MAX_XY_STD = 3.0, 0.30, 1.0
clean_clusters, rejected = [], 0
for cl in expanded_clusters:
    _, idx = tree_full_3d.query(cl, k=1)
    cl_hag = hag[idx]
    if np.median(cl_hag) < MIN_MEDIAN_HAG or (cl_hag > 4.0).sum()/len(cl_hag) < MIN_HIGH_FRAC or np.std(cl[:, :2]) > MAX_XY_STD:
        rejected += 1
    else:
        clean_clusters.append(cl)
print(f'Tree rejection: kept {len(clean_clusters)} poles, rejected {rejected}')

# Stage 6 — Component labelling
CLASS_SHAFT, CLASS_CROSSARM, CLASS_TRANSFORMER = 1, 2, 8
CLASSES = {1:'SHAFT', 2:'CROSSARM', 8:'TRANSFORMER', 9:'DOWNGUY'}
pole_list, pole_tops, clean_pts_list, clean_cls_list, clean_pid_list = [], {}, [], [], []

for pid, cl in enumerate(clean_clusters):
    cxy, top_z, bot_z = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    gx, gy, gz = cxy[0] + LOCAL_SHIFT[0], cxy[1] + LOCAL_SHIFT[1], cl[:, 2].mean() + LOCAL_SHIFT[2]
    pole_tops[pid] = (gx, gy, top_z + LOCAL_SHIFT[2])
    pole_list.append({'Pole_ID': pid, 'X': gx, 'Y': gy, 'Z': gz, 'Height': round(top_z - bot_z, 2)})
    d_centre, h_from_top = np.linalg.norm(cl[:, :2] - cxy, axis=1), top_z - cl[:, 2]
    labels_cl = np.full(len(cl), CLASS_SHAFT, dtype=np.uint8)
    labels_cl[(h_from_top < 1.2) & (d_centre > 0.4)] = CLASS_CROSSARM
    labels_cl[(h_from_top > 2.0) & (h_from_top < 6.0) & (d_centre > 0.5)] = CLASS_TRANSFORMER
    clean_pts_list.append(cl.astype(np.float64) + LOCAL_SHIFT)
    clean_cls_list.append(labels_cl)
    clean_pid_list.append(np.full(len(cl), pid, dtype=np.int32))

poles_df = pd.DataFrame(pole_list)
clean_pts = np.vstack(clean_pts_list); clean_cls = np.concatenate(clean_cls_list); clean_pid = np.concatenate(clean_pid_list)

# Stage 6.5 — Downguy
CLASS_DOWNGUY = 9
GUY_SEARCH_XY, GUY_MIN_HAG_TOP, GUY_MAX_HAG_BOT = 8.0, 3.5, 1.5
GUY_MIN_LIN, GUY_MAX_PLAN, GUY_MIN_Z_DIAG, GUY_MAX_Z_DIAG = 0.65, 0.30, 0.25, 0.90
GUY_MAX_XY_STD, GUY_MIN_HAG_SPAN, GUY_DBSCAN_EPS, GUY_DBSCAN_MSMP, GUY_MIN_PTS, SHAFT_EXCL_R = 0.25, 2.5, 0.4, 5, 15, 0.35

downguy_pts_list, downguy_cls_list, downguy_pid_list, n_guys_total = [], [], [], 0
print('Running downguy detection...')
for pid, cl in enumerate(clean_clusters):
    pole_xy, pole_topZ, pole_botZ = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    dist_xy = np.sqrt((local[:, 0] - pole_xy[0])**2 + (local[:, 1] - pole_xy[1])**2)
    mask = (dist_xy <= GUY_SEARCH_XY) & (local[:, 2] >= pole_botZ - 1.0) & (local[:, 2] <= pole_topZ + 0.5) & (hag <= GUY_MIN_HAG_TOP + 1.0)
    c_idx = np.where(mask)[0]
    if len(c_idx) < GUY_MIN_PTS: continue
    c_pts, c_hag = local[c_idx], hag[c_idx]
    not_shaft = np.linalg.norm(c_pts[:, :2] - pole_xy, axis=1) > SHAFT_EXCL_R
    c_pts, c_hag, c_idx = c_pts[not_shaft], c_hag[not_shaft], c_idx[not_shaft]
    if len(c_pts) < GUY_MIN_PTS: continue
    db_sub = DBSCAN(eps=GUY_DBSCAN_EPS, min_samples=GUY_DBSCAN_MSMP).fit(c_pts)
    for lbl in set(db_sub.labels_):
        if lbl == -1: continue
        s_mask = db_sub.labels_ == lbl
        s_pts, s_hag = c_pts[s_mask], c_hag[s_mask]
        if len(s_pts) < GUY_MIN_PTS: continue
        lin, pla, _, z_dir = extract_signatures(s_pts)
        if lin > GUY_MIN_LIN and pla < GUY_MAX_PLAN and GUY_MIN_Z_DIAG < z_dir < GUY_MAX_Z_DIAG:
            h_max, h_min = s_hag.max(), s_hag.min()
            if h_max >= GUY_MIN_HAG_TOP and h_min <= GUY_MAX_HAG_BOT and (h_max - h_min) >= GUY_MIN_HAG_SPAN and np.std(s_pts[:, :2]) <= GUY_MAX_XY_STD:
                downguy_pts_list.append(s_pts.astype(np.float64) + LOCAL_SHIFT)
                downguy_cls_list.append(np.full(len(s_pts), CLASS_DOWNGUY, dtype=np.uint8))
                downguy_pid_list.append(np.full(len(s_pts), pid, dtype=np.int32))
                n_guys_total += 1
print(f'Downguy detection complete: {n_guys_total} guys found')

if downguy_pts_list:
    clean_pts = np.vstack([clean_pts, np.vstack(downguy_pts_list)])
    clean_cls = np.concatenate([clean_cls, np.concatenate(downguy_cls_list)])
    clean_pid = np.concatenate([clean_pid, np.concatenate(downguy_pid_list)])

# Stage 7 — Lane split
km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(poles_df[['X']].values)
llabels = km.labels_
if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]: llabels = 1 - llabels
poles_df['Lane'] = llabels
pole_lane = {int(r['Pole_ID']): int(r['Lane']) for _, r in poles_df.iterrows()}

# Stage 8 — Robust top Z
POLE_TOP_RADIUS, POLE_TOP_PCT = 1.2, 98
for pid, row in poles_df.iterrows():
    idxs = tree_full_2d.query_ball_point(row[['X', 'Y']] - LOCAL_SHIFT[:2], r=POLE_TOP_RADIUS)
    top_z = float(np.percentile(local[idxs, 2], POLE_TOP_PCT)) + LOCAL_SHIFT[2] if len(idxs) >= 10 else pole_tops[pid][2]
    pole_tops[pid] = (row['X'], row['Y'], top_z)

# Stage 9 — Wire detection
MIN_SPAN, MAX_SPAN, TUBE_RADIUS, Z_BELOW, Z_ABOVE = 5.0, 90.0, 1.5, 3.5, 1.5
CAT_RES_A, CAT_RES_B, MIN_SPAN_COV, WIRE_MIN_LIN, WIRE_MAX_PLA, EIG_K, MAX_ACROSS_STD = 0.6, 1.0, 0.30, 0.65, 0.25, 12, 0.35

def local_eigenfeatures(pts_local, k=EIG_K):
    n = len(pts_local); lin, pla = np.zeros(n), np.ones(n)
    if n < 4: return lin, pla
    lt = cKDTree(pts_local); kk = min(k + 1, n)
    _, nbr_idx = lt.query(pts_local, k=kk)
    for i in range(n):
        nbrs = pts_local[nbr_idx[i][1:]]
        if len(nbrs) < 3: continue
        evals = np.linalg.eigvalsh(np.cov(nbrs.T))
        l1, l2, l3 = sorted(evals, reverse=True)
        if l1 > 1e-10: lin[i], pla[i] = (l1-l2)/l1, (l2-l3)/l1
    return lin, pla

def project_span(pts, A_local, u):
    rel = pts[:, :2] - A_local; along = rel @ u; nv = np.array([-u[1], u[0]]); across = rel @ nv
    return along, across

def fetch_tube(A_local, B_local, aZ_local, bZ_local, span):
    mid3d = np.append((A_local + B_local)/2, (aZ_local + bZ_local)/2)
    u = (B_local - A_local) / span
    cand_g = np.array(tree_full_3d.query_ball_point(mid3d, r=span/2 + TUBE_RADIUS + 2.0), dtype=np.int64)
    if not len(cand_g): return None
    cpts = local[cand_g]; along, across = project_span(cpts, A_local, u)
    mask = (along > 0.3) & (along < span - 0.3) & (np.abs(across) < TUBE_RADIUS) & (cpts[:, 2] > min(aZ_local, bZ_local) - Z_BELOW) & (cpts[:, 2] < max(aZ_local, bZ_local) + Z_ABOVE)
    if mask.sum() < 5: return None
    return cand_g[mask], cpts[mask], along[mask], across[mask]

def wire_pipeline(tube_result, span, cat_res):
    cg, cp, al, ac = tube_result
    try:
        coef = np.linalg.lstsq(np.stack([np.ones_like(al), al, al*al], axis=1), cp[:, 2], rcond=None)[0]
        kp1 = np.abs(cp[:, 2] - (coef[0] + coef[1]*al + coef[2]*al*al)) < cat_res
    except: return []
    if kp1.sum() < 5 or np.std(ac[kp1]) > MAX_ACROSS_STD: return []
    lin, pla = local_eigenfeatures(cp[kp1])
    kp2 = (lin >= WIRE_MIN_LIN) & (pla <= WIRE_MAX_PLA)
    if kp2.sum() < 4 or (al[kp1][kp2].max() - al[kp1][kp2].min())/span < MIN_SPAN_COV: return []
    return cg[kp1][kp2]

wire_assignments, span_rows = {}, []
p_ids, p_xys = poles_df['Pole_ID'].values, poles_df[['X', 'Y']].values
for i in range(len(p_xys)):
    for j in range(i + 1, len(p_xys)):
        d = np.linalg.norm(p_xys[i] - p_xys[j])
        if not (MIN_SPAN < d < MAX_SPAN): continue
        pid_a, pid_b = int(p_ids[i]), int(p_ids[j])
        A_loc, B_loc = p_xys[i] - LOCAL_SHIFT[:2], p_xys[j] - LOCAL_SHIFT[:2]
        aZ_loc, bZ_loc = pole_tops[pid_a][2] - LOCAL_SHIFT[2], pole_tops[pid_b][2] - LOCAL_SHIFT[2]
        tube = fetch_tube(A_loc, B_loc, aZ_loc, bZ_loc, d)
        if tube is None: continue
        pts = wire_pipeline(tube, d, CAT_RES_A)
        p_str = 'A'
        if not len(pts):
            pts = wire_pipeline(tube, d, CAT_RES_B)
            p_str = 'B'
        if not len(pts): continue
        cls = 71 if pole_lane[pid_a] != pole_lane[pid_b] else 70
        for g in pts:
            if int(g) not in wire_assignments: wire_assignments[int(g)] = cls
        span_rows.append({'pole_a': pid_a, 'pole_b': pid_b, 'span_m': round(d, 1), 'kind': 'CROSS_ROAD' if cls==71 else 'SAME_LANE', 'pass': p_str, 'points': len(pts)})
        print(f"  Span {pid_a}-{pid_b}: {len(pts)} pts")

pd.DataFrame(span_rows).to_csv(OUT_SPANS, index=False)

# Stage 10 — Output
wg = np.array(list(wire_assignments.keys()), dtype=np.int64)
w_pts, w_cls = local[wg].astype(np.float64) + LOCAL_SHIFT, np.array([wire_assignments[g] for g in wg], dtype=np.uint8)
final_p, final_c = np.vstack([clean_pts, w_pts]), np.concatenate([clean_cls + 64, w_cls])
final_id = np.clip(np.concatenate([clean_pid, np.zeros(len(w_pts), dtype=np.int32)]) + 1, 0, 255).astype(np.uint8)

header = laspy.LasHeader(point_format=raw_pt_fmt, version=raw_version)
header.offsets, header.scales = raw_offsets, raw_scales
out = laspy.LasData(header)
out.x, out.y, out.z = final_p.T
out.classification, out.user_data = final_c, final_id
out.write(OUT_LAS)
print(f'Done. Output: {OUT_LAS}')
