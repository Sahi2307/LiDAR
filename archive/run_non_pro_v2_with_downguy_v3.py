print('Script started!')
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
import pdal # This will use the local pdal.py shim
import json
import os
import gc
import time

# ── paths ────────────────────────────────────────────────────────────────────────────
RAW_LAS    = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\laser_left.las'
OUT_LAS    = 'poles_wires_final_v3.las'
OUT_QML    = 'poles_wires_final_v3.qml'
OUT_SPANS  = 'span_report_final_v3.csv'

# ── global shift ──────────────────────────────────────────────────────────────────────
LOCAL_SHIFT = np.array([356534.91, 4661602.80, 215.93], dtype=np.float64)

# ── load ────────────────────────────────────────────────────────────────────────────
print(f'Loading {RAW_LAS} ...')
start_time = time.time()
las = laspy.read(RAW_LAS)
raw_pt_fmt = las.header.point_format
raw_version = las.header.version
raw_scales = las.header.scales
raw_offsets = las.header.offsets

points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
local  = (points - LOCAL_SHIFT).astype(np.float32)
print(f'  {len(local):,} points loaded in {time.time()-start_time:.1f}s')
del points, las
gc.collect()

# ── spatial indices ───────────────────────────────────────────────────────────
print('Building spatial indices...')
t0 = time.time()
decimated   = local[::5]
tree_dec_2d = cKDTree(decimated[:, :2])
tree_full_2d = cKDTree(local[:, :2])
tree_full_3d = cKDTree(local)
print(f'  Spatial indices built in {time.time()-t0:.1f}s')

# Stage 1 — PDAL SMRF (Using the shim)
print('Running Stage 1: PDAL SMRF + HAG ...')
t1 = time.time()
# Check if we can reuse the existing temp file to save ~5 mins
TEMP_HAG = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\_temp_hag.las'
if os.path.exists(TEMP_HAG):
    print(f'  Found existing {TEMP_HAG}, reusing it...')
    las_hag = laspy.read(TEMP_HAG)
else:
    pdal_pipeline = [
        {"type": "readers.las", "filename": RAW_LAS},
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn"},
        {"type": "writers.las", "filename": "_temp_hag_v3.las", "extra_dims": "all"}
    ]
    p = pdal.Pipeline(json.dumps(pdal_pipeline))
    p.execute()
    las_hag = laspy.read("_temp_hag_v3.las")

hag = np.array(las_hag.HeightAboveGround).astype(np.float32)
del las_hag
gc.collect()
print(f'  SMRF complete in {time.time()-t1:.1f}s')

# Stage 2 — Eigenvalue pre-screen
print('Running Stage 2: Eigenvalue pre-screen...')
t2 = time.time()
EIGEN_RADIUS, MIN_NEIGHBOURS, MIN_LINEARITY, MAX_PLANARITY = 1.2, 8, 0.75, 0.25

def extract_signatures(pts_3d):
    if len(pts_3d) < 5: return 0, 0, 0, 0
    pts_c = pts_3d - pts_3d.mean(axis=0)
    cov   = np.cov(pts_c.T) + np.eye(3) * 1e-6
    evals, evecs = np.linalg.eigh(cov)
    idx  = evals.argsort()[::-1]
    L1, L2, L3 = evals[idx]
    linearity = (L1 - L2) / L1 if L1 > 0 else 0
    planarity = (L2 - L3) / L1 if L1 > 0 else 0
    z_dir     = abs(evecs[2, idx[0]])
    return linearity, planarity, 0, z_dir

pole_cand_idx = []
for i in range(0, len(decimated), 10):
    nbr_idx = tree_dec_2d.query_ball_point(decimated[i, :2], r=EIGEN_RADIUS)
    if len(nbr_idx) > MIN_NEIGHBOURS:
        lin, pla, _, _ = extract_signatures(decimated[nbr_idx])
        if lin > MIN_LINEARITY and pla < MAX_PLANARITY:
            pole_cand_idx.append(i)
candidate_pts = decimated[pole_cand_idx]
print(f'  {len(candidate_pts):,} candidates in {time.time()-t2:.1f}s')

# Stage 3 — DBSCAN
print('Running Stage 3: DBSCAN...')
db = DBSCAN(eps=2.0, min_samples=10, n_jobs=-1).fit(candidate_pts)
initial_clusters = []
for lbl in set(db.labels_):
    if lbl == -1: continue
    cl = candidate_pts[db.labels_ == lbl]
    extent = cl.max(axis=0) - cl.min(axis=0)
    if extent[2] > 4.0 and max(extent[0], extent[1]) < 4.0:
        initial_clusters.append(cl)
print(f'  {len(initial_clusters)} clusters found')

# Stage 4 — Expansion
print('Running Stage 4: 3-D Expansion...')
expanded_clusters = []
for cl in initial_clusters:
    cxy, max_z, min_z = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    dist_sq = ((local[:, 0] - cxy[0])**2 + (local[:, 1] - cxy[1])**2)
    mask = (dist_sq < 2.5**2) & (local[:, 2] >= min_z) & (local[:, 2] <= max_z + 6.0)
    full = local[mask]
    if len(full) == 0: continue
    if np.std(full[:, :2]) < 1.5:
        expanded_clusters.append(full)
    else:
        tight = np.linalg.norm(full[:, :2] - cxy, axis=1) < 1.2
        expanded_clusters.append(full[tight])

# Stage 5 — Rejection (Using HAG)
print('Running Stage 5: Tree rejection...')
clean_clusters = []
for cl in expanded_clusters:
    _, idx = tree_full_3d.query(cl, k=1)
    cl_hag = hag[idx]
    if np.median(cl_hag) >= 3.0 and (cl_hag > 4.0).sum()/len(cl_hag) >= 0.30 and np.std(cl[:, :2]) <= 1.0:
        clean_clusters.append(cl)
print(f'  Kept {len(clean_clusters)} poles')

# Stage 6 — Component labelling
print('Running Stage 6: Component labelling...')
pole_list, pole_tops, clean_pts_list, clean_cls_list, clean_pid_list = [], {}, [], [], []
for pid, cl in enumerate(clean_clusters):
    cxy, top_z, bot_z = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    gx, gy, gz = cxy[0] + LOCAL_SHIFT[0], cxy[1] + LOCAL_SHIFT[1], cl[:, 2].mean() + LOCAL_SHIFT[2]
    pole_tops[pid] = (gx, gy, top_z + LOCAL_SHIFT[2])
    pole_list.append({'Pole_ID': pid, 'X': gx, 'Y': gy, 'Z': gz, 'Height': round(top_z - bot_z, 2)})
    d_c, h_top = np.linalg.norm(cl[:, :2] - cxy, axis=1), top_z - cl[:, 2]
    cls_cl = np.full(len(cl), 1, dtype=np.uint8) # SHAFT
    cls_cl[(h_top < 1.2) & (d_c > 0.4)] = 2 # CROSSARM
    cls_cl[(h_top > 2.0) & (h_top < 6.0) & (d_c > 0.5)] = 8 # TRANSFORMER
    clean_pts_list.append(cl.astype(np.float64) + LOCAL_SHIFT)
    clean_cls_list.append(cls_cl)
    clean_pid_list.append(np.full(len(cl), pid, dtype=np.int32))
poles_df = pd.DataFrame(pole_list)
clean_pts = np.vstack(clean_pts_list); clean_cls = np.concatenate(clean_cls_list); clean_pid = np.concatenate(clean_pid_list)

# Stage 6.5 — Downguy
print('Running Stage 6.5: Downguy detection...')
n_guys = 0
for pid, cl in enumerate(clean_clusters):
    p_xy, p_topZ, p_botZ = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    mask = (np.sqrt((local[:, 0] - p_xy[0])**2 + (local[:, 1] - p_xy[1])**2) <= 8.0) & \
           (local[:, 2] >= p_botZ - 1.0) & (local[:, 2] <= p_topZ + 0.5) & (hag <= 4.5)
    c_idx = np.where(mask)[0]
    if len(c_idx) < 15: continue
    c_pts, c_hag = local[c_idx], hag[c_idx]
    not_s = np.linalg.norm(c_pts[:, :2] - p_xy, axis=1) > 0.35
    c_pts, c_hag, c_idx = c_pts[not_s], c_hag[not_s], c_idx[not_s]
    if len(c_pts) < 15: continue
    db_s = DBSCAN(eps=0.4, min_samples=5).fit(c_pts)
    for lbl in set(db_s.labels_):
        if lbl == -1: continue
        s_pts, s_hag = c_pts[db_s.labels_ == lbl], c_hag[db_s.labels_ == lbl]
        if len(s_pts) < 15: continue
        lin, pla, _, z_d = extract_signatures(s_pts)
        if lin > 0.65 and pla < 0.30 and 0.25 < z_d < 0.90:
            if s_hag.max() >= 3.5 and s_hag.min() <= 1.5 and (s_hag.max()-s_hag.min()) >= 2.5:
                clean_pts = np.vstack([clean_pts, s_pts.astype(np.float64) + LOCAL_SHIFT])
                clean_cls = np.concatenate([clean_cls, np.full(len(s_pts), 9, dtype=np.uint8)])
                clean_pid = np.concatenate([clean_pid, np.full(len(s_pts), pid, dtype=np.int32)])
                n_guys += 1
print(f'  {n_guys} downguys detected')

# Stage 7-9: Lane split, Tops, Wires
print('Running Stages 7-9: Wires...')
km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(poles_df[['X']].values)
ll = km.labels_; 
if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]: ll = 1 - ll
poles_df['Lane'] = ll; p_lane = {int(r['Pole_ID']): int(r['Lane']) for _, r in poles_df.iterrows()}

for pid, row in poles_df.iterrows():
    idxs = tree_full_2d.query_ball_point(row[['X', 'Y']] - LOCAL_SHIFT[:2], r=1.2)
    top_z = float(np.percentile(local[idxs, 2], 98)) + LOCAL_SHIFT[2] if len(idxs) >= 10 else pole_tops[pid][2]
    pole_tops[pid] = (row['X'], row['Y'], top_z)

def wire_pipeline(tube_res, span, res):
    cg, cp, al, ac = tube_res
    try:
        coef = np.linalg.lstsq(np.stack([np.ones_like(al), al, al*al], axis=1), cp[:, 2], rcond=None)[0]
        kp1 = np.abs(cp[:, 2] - (coef[0] + coef[1]*al + coef[2]*al*al)) < res
    except: return []
    if kp1.sum() < 5 or np.std(ac[kp1]) > 0.35: return []
    lt = cKDTree(cp[kp1]); kk = min(13, kp1.sum())
    _, nbr = lt.query(cp[kp1], k=kk)
    lin, pla = np.zeros(kp1.sum()), np.ones(kp1.sum())
    for k in range(kp1.sum()):
        nbrs = cp[kp1][nbr[k][1:]]
        if len(nbrs) >= 3:
            ev = np.linalg.eigvalsh(np.cov(nbrs.T))
            l1, l2, l3 = sorted(ev, reverse=True)
            if l1 > 1e-10: lin[k], pla[k] = (l1-l2)/l1, (l2-l3)/l1
    kp2 = (lin >= 0.65) & (pla <= 0.25)
    return cg[kp1][kp2] if (kp2.sum() >= 4 and (al[kp1][kp2].max() - al[kp1][kp2].min())/span >= 0.30) else []

wire_asgn, p_ids, p_xys = {}, poles_df['Pole_ID'].values, poles_df[['X', 'Y']].values
for i in range(len(p_xys)):
    for j in range(i + 1, len(p_xys)):
        d = np.linalg.norm(p_xys[i] - p_xys[j])
        if not (5.0 < d < 90.0): continue
        pid_a, pid_b = int(p_ids[i]), int(p_ids[j])
        A_l, B_l = p_xys[i]-LOCAL_SHIFT[:2], p_xys[j]-LOCAL_SHIFT[:2]
        aZ_l, bZ_l = pole_tops[pid_a][2]-LOCAL_SHIFT[2], pole_tops[pid_b][2]-LOCAL_SHIFT[2]
        m3d = np.append((A_l+B_l)/2, (aZ_l+bZ_l)/2)
        u = (B_l-A_l)/d
        cg = np.array(tree_full_3d.query_ball_point(m3d, r=d/2+3.5), dtype=np.int64)
        if not len(cg): continue
        cp = local[cg]; rel = cp[:, :2]-A_l; al, ac = rel@u, rel@np.array([-u[1], u[0]])
        mask = (al>0.3) & (al<d-0.3) & (np.abs(ac)<1.5) & (cp[:, 2]>min(aZ_l, bZ_l)-3.5) & (cp[:, 2]<max(aZ_l, bZ_l)+1.5)
        if mask.sum() < 5: continue
        tube = (cg[mask], cp[mask], al[mask], ac[mask])
        pts = wire_pipeline(tube, d, 0.6)
        if not len(pts): pts = wire_pipeline(tube, d, 1.0)
        if not len(pts): continue
        cls = 71 if p_lane[pid_a] != p_lane[pid_b] else 70
        for g in pts:
            if int(g) not in wire_asgn: wire_asgn[int(g)] = cls
        print(f"  Span {pid_a}-{pid_b}: {len(pts)} pts")

# Stage 10: Output
wg = np.array(list(wire_asgn.keys()), dtype=np.int64)
w_pts, w_cls = local[wg].astype(np.float64) + LOCAL_SHIFT, np.array([wire_asgn[g] for g in wg], dtype=np.uint8)
final_p, final_c = np.vstack([clean_pts, w_pts]), np.concatenate([clean_cls + 64, w_cls])
final_id = np.clip(np.concatenate([clean_pid, np.zeros(len(w_pts), dtype=np.int32)]) + 1, 0, 255).astype(np.uint8)

header = laspy.LasHeader(point_format=raw_pt_fmt, version=raw_version)
header.offsets, header.scales = raw_offsets, raw_scales
out = laspy.LasData(header)
out.x, out.y, out.z = final_p.T
out.classification, out.user_data = final_c, final_id
out.write(OUT_LAS)
print(f'DONE. Result: {OUT_LAS}')
print(f'Total time: {time.time()-start_time:.1f}s')
