print('Script started!')
import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
import os
import pdal
import json
import time
import gc

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_LAS    = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\laser_left.las'
OUT_LAS    = 'poles_wires_v3.las'
OUT_QML    = 'poles_wires_v3.qml'
OUT_SPANS  = 'span_report_v3.csv'
TEMP_HAG   = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\_temp_hag.las'

# ── global shift ──────────────────
LOCAL_SHIFT = np.array([356534.91, 4661602.80, 215.93])

# ── load ───────────────────────────────────────────────────────────────────────
print(f'Loading {RAW_LAS} ...')
start_time = time.time()
las    = laspy.read(RAW_LAS)
points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
local  = (points - LOCAL_SHIFT).astype(np.float32)
print(f'  {len(local):,} points loaded in {time.time()-start_time:.1f}s')
del points
gc.collect()

# ── spatial indices ───────────────────────────────────────────
print('Building spatial indices...')
t0 = time.time()
decimated   = local[::5]
tree_dec_2d = cKDTree(decimated[:, :2])
tree_full_2d = cKDTree(local[:, :2])
tree_full_3d = cKDTree(local)
print(f'  Spatial indices built in {time.time()-t0:.1f}s')

# ── Stage 1: HAG ─────────────────────────────────────────────
print('Running Stage 1: PDAL SMRF + HAG ...')
if os.path.exists(TEMP_HAG):
    print(f'  Found existing {TEMP_HAG}, reusing it...')
    las_hag = laspy.read(TEMP_HAG)
else:
    pdal_pipeline = [
        {"type": "readers.las",    "filename": RAW_LAS},
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn"},
        {"type": "writers.las",    "filename": "_temp_hag_new.las", "extra_dims": "all"}
    ]
    p = pdal.Pipeline(json.dumps(pdal_pipeline))
    p.execute()
    las_hag = laspy.read("_temp_hag_new.las")

hag = np.array(las_hag.HeightAboveGround).astype(np.float32)
del las_hag
gc.collect()
print(f'  SMRF complete.')

# ── STAGE 2 — Eigenvalue pre-screen (batch-vectorised) ────────────────────────
print('Running Stage 2: Eigenvalue pre-screen...')
t2 = time.time()
EIGEN_RADIUS, MIN_NEIGHBOURS, MIN_LINEARITY, MAX_PLANARITY = 1.2, 8, 0.75, 0.25
sample = decimated[::10]
nbr_lists = tree_dec_2d.query_ball_point(sample[:, :2], r=EIGEN_RADIUS)
pole_cand_idx = []
for i, nbr_idx in enumerate(nbr_lists):
    if len(nbr_idx) <= MIN_NEIGHBOURS: continue
    nbr_pts = decimated[nbr_idx]
    pts_c = nbr_pts - nbr_pts.mean(axis=0)
    cov = np.cov(pts_c.T) + np.eye(3) * 1e-6
    evals = np.linalg.eigvalsh(cov)
    L3, L2, L1 = evals[0], evals[1], evals[2]
    if L1 < 1e-9: continue
    if (L1 - L2) / L1 > MIN_LINEARITY and (L2 - L3) / L1 < MAX_PLANARITY:
        pole_cand_idx.append(i)
candidate_pts = sample[pole_cand_idx]
print(f'  {len(candidate_pts):,} candidates in {time.time()-t2:.1f}s')

# Stage 3 — DBSCAN
print('Running Stage 3: DBSCAN...')
db = DBSCAN(eps=2.0, min_samples=10).fit(candidate_pts)
initial_clusters = []
for lbl in set(db.labels_):
    if lbl == -1: continue
    cl = candidate_pts[db.labels_ == lbl]
    extent = cl.max(axis=0) - cl.min(axis=0)
    if extent[2] > 4.0 and max(extent[0], extent[1]) < 4.0:
        initial_clusters.append(cl)
print(f'  {len(initial_clusters)} clusters pass filter')

# Stage 4 — Expansion
print('Running Stage 4: 3-D Expansion...')
expanded_clusters = []
for cl in initial_clusters:
    cxy, max_z, min_z = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    dist_sq = ((local[:, 0] - cxy[0])**2 + (local[:, 1] - cxy[1])**2)
    mask = (dist_sq < 2.5**2) & (local[:, 2] >= min_z) & (local[:, 2] <= max_z + 6.0)
    full = local[mask]
    if len(full) == 0: continue
    if np.std(full[:, :2]) < 1.5: expanded_clusters.append(full)
    else:
        tight = np.linalg.norm(full[:, :2] - cxy, axis=1) < 1.2
        expanded_clusters.append(full[tight])

# Stage 5 — Tree rejection
print('Running Stage 5: Tree rejection...')
clean_clusters = []
for cl in expanded_clusters:
    _, idx = tree_full_3d.query(cl, k=1)
    cl_hag = hag[idx]
    if np.median(cl_hag) >= 3.0 and (cl_hag > 4.0).sum()/len(cl_hag) >= 0.30 and np.std(cl[:, :2]) <= 1.0:
        clean_clusters.append(cl)
print(f'  Kept {len(clean_clusters)} poles')

# Stage 6 — Labelling
print('Running Stage 6: Labelling...')
pole_list, pole_tops, clean_pts_list, clean_cls_list, clean_pid_list = [], {}, [], [], []
for pid, cl in enumerate(clean_clusters):
    cxy, top_z, bot_z = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    gx, gy, gz = cxy[0]+LOCAL_SHIFT[0], cxy[1]+LOCAL_SHIFT[1], cl[:, 2].mean()+LOCAL_SHIFT[2]
    pole_tops[pid] = (gx, gy, top_z + LOCAL_SHIFT[2])
    pole_list.append({'Pole_ID': pid, 'X': gx, 'Y': gy, 'Z': gz, 'Height': round(top_z-bot_z, 2)})
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
print('Running Stage 6.5: Downguy...')
n_gt = 0
for pid, cl in enumerate(clean_clusters):
    p_xy, p_tZ, p_bZ = cl[:, :2].mean(axis=0), cl[:, 2].max(), cl[:, 2].min()
    raw_idx = np.array(tree_full_2d.query_ball_point(p_xy, r=8.0), dtype=np.int64)
    if len(raw_idx) < 15: continue
    m = (local[raw_idx, 2] >= p_bZ-1.0) & (local[raw_idx, 2] <= p_tZ+0.5) & (hag[raw_idx] <= 5.0)
    raw_idx = raw_idx[m]
    if len(raw_idx) < 15: continue
    c_pts = local[raw_idx]; d_a = np.linalg.norm(c_pts[:, :2]-p_xy, axis=1)
    raw_idx = raw_idx[d_a > 0.35]; c_pts = local[raw_idx]; c_hag = hag[raw_idx]
    if len(c_pts) < 15: continue
    db_s = DBSCAN(eps=0.4, min_samples=5).fit(c_pts)
    for lbl in set(db_s.labels_):
        if lbl == -1: continue
        s_pts, s_hag = c_pts[db_s.labels_==lbl], c_hag[db_s.labels_==lbl]
        if len(s_pts) < 15: continue
        h_max, h_min = float(s_hag.max()), float(s_hag.min())
        if h_max >= 3.5 and h_min <= 1.5 and h_max-h_min >= 2.5:
            pts_c = s_pts - s_pts.mean(axis=0); cov = np.cov(pts_c.T) + np.eye(3)*1e-6
            ev = np.linalg.eigvalsh(cov); l3, l2, l1 = ev; lin, pla = (l1-l2)/l1, (l2-l3)/l1
            if lin > 0.65 and pla < 0.30:
                clean_pts_list.append(s_pts.astype(np.float64) + LOCAL_SHIFT)
                clean_cls_list.append(np.full(len(s_pts), 9, dtype=np.uint8))
                clean_pid_list.append(np.full(len(s_pts), pid, dtype=np.int32))
                n_gt += 1
clean_pts = np.vstack(clean_pts_list); clean_cls = np.concatenate(clean_cls_list); clean_pid = np.concatenate(clean_pid_list)
print(f'  {n_gt} downguys found')

# Stage 7-9: Wires (Vectorized logic)
print('Running Stages 7-9: Wires...')
km   = KMeans(n_clusters=2, random_state=0, n_init=10).fit(poles_df[['X']].values)
ll = km.labels_; 
if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]: ll = 1 - ll
poles_df['Lane'] = ll; p_lane = {int(r['Pole_ID']): int(r['Lane']) for _, r in poles_df.iterrows()}

for pid, row in poles_df.iterrows():
    idxs = tree_full_2d.query_ball_point(row[['X', 'Y']] - LOCAL_SHIFT[:2], r=1.2)
    top_z = float(np.percentile(local[idxs, 2], 98)) + LOCAL_SHIFT[2] if len(idxs) >= 10 else pole_tops[pid][2]
    pole_tops[pid] = (row['X'], row['Y'], top_z)

def wire_pipeline_v3(tube_res, span, res):
    cg, cp, al, ac = tube_res
    try:
        coef = np.linalg.lstsq(np.stack([np.ones_like(al), al, al*al], axis=1), cp[:, 2], rcond=None)[0]
        kp1 = np.abs(cp[:, 2] - (coef[0] + coef[1]*al + coef[2]*al*al)) < res
    except: return []
    if kp1.sum() < 5 or np.std(ac[kp1]) > 0.35: return []
    # Vectorized eigenfeatures
    pts = cp[kp1]; n = len(pts); lt = cKDTree(pts); kk = min(13, n)
    _, nbr = lt.query(pts, k=kk); nbrs = pts[nbr[:,1:]]
    mu = nbrs.mean(axis=1, keepdims=True); c = nbrs - mu
    covs = np.einsum('nki,nkj->nij', c, c) / max(kk-2, 1)
    ev = np.linalg.eigvalsh(covs); L1, L2, L3 = ev[:,2], ev[:,1], ev[:,0]
    lin = (L1-L2)/L1; pla = (L2-L3)/L1
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
        m3d = np.append((A_l+B_l)/2, (aZ_l+bZ_l)/2); u = (B_l-A_l)/d
        cg = np.array(tree_full_3d.query_ball_point(m3d, r=d/2+3.5), dtype=np.int64)
        if not len(cg): continue
        cp = local[cg]; rel = cp[:, :2]-A_l; al, ac = rel@u, rel@np.array([-u[1], u[0]])
        mask = (al>0.3) & (al<d-0.3) & (np.abs(ac)<1.5) & (cp[:, 2]>min(aZ_l, bZ_l)-3.5) & (cp[:, 2]<max(aZ_l, bZ_l)+1.5)
        if mask.sum() < 5: continue
        tube = (cg[mask], cp[mask], al[mask], ac[mask])
        pts = wire_pipeline_v3(tube, d, 0.6)
        if not len(pts): pts = wire_pipeline_v3(tube, d, 1.0)
        if not len(pts): continue
        cls = 71 if p_lane[pid_a] != p_lane[pid_b] else 70
        for g in pts:
            if int(g) not in wire_asgn: wire_asgn[int(g)] = cls
        print(f"  Span {pid_a}-{pid_b}: {len(pts)} pts")

# Stage 10: Output
wg = np.array(list(wire_asgn.keys()), dtype=np.int64)
w_pts, w_cls = local[wg].astype(np.float64) + LOCAL_SHIFT, np.array([wire_asgn[g] for g in wg], dtype=np.uint8)
final_p, final_c = np.vstack([clean_pts, w_pts]), np.concatenate([clean_cls + 64, w_cls])
final_id = np.clip(np.concatenate([clean_pid, np.zeros(len(w_pts), dtype=np.int32)])+1, 0, 255).astype(np.uint8)

header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
header.offsets, header.scales = las.header.offsets, las.header.scales
out = laspy.LasData(header)
out.x, out.y, out.z = final_p.T; out.classification, out.user_data = final_c, final_id
out.write(OUT_LAS)
print(f'DONE. Result: {OUT_LAS} and {OUT_QML}'); print(f'Total time: {time.time()-start_time:.1f}s')
