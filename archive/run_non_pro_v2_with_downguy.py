import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
import pdal
import json
import os

# ── paths ────────────────────────────────────────────────────────────────────────────
RAW_LAS    = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles\laser_left.las'
OUT_LAS    = 'poles_wires_with_downguy.las'
OUT_QML    = 'poles_wires_with_downguy.qml'
OUT_SPANS  = 'span_report_with_downguy.csv'

# ── global shift (from poles.ipynb — keeps arithmetic stable) ──────────────────────
LOCAL_SHIFT = np.array([356534.91, 4661602.80, 215.93])

# ── load ────────────────────────────────────────────────────────────────────────────
print(f'Loading {RAW_LAS} ...')
if not os.path.exists(RAW_LAS):
    print(f"ERROR: {RAW_LAS} not found.")
    exit(1)

las    = laspy.read(RAW_LAS)
points = np.vstack((las.x, las.y, las.z)).T          # global UTM coords
local  = points - LOCAL_SHIFT                         # shifted local coords
print(f'  {len(local):,} points loaded')
print(f'  Local bbox X [{local[:,0].min():.1f}, {local[:,0].max():.1f}]')
print(f'            Y [{local[:,1].min():.1f}, {local[:,1].max():.1f}]')
print(f'            Z [{local[:,2].min():.1f}, {local[:,2].max():.1f}]')

# ── spatial indices on local coords ───────────────────────────────────────────
# Decimated 2-D tree: fast candidate screening
print('Building decimated 2-D tree...')
decimated   = local[::5]
tree_dec_2d = cKDTree(decimated[:, :2])

# Full 2-D and 3-D trees on ALL points (needed for wire fetching)
print('Building full 2-D tree...')
tree_full_2d = cKDTree(local[:, :2])
print('Building full 3-D tree...')
tree_full_3d = cKDTree(local)           # 3-D — critical for wire Z capture

print('Spatial indices built.')

# Stage 1 — PDAL SMRF: Height Above Ground
pdal_pipeline = [
    {"type": "readers.las",    "filename": RAW_LAS},
    {"type": "filters.smrf"},
    {"type": "filters.hag_nn"},
    {"type": "writers.las",    "filename": "_temp_hag_v2.las", "extra_dims": "all"}
]

print('Running PDAL SMRF + HAG ...')
p = pdal.Pipeline(json.dumps(pdal_pipeline))
p.execute()

las_hag = laspy.read('_temp_hag_v2.las')
hag     = np.array(las_hag.HeightAboveGround)   # same order as `local`
print(f'  HAG range: {hag.min():.1f} – {hag.max():.1f} m')
print(f'  Points with HAG > 4 m: {(hag > 4).sum():,}  (pole/wire candidates)')
print('SMRF complete.')

# Stage 2 — Eigenvalue pre-screen on decimated cloud
EIGEN_RADIUS   = 1.2   # neighbourhood radius (m)
MIN_NEIGHBOURS = 8     # minimum points to compute geometry
MIN_LINEARITY  = 0.75
MAX_PLANARITY  = 0.25

def extract_signatures(pts_3d):
    """Return (linearity, planarity, sphericity, z_dir) for a local neighbourhood."""
    if len(pts_3d) < 5:
        return 0, 0, 0, 0
    pts_c = pts_3d - pts_3d.mean(axis=0)
    cov   = np.cov(pts_c.T) + np.eye(3) * 1e-6
    evals, evecs = np.linalg.eigh(cov)
    idx  = evals.argsort()[::-1]
    L1, L2, L3 = evals[idx]
    linearity  = (L1 - L2) / L1 if L1 > 0 else 0
    planarity  = (L2 - L3) / L1 if L1 > 0 else 0
    sphericity = L3 / L1         if L1 > 0 else 0
    z_dir      = abs(evecs[2, idx[0]])
    return linearity, planarity, sphericity, z_dir

print('Running eigenvalue pre-screen on decimated cloud ...')
pole_cand_idx = []

for i in range(0, len(decimated), 10):   # stride-10 for speed
    nbr_idx  = tree_dec_2d.query_ball_point(decimated[i, :2], r=EIGEN_RADIUS)
    if len(nbr_idx) <= MIN_NEIGHBOURS:
        continue
    nbr_pts  = decimated[nbr_idx]
    lin, pla, _, _ = extract_signatures(nbr_pts)
    if lin > MIN_LINEARITY and pla < MAX_PLANARITY:
        pole_cand_idx.append(i)

candidate_pts = decimated[pole_cand_idx]
print(f'  {len(candidate_pts):,} eigen-candidate points')

# Stage 3 — DBSCAN clustering → pole structures
DBSCAN_EPS      = 2.0
DBSCAN_MIN_SAMP = 10
MIN_HEIGHT      = 4.0   # m  — minimum cluster vertical extent
MAX_WIDTH       = 4.0   # m  — maximum horizontal spread

print('DBSCAN clustering ...')
db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMP).fit(candidate_pts)
labels = db.labels_

initial_clusters = []
for lbl in set(labels):
    if lbl == -1:
        continue
    cl     = candidate_pts[labels == lbl]
    extent = cl.max(axis=0) - cl.min(axis=0)
    h, w   = extent[2], max(extent[0], extent[1])
    if h > MIN_HEIGHT and w < MAX_WIDTH:
        initial_clusters.append(cl)

print(f'  {len(initial_clusters)} clusters pass height/width filter')

# Stage 4 — 3-D expansion: recover full structure + crossarms
EXPAND_XY_RADIUS = 2.5   # m — wide enough for crossarms
EXPAND_Z_BUFFER  = 6.0   # m — look this far above the cluster top
TIGHT_RADIUS     = 1.2   # m — fallback tight cylinder if spread is too large
MAX_SPREAD       = 1.5   # m — std dev threshold for tree rejection at expansion

expanded_clusters = []

for cl in initial_clusters:
    cxy   = cl[:, :2].mean(axis=0)
    max_z = cl[:, 2].max()
    min_z = cl[:, 2].min()

    # Distance filter on the full cloud (vectorised, no tree call)
    dist_sq = ((local[:, 0] - cxy[0])**2 +
               (local[:, 1] - cxy[1])**2)
    mask = (
        (dist_sq < EXPAND_XY_RADIUS**2) &
        (local[:, 2] >= min_z) &
        (local[:, 2] <= max_z + EXPAND_Z_BUFFER)
    )
    full = local[mask]
    if len(full) == 0:
        continue

    spread = np.std(full[:, :2])
    if spread < MAX_SPREAD:
        expanded_clusters.append(full)
    else:
        # Too wide → keep only the tight cylinder to cut away tree branches
        tight = np.linalg.norm(full[:, :2] - cxy, axis=1) < TIGHT_RADIUS
        expanded_clusters.append(full[tight])

print(f'Expansion complete: {len(expanded_clusters)} full structures')

# Stage 5 — Tree rejection using HAG + spread
MIN_MEDIAN_HAG    = 3.0   # m  — median height above ground
MIN_HIGH_FRAC     = 0.30  # fraction of cluster points with HAG > 4 m
MAX_XY_STD        = 1.0   # m  — horizontal spread std (poles are narrow)

# Build a KDTree on the HAG-indexed points (full cloud, same order as `local`)
# We map each expanded cluster point back to its HAG value via nearest-neighbour
# (The full cloud points ARE in `local`, so we can index directly by original index)
# Use the tree to find the global indices of each cluster's points
full_tree_3d_local = cKDTree(local)   # 3-D tree in local coords (reuse if already built)

clean_clusters = []
rejected = 0

for cl in expanded_clusters:
    # Map each cluster point → its index in `local` → its HAG value
    _, idx = full_tree_3d_local.query(cl, k=1)
    cl_hag = hag[idx]

    median_hag = float(np.median(cl_hag))
    high_frac  = float((cl_hag > 4.0).sum()) / len(cl_hag)
    xy_std     = float(np.std(cl[:, :2]))

    if median_hag < MIN_MEDIAN_HAG or high_frac < MIN_HIGH_FRAC or xy_std > MAX_XY_STD:
        rejected += 1
        continue

    clean_clusters.append(cl)

print(f'Tree rejection: kept {len(clean_clusters)} poles, rejected {rejected} tree/noise clusters')

# Stage 6 — Component labelling (SHAFT / CROSSARM / TRANSFORMER)
CLASS_SHAFT       = 1
CLASS_CROSSARM    = 2
CLASS_TRANSFORMER = 8

CLASSES = {1:'SHAFT', 2:'CROSSARM', 3:'PRIMARY', 4:'SECONDARY',
           5:'NEUTRAL', 6:'GUY', 7:'ANCHOR', 8:'TRANSFORMER'}

pole_list   = []
pole_tops   = {}   # pid -> (global_X, global_Y, global_Z_top)
clean_pts_list  = []
clean_cls_list  = []
clean_pid_list  = []

for pid, cl in enumerate(clean_clusters):
    cxy    = cl[:, :2].mean(axis=0)
    top_z  = float(cl[:, 2].max())
    bot_z  = float(cl[:, 2].min())
    height = top_z - bot_z

    # Global coords of pole centre
    gx, gy, gz = float(cxy[0] + LOCAL_SHIFT[0]), \
                 float(cxy[1] + LOCAL_SHIFT[1]), \
                 float(cl[:, 2].mean() + LOCAL_SHIFT[2])
    pole_tops[pid] = (gx, gy, top_z + LOCAL_SHIFT[2])   # global top Z

    pole_list.append({'Pole_ID': pid, 'X': gx, 'Y': gy, 'Z': gz, 'Height': round(height, 2)})

    # Per-point class labels (work in local coords)
    d_centre   = np.linalg.norm(cl[:, :2] - cxy, axis=1)
    h_from_top = top_z - cl[:, 2]

    labels_cl  = np.full(len(cl), CLASS_SHAFT, dtype=np.uint8)
    crossarm   = (h_from_top < 1.2)  & (d_centre > 0.4)
    transformer= (h_from_top > 2.0)  & (h_from_top < 6.0) & (d_centre > 0.5)
    labels_cl[crossarm]    = CLASS_CROSSARM
    labels_cl[transformer] = CLASS_TRANSFORMER

    # Convert back to global coords for LAS output
    clean_pts_list.append(cl + LOCAL_SHIFT)
    clean_cls_list.append(labels_cl)
    clean_pid_list.append(np.full(len(cl), pid, dtype=np.int32))

poles_df  = pd.DataFrame(pole_list)
clean_pts = np.vstack(clean_pts_list)
clean_cls = np.concatenate(clean_cls_list)
clean_pid = np.concatenate(clean_pid_list)

print(f'{len(poles_df)} poles found')
vals, cnts = np.unique(clean_cls, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"{CLASSES.get(int(v),'?'):<12}  {c:>7,}")

# Stage 6.5 — Downguy Detection
CLASS_DOWNGUY = 9
CLASSES[9]    = 'DOWNGUY'

GUY_SEARCH_XY   = 8.0
GUY_MIN_HAG_TOP = 3.5
GUY_MAX_HAG_BOT = 1.5
GUY_MIN_LIN     = 0.65
GUY_MAX_PLAN    = 0.30
GUY_MIN_Z_DIAG  = 0.25
GUY_MAX_Z_DIAG  = 0.90
GUY_MAX_XY_STD  = 0.25
GUY_MIN_HAG_SPAN= 2.5
GUY_DBSCAN_EPS  = 0.4
GUY_DBSCAN_MSMP = 5
GUY_MIN_PTS     = 15
SHAFT_EXCL_R    = 0.35

downguy_pts_list = []
downguy_cls_list = []
downguy_pid_list = []
n_guys_total     = 0

print('Running downguy detection...')

for pid, cl in enumerate(clean_clusters):
    pole_xy   = cl[:, :2].mean(axis=0)
    pole_topZ = float(cl[:, 2].max())
    pole_botZ = float(cl[:, 2].min())

    dist_xy = np.sqrt((local[:, 0] - pole_xy[0])**2 +
                      (local[:, 1] - pole_xy[1])**2)
    candidate_mask = (
        (dist_xy <= GUY_SEARCH_XY)           &
        (local[:, 2] >= pole_botZ - 1.0)     &
        (local[:, 2] <= pole_topZ + 0.5)     &
        (hag <= GUY_MIN_HAG_TOP + 1.0)
    )
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) < GUY_MIN_PTS:
        continue

    cand_pts = local[candidate_idx]
    cand_hag = hag[candidate_idx]

    d_from_axis  = np.linalg.norm(cand_pts[:, :2] - pole_xy, axis=1)
    not_shaft    = d_from_axis > SHAFT_EXCL_R
    cand_pts     = cand_pts[not_shaft]
    cand_hag     = cand_hag[not_shaft]
    candidate_idx = candidate_idx[not_shaft]
    if len(cand_pts) < GUY_MIN_PTS:
        continue

    db_sub    = DBSCAN(eps=GUY_DBSCAN_EPS, min_samples=GUY_DBSCAN_MSMP).fit(cand_pts)
    sub_labels = db_sub.labels_

    for lbl in set(sub_labels):
        if lbl == -1:
            continue
        sub_mask = sub_labels == lbl
        sub_pts  = cand_pts[sub_mask]
        sub_hag  = cand_hag[sub_mask]
        sub_idx  = candidate_idx[sub_mask]
        if len(sub_pts) < GUY_MIN_PTS:
            continue

        pts_c        = sub_pts - sub_pts.mean(axis=0)
        cov          = np.cov(pts_c.T) + np.eye(3) * 1e-6
        evals, evecs = np.linalg.eigh(cov)
        order        = evals.argsort()[::-1]
        L1, L2, L3  = evals[order]
        if L1 < 1e-9:
            continue
        linearity    = (L1 - L2) / L1
        planarity    = (L2 - L3) / L1
        primary_evec = evecs[:, order[0]]
        z_dir        = abs(primary_evec[2])

        if linearity < GUY_MIN_LIN or planarity > GUY_MAX_PLAN or not (GUY_MIN_Z_DIAG < z_dir < GUY_MAX_Z_DIAG):
            continue

        hag_max = float(sub_hag.max())
        hag_min = float(sub_hag.min())
        if hag_max < GUY_MIN_HAG_TOP or hag_min > GUY_MAX_HAG_BOT or (hag_max - hag_min) < GUY_MIN_HAG_SPAN:
            continue

        xy_std = float(np.std(sub_pts[:, :2]))
        if xy_std > GUY_MAX_XY_STD:
            continue

        guy_labels = np.full(len(sub_pts), CLASS_DOWNGUY, dtype=np.uint8)
        downguy_pts_list.append(sub_pts + LOCAL_SHIFT)
        downguy_cls_list.append(guy_labels)
        downguy_pid_list.append(np.full(len(sub_pts), pid, dtype=np.int32))
        n_guys_total += 1

print(f'Downguy detection complete: {n_guys_total} guys found')

if downguy_pts_list:
    clean_pts_list.extend(downguy_pts_list)
    clean_cls_list.extend(downguy_cls_list)
    clean_pid_list.extend(downguy_pid_list)
    clean_pts = np.vstack(clean_pts_list)
    clean_cls = np.concatenate(clean_cls_list)
    clean_pid = np.concatenate(clean_pid_list)

# Stage 7 — KMeans lane split
km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(poles_df[['X']].values)
llabels = km.labels_
if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]:
    llabels = 1 - llabels
poles_df['Lane'] = llabels
pole_lane = {int(r['Pole_ID']): int(r['Lane']) for _, r in poles_df.iterrows()}

# Stage 8 — Robust pole-top Z estimation
POLE_TOP_RADIUS = 1.2
POLE_TOP_PCT    = 98
global_tree_2d = cKDTree(points[:, :2])
for _, row in poles_df.iterrows():
    pid = int(row['Pole_ID'])
    idxs = global_tree_2d.query_ball_point([row['X'], row['Y']], r=POLE_TOP_RADIUS)
    if len(idxs) >= 10:
        top_z = float(np.percentile(points[idxs, 2], POLE_TOP_PCT))
    else:
        top_z = pole_tops[pid][2]
    pole_tops[pid] = (row['X'], row['Y'], top_z)

# Stage 9 — Wire detection
MIN_SPAN        = 5.0
MAX_SPAN        = 90.0
TUBE_RADIUS     = 1.5
Z_BELOW         = 3.5
Z_ABOVE         = 1.5
CAT_RESIDUAL_A  = 0.6
CAT_RESIDUAL_B  = 1.0
MIN_SPAN_COV    = 0.30
WIRE_MIN_LIN    = 0.65
WIRE_MAX_PLA    = 0.25
EIG_K           = 12
MAX_ACROSS_STD  = 0.35

def local_eigenfeatures(pts_local, k=EIG_K):
    n = len(pts_local)
    lin, pla = np.zeros(n), np.ones(n)
    if n < 4: return lin, pla
    lt = cKDTree(pts_local)
    kk = min(k + 1, n)
    _, nbr_idx = lt.query(pts_local, k=kk)
    for i in range(n):
        nbrs = pts_local[nbr_idx[i][1:]]
        if len(nbrs) < 3: continue
        cov = np.cov(nbrs.T)
        evals = np.linalg.eigvalsh(cov)
        l1, l2, l3 = sorted(evals, reverse=True)
        if l1 > 1e-10:
            lin[i] = (l1 - l2) / l1
            pla[i] = (l2 - l3) / l1
    return lin, pla

def project_span(pts, A_local, u):
    rel = pts[:, :2] - A_local
    along = rel @ u
    nv = np.array([-u[1], u[0]])
    across = rel @ nv
    return along, across

def fit_parabola(s, z):
    M = np.stack([np.ones_like(s), s, s * s], axis=1)
    return np.linalg.lstsq(M, z, rcond=None)[0]

def fetch_tube(A_local, B_local, aZ_local, bZ_local, span):
    mid2d, midZ = (A_local + B_local) / 2, (aZ_local + bZ_local) / 2
    mid3d = np.array([mid2d[0], mid2d[1], midZ])
    u = (B_local - A_local) / span
    cand_g = np.array(tree_full_3d.query_ball_point(mid3d, r=span/2 + TUBE_RADIUS + 2.0), dtype=np.int64)
    if not len(cand_g): return None
    cpts = local[cand_g]
    along, across = project_span(cpts, A_local, u)
    mask = (along > 0.3) & (along < span - 0.3) & (np.abs(across) < TUBE_RADIUS) & (cpts[:, 2] > min(aZ_local, bZ_local) - Z_BELOW) & (cpts[:, 2] < max(aZ_local, bZ_local) + Z_ABOVE)
    if mask.sum() < 5: return None
    return cand_g[mask], cpts[mask], along[mask], across[mask]

def wire_pipeline(tube_result, span, cat_residual):
    cand_g, cpts, along, across = tube_result
    try:
        coef = fit_parabola(along, cpts[:, 2])
        z_pred = coef[0] + coef[1]*along + coef[2]*along*along
        keep1 = np.abs(cpts[:, 2] - z_pred) < cat_residual
    except: return np.array([], dtype=np.int64)
    if keep1.sum() < 5 or np.std(across[keep1]) > MAX_ACROSS_STD: return np.array([], dtype=np.int64)
    lin, pla = local_eigenfeatures(cpts[keep1])
    keep2 = (lin >= WIRE_MIN_LIN) & (pla <= WIRE_MAX_PLA)
    if keep2.sum() < 4 or (along[keep1][keep2].max() - along[keep1][keep2].min()) / span < MIN_SPAN_COV: return np.array([], dtype=np.int64)
    try:
        coef2 = fit_parabola(along[keep1][keep2], cpts[keep1][keep2][:, 2])
        z_pred2 = coef2[0] + coef2[1]*along + coef2[2]*along*along
        keep3 = (np.abs(cpts[:, 2] - z_pred2) < cat_residual) & (np.abs(across) < MAX_ACROSS_STD * 2.5)
        keep3[keep1] &= keep2
    except: keep3 = keep1.copy()
    out = cand_g[keep3]
    return out if len(out) >= 4 else np.array([], dtype=np.int64)

def linear_fallback(A_local, B_local, aZ_local, bZ_local, span, u):
    mid2d, midZ = (A_local + B_local) / 2, (aZ_local + bZ_local) / 2
    mid3d = np.array([mid2d[0], mid2d[1], midZ])
    cand_g = np.array(tree_full_3d.query_ball_point(mid3d, r=span/2 + TUBE_RADIUS + 2.0), dtype=np.int64)
    if not len(cand_g): return np.array([], dtype=np.int64)
    cpts = local[cand_g]
    along, across = project_span(cpts, A_local, u)
    line_z = aZ_local + (bZ_local - aZ_local) * (along / span)
    mask = (along > 0.3) & (along < span - 0.3) & (np.abs(across) < TUBE_RADIUS) & (np.abs(cpts[:, 2] - line_z) < 1.5) & (cpts[:, 2] > min(aZ_local, bZ_local) - 3.5)
    if mask.sum() < 5 or np.std(across[mask]) > MAX_ACROSS_STD or (along[mask].max() - along[mask].min()) / span < MIN_SPAN_COV: return np.array([], dtype=np.int64)
    return cand_g[mask]

pole_ids = poles_df['Pole_ID'].values.astype(int)
pole_xys = poles_df[['X', 'Y']].values
pairs = []
for i in range(len(pole_xys)):
    for j in range(i + 1, len(pole_xys)):
        d = float(np.linalg.norm(pole_xys[i] - pole_xys[j]))
        if MIN_SPAN < d < MAX_SPAN: pairs.append((i, j, d))

wire_assignments = {}
span_rows = []
for (i, j, d) in pairs:
    pid_a, pid_b = int(pole_ids[i]), int(pole_ids[j])
    A_loc, B_loc = pole_xys[i] - LOCAL_SHIFT[:2], pole_xys[j] - LOCAL_SHIFT[:2]
    aZ_loc, bZ_loc = pole_tops[pid_a][2] - LOCAL_SHIFT[2], pole_tops[pid_b][2] - LOCAL_SHIFT[2]
    cls = 71 if pole_lane[pid_a] != pole_lane[pid_b] else 70
    tube = fetch_tube(A_loc, B_loc, aZ_loc, bZ_loc, d)
    survivors, used_pass = np.array([], dtype=np.int64), None
    if tube is not None:
        survivors = wire_pipeline(tube, d, CAT_RESIDUAL_A)
        if len(survivors): used_pass = 'A'
    if not len(survivors) and tube is not None:
        survivors = wire_pipeline(tube, d, CAT_RESIDUAL_B)
        if len(survivors): used_pass = 'B'
    if not len(survivors):
        survivors = linear_fallback(A_loc, B_loc, aZ_loc, bZ_loc, d, (B_loc-A_loc)/d)
        if len(survivors): used_pass = 'C'
    if not len(survivors): continue
    for g in survivors:
        if int(g) not in wire_assignments: wire_assignments[int(g)] = cls
    span_rows.append({'pole_a': pid_a, 'pole_b': pid_b, 'span_m': round(d, 1), 'kind': 'CROSS_ROAD' if cls==71 else 'SAME_LANE', 'pass': used_pass, 'points': len(survivors)})

pd.DataFrame(span_rows).to_csv(OUT_SPANS, index=False)

# Stage 10 — Write output
LAS_OFFSET = 64
if wire_assignments:
    wg = np.array(list(wire_assignments.keys()), dtype=np.int64)
    wire_pts, wire_cls = local[wg] + LOCAL_SHIFT, np.array([wire_assignments[g] for g in wg], dtype=np.uint8)
else:
    wire_pts, wire_cls = np.zeros((0, 3)), np.zeros(0, dtype=np.uint8)

final_pts = np.vstack([clean_pts, wire_pts])
final_cls = np.concatenate([clean_cls + LAS_OFFSET, wire_cls])
final_pid = np.clip(np.concatenate([clean_pid, np.zeros(len(wire_pts), dtype=np.int32)]) + 1, 0, 255).astype(np.uint8)

header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
header.offsets, header.scales = las.header.offsets, las.header.scales
out = laspy.LasData(header)
out.x, out.y, out.z = final_pts.T
out.classification, out.user_data = final_cls, final_pid
out.write(OUT_LAS)
print(f'Done. Output written to {OUT_LAS}')
