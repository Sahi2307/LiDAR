import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'e:\SAAHIE\poles-84376429-33333333')  # ensure fake_sklearn can be imported if cwd changes

import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from fake_sklearn import DBSCAN, KMeans

RAW_LAS     = 'laser_left.las'
OUT_LAS     = 'poles_wires.las'
OUT_QML     = 'poles_wires.qml'
OUT_SPANS   = 'span_report.csv'
CACHE_L1    = '_cache_L1_las_hag.npz'      # LAS coords + HAG
CACHE_L2    = '_cache_L2_clusters.npz'     # clean pole clusters

LOCAL_SHIFT = np.array([356534.91, 4661602.80, 215.93])

print('Imports OK. Cache paths:')
print(f'  L1 ({CACHE_L1}): {"EXISTS" if os.path.exists(CACHE_L1) else "not found — will build"}')
print(f'  L2 ({CACHE_L2}): {"EXISTS" if os.path.exists(CACHE_L2) else "not found — will build"}')


# ── LEVEL-1 CACHE — LAS + HAG ─────────────────────────────────────────────────

if os.path.exists(CACHE_L1):
    # ── FAST PATH: load from cache ────────────────────────────────────────────
    print(f'Loading L1 cache from {CACHE_L1} ...')
    _c     = np.load(CACHE_L1)
    points = _c['points']          # global UTM (N,3)
    local  = _c['local']           # shifted local (N,3)
    hag    = _c['hag']             # height above ground (N,)
    # Restore laspy header for Stage 10 LAS write
    las    = laspy.read(RAW_LAS)   # header only — no processing
    print(f'  Loaded {len(local):,} points + HAG from cache (skipped PDAL)')

else:
    # ── SLOW PATH: run full pipeline, then save ───────────────────────────────
    import pdal, json as _json

    print(f'L1 cache not found — reading {RAW_LAS} and running PDAL SMRF ...')
    las    = laspy.read(RAW_LAS)
    points = np.vstack((las.x, las.y, las.z)).T
    local  = points - LOCAL_SHIFT
    print(f'  {len(local):,} points loaded')

    if not os.path.exists('_temp_hag.las'):
        print('    _temp_hag.las not found. Running PDAL...')
        pdal_pipeline = [
            {"type": "readers.las",   "filename": RAW_LAS},
            {"type": "filters.smrf"},
            {"type": "filters.hag_nn"},
            {"type": "writers.las",   "filename": "_temp_hag.las", "extra_dims": "all"}
        ]
        p = pdal.Pipeline(_json.dumps(pdal_pipeline))
        p.execute()
    else:
        print('    Found existing _temp_hag.las, skipping PDAL processing.')
    las_hag = laspy.read('_temp_hag.las')
    hag     = np.array(las_hag.HeightAboveGround)
    print(f'  HAG range: {hag.min():.1f} – {hag.max():.1f} m')

    # Save L1 cache
    np.savez_compressed(CACHE_L1, points=points, local=local, hag=hag)
    print(f'  L1 cache saved → {CACHE_L1}')

print(f'HAG stats: {(hag>4).sum():,} points above 4 m')


# ── REBUILD SPATIAL TREES (always — fast from cached arrays, ~10–30 s) ────────
# cKDTree cannot be reliably serialised across scipy versions, so we rebuild.
# With cached local array this takes seconds vs the hours saved by skipping PDAL.

print('Building spatial indices ...')
decimated    = local[::5]
tree_dec_2d  = cKDTree(decimated[:, :2])
tree_full_2d = cKDTree(local[:, :2])
tree_full_3d = cKDTree(local)
print(f'  decimated={len(decimated):,}  full={len(local):,}  trees ready')


# ── LEVEL-2 CACHE — clean pole clusters ───────────────────────────────────────

if os.path.exists(CACHE_L2):
    # ── FAST PATH ─────────────────────────────────────────────────────────────
    print(f'Loading L2 cache from {CACHE_L2} ...')
    _c2          = np.load(CACHE_L2)
    _pts         = _c2['pts']          # (TotalPts, 3) local coords
    _gidx        = _c2['glob_idx']     # (TotalPts,)   global indices into local
    _pid         = _c2['pid']          # (TotalPts,)   cluster id per point
    n_poles      = int(_pid.max()) + 1
    clean_clusters  = [_pts[_pid == i]  for i in range(n_poles)]
    clean_glob_idx  = [_gidx[_pid == i] for i in range(n_poles)]
    print(f'  Loaded {n_poles} clean clusters ({len(_pts):,} pts) — skipped Stages 2–5')

else:
    # ── SLOW PATH: run Stages 2–5 ─────────────────────────────────────────────
    print('L2 cache not found — running Stages 2–5 ...')

    # ── Stage 2: HAG-filtered eigenvalue pre-screen ───────────────────────────
    EIGEN_RADIUS   = 1.2
    MIN_NEIGHBOURS = 8
    MIN_LINEARITY  = 0.75
    MAX_PLANARITY  = 0.25

    hag_dec   = hag[::5]
    hag_samp  = hag_dec[::10]
    sample_all= decimated[::10]
    sample    = sample_all[hag_samp > 1.5]
    print(f'  Stage 2: {len(sample):,} elevated sample pts')

    nbr_lists     = tree_dec_2d.query_ball_point(sample[:, :2], r=EIGEN_RADIUS)
    pole_cand_idx = []
    for i, nbr_idx in enumerate(nbr_lists):
        if len(nbr_idx) <= MIN_NEIGHBOURS:
            continue
        pts_c = decimated[nbr_idx] - decimated[nbr_idx].mean(axis=0)
        cov   = np.cov(pts_c.T) + np.eye(3) * 1e-6
        ev    = np.linalg.eigvalsh(cov)
        L3, L2, L1 = ev[0], ev[1], ev[2]
        if L1 < 1e-9: continue
        if (L1-L2)/L1 > MIN_LINEARITY and (L2-L3)/L1 < MAX_PLANARITY:
            pole_cand_idx.append(i)
    candidate_pts = sample[pole_cand_idx]
    print(f'  Stage 2: {len(candidate_pts):,} eigen-candidates')

    # ── Stage 3: DBSCAN ───────────────────────────────────────────────────────
    db     = DBSCAN(eps=2.0, min_samples=10).fit(candidate_pts)
    lbs    = db.labels_
    initial_clusters = []
    for lbl in set(lbs):
        if lbl == -1: continue
        cl = candidate_pts[lbs == lbl]
        ext = cl.max(0) - cl.min(0)
        if ext[2] > 4.0 and max(ext[0], ext[1]) < 4.0:
            initial_clusters.append(cl)
    print(f'  Stage 3: {len(initial_clusters)} clusters pass height/width filter')

    # ── Stage 4: 3-D expansion via tree_full_2d ───────────────────────────────
    EXPAND_XY_RADIUS = 2.5
    EXPAND_Z_BUFFER  = 6.0
    TIGHT_RADIUS     = 1.2
    MAX_SPREAD       = 1.5
    expanded_clusters = []
    expanded_glob_idx = []
    for cl in initial_clusters:
        cxy   = cl[:, :2].mean(axis=0)
        max_z = float(cl[:, 2].max())
        min_z = float(cl[:, 2].min())
        raw   = np.array(tree_full_2d.query_ball_point(cxy, r=EXPAND_XY_RADIUS), dtype=np.int64)
        if len(raw) == 0: continue
        zmask = (local[raw, 2] >= min_z) & (local[raw, 2] <= max_z + EXPAND_Z_BUFFER)
        idx   = raw[zmask]
        if len(idx) == 0: continue
        full  = local[idx]
        if np.std(full[:, :2], axis=0).max() < MAX_SPREAD:
            expanded_clusters.append(full); expanded_glob_idx.append(idx)
        else:
            keep = np.linalg.norm(full[:, :2] - cxy, axis=1) < TIGHT_RADIUS
            expanded_clusters.append(full[keep]); expanded_glob_idx.append(idx[keep])
    print(f'  Stage 4: {len(expanded_clusters)} expanded structures')

    # ── Stage 5: Tree rejection (direct HAG index) ────────────────────────────
    MIN_MEDIAN_HAG = 3.0
    MIN_HIGH_FRAC  = 0.30
    MAX_XY_STD     = 1.0
    clean_clusters = []
    clean_glob_idx = []
    rejected       = 0
    for cl, gidx in zip(expanded_clusters, expanded_glob_idx):
        cl_hag = hag[gidx]
        median_hag = np.median(cl_hag)
        high_frac = (cl_hag > 4.0).mean()
        xy_std = np.std(cl[:, :2], axis=0).max()
        if median_hag < MIN_MEDIAN_HAG:
            rejected += 1; print(f'Reject median_hag {median_hag:.2f}'); continue
        if high_frac < MIN_HIGH_FRAC:
            rejected += 1; print(f'Reject high_frac {high_frac:.2f}'); continue
        if xy_std > MAX_XY_STD:
            rejected += 1; print(f'Reject xy_std {xy_std:.2f}'); continue
        clean_clusters.append(cl)
        clean_glob_idx.append(gidx)
    print(f'  Stage 5: kept {len(clean_clusters)} poles, rejected {rejected}')

    # ── Save L2 cache ─────────────────────────────────────────────────────────
    if len(clean_clusters) > 0:
        _pts_all  = np.vstack(clean_clusters)
        _gidx_all = np.concatenate(clean_glob_idx)
        _pid_all  = np.concatenate([np.full(len(c), i, dtype=np.int32)
                                    for i, c in enumerate(clean_clusters)])
        np.savez_compressed(CACHE_L2, pts=_pts_all, glob_idx=_gidx_all, pid=_pid_all)
    else:
        print("WARNING: No clean clusters to save! Creating empty L2 cache.")
        np.savez_compressed(CACHE_L2, pts=np.zeros((0,3)), glob_idx=np.zeros(0, dtype=np.int64), pid=np.zeros(0, dtype=np.int32))
    print(f'  L2 cache saved → {CACHE_L2}  ({len(clean_clusters)} poles)')

print(f'\nReady: {len(clean_clusters)} clean pole clusters')


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
    print(f"  {CLASSES.get(int(v),'?'):<12}  {c:>7,}")

# ── STAGE 6.5 — DOWNGUY DETECTION ─────────────────────────────────────────────
# Source: raw laser_left.las via `local` array (loaded Stage 0).
# DO NOT pull from poles_wires.las or any intermediate file.
#
# Speed: uses tree_full_2d (Stage 0) for candidate pull — no brute-force sqrt.

CLASS_DOWNGUY = 9        # internal → LAS class 73 via +64 in Stage 10
CLASSES[9]    = 'DOWNGUY'

# ── Tuning ────────────────────────────────────────────────────────────────────
GUY_SEARCH_XY    = 8.0   # m  XY search radius from pole axis (covers anchor offset)
GUY_MIN_HAG_TOP  = 3.5   # m  candidate must reach at least this high (near pole top)
GUY_MAX_HAG_BOT  = 1.5   # m  candidate must reach at least this low (near anchor)
GUY_MIN_HAG_SPAN = 2.5   # m  must span this much HAG vertically
GUY_MIN_LIN      = 0.65  # eigenvalue linearity minimum
GUY_MAX_PLAN     = 0.30  # eigenvalue planarity maximum
GUY_MIN_Z_DIAG   = 0.25  # primary eigenvec |Z| min — not horizontal like crossarm
GUY_MAX_Z_DIAG   = 0.90  # primary eigenvec |Z| max — not vertical like shaft
GUY_MAX_XY_STD   = 0.25  # m  XY cross-section std — guys are thin cables
GUY_DBSCAN_EPS   = 0.4   # m  tight eps for sub-clustering thin cables
GUY_DBSCAN_MSMP  = 5     # DBSCAN min_samples
GUY_MIN_PTS      = 15    # minimum points for a valid downguy cluster
SHAFT_EXCL_R     = 0.35  # m  shaft cylinder exclusion radius

# ── Pre-compute global HAG mask — points that could belong to a guy ───────────
# Guys span from HAG≈pole_top down to HAG≈0; exclude pure ground and pure high air
guy_hag_mask = (hag <= GUY_MIN_HAG_TOP + 2.0)   # rough pre-filter; tightened per pole
print(f'HAG pre-filter: {guy_hag_mask.sum():,} / {len(local):,} points in candidate pool')

downguy_pts_list = []
downguy_cls_list = []
downguy_pid_list = []
n_guys_total     = 0

print('Running downguy detection from laser_left.las (local coords)...')

for pid, cl in enumerate(clean_clusters):
    pole_xy   = cl[:, :2].mean(axis=0)    # local XY centroid
    pole_topZ = float(cl[:, 2].max())
    pole_botZ = float(cl[:, 2].min())

    # ── Step 1: Candidate pull via tree_full_2d (fast — no brute sqrt) ────────
    raw_idx = np.array(
        tree_full_2d.query_ball_point(pole_xy, r=GUY_SEARCH_XY),
        dtype=np.int64
    )
    if len(raw_idx) < GUY_MIN_PTS:
        continue

    # ── Step 2: Z + HAG filter (vectorised, no Python loop) ───────────────────
    z_vals   = local[raw_idx, 2]
    hag_vals = hag[raw_idx]

    z_mask = (
        (z_vals  >= pole_botZ - 1.0) &   # at or below base
        (z_vals  <= pole_topZ + 0.5) &   # at or above top
        (hag_vals <= GUY_MIN_HAG_TOP + 1.5)
    )
    raw_idx  = raw_idx[z_mask]
    if len(raw_idx) < GUY_MIN_PTS:
        continue

    cand_pts = local[raw_idx]             # (M, 3)  — local coords from laser_left
    cand_hag = hag[raw_idx]

    # ── Step 3: Exclude shaft cylinder ────────────────────────────────────────
    d_axis   = np.linalg.norm(cand_pts[:, :2] - pole_xy, axis=1)
    keep     = d_axis > SHAFT_EXCL_R
    cand_pts = cand_pts[keep]
    cand_hag = cand_hag[keep]
    raw_idx  = raw_idx[keep]
    if len(cand_pts) < GUY_MIN_PTS:
        continue

    # ── Step 4: DBSCAN sub-cluster — isolates thin cable-like groups ──────────
    db_sub     = DBSCAN(eps=GUY_DBSCAN_EPS, min_samples=GUY_DBSCAN_MSMP).fit(cand_pts)
    sub_labels = db_sub.labels_

    for lbl in set(sub_labels):
        if lbl == -1:
            continue

        m        = sub_labels == lbl
        sub_pts  = cand_pts[m]
        sub_hag  = cand_hag[m]

        if len(sub_pts) < GUY_MIN_PTS:
            continue

        # ── Step 5: HAG span gate (cheap — no eigenvec yet) ───────────────────
        hag_max  = float(sub_hag.max())
        hag_min  = float(sub_hag.min())
        if hag_max < GUY_MIN_HAG_TOP:
            continue
        if hag_min > GUY_MAX_HAG_BOT:
            continue
        if hag_max - hag_min < GUY_MIN_HAG_SPAN:
            continue

        # ── Step 6: Eigenvalue geometry (one call per sub-cluster) ────────────
        pts_c        = sub_pts - sub_pts.mean(axis=0)
        cov          = np.cov(pts_c.T) + np.eye(3) * 1e-6
        evals, evecs = np.linalg.eigh(cov)           # ascending
        L3, L2, L1  = evals[0], evals[1], evals[2]

        if L1 < 1e-9:
            continue

        linearity    = (L1 - L2) / L1
        planarity    = (L2 - L3) / L1
        # Primary eigenvec = direction of max variance = cable axis
        primary_evec = evecs[:, 2]                   # eigh: col 2 = largest eval
        z_dir        = abs(primary_evec[2])          # 1=vertical, 0=horizontal

        if linearity < GUY_MIN_LIN:
            continue
        if planarity > GUY_MAX_PLAN:
            continue
        if not (GUY_MIN_Z_DIAG < z_dir < GUY_MAX_Z_DIAG):
            continue   # catches shaft (z≈1) and crossarm/wire (z≈0)

        # ── Step 7: Cross-section thinness ────────────────────────────────────
        if float(np.std(sub_pts[:, :2])) > GUY_MAX_XY_STD:
            continue

        # ── PASSED ────────────────────────────────────────────────────────────
        guy_labels = np.full(len(sub_pts), CLASS_DOWNGUY, dtype=np.uint8)
        downguy_pts_list.append(sub_pts + LOCAL_SHIFT)   # local → global for LAS
        downguy_cls_list.append(guy_labels)
        downguy_pid_list.append(np.full(len(sub_pts), pid, dtype=np.int32))
        n_guys_total += 1

        print(f'  Pole {pid:3d}: DOWNGUY  pts={len(sub_pts):4d}  '
              f'lin={linearity:.2f}  z_dir={z_dir:.2f}  '
              f'HAG {hag_min:.1f}→{hag_max:.1f} m')

print(f'\nDownguy detection complete: {n_guys_total} guys across {len(clean_clusters)} poles')

# ── Merge into master lists for Stage 10 ──────────────────────────────────────
if downguy_pts_list:
    clean_pts_list.extend(downguy_pts_list)
    clean_cls_list.extend(downguy_cls_list)
    clean_pid_list.extend(downguy_pid_list)
    # Restack — Stage 10 reads clean_pts / clean_cls / clean_pid
    clean_pts = np.vstack(clean_pts_list)
    clean_cls = np.concatenate(clean_cls_list)
    clean_pid = np.concatenate(clean_pid_list)
    print(f'  → {sum(len(x) for x in downguy_pts_list):,} downguy pts added')
    print(f'  → Total component points: {len(clean_pts):,}')
else:
    print('  No downguys detected. Tuning guide:')
    print('    Missing short guys : lower GUY_MIN_PTS→8, GUY_MIN_HAG_SPAN→1.5')
    print('    Crossarms caught   : raise GUY_MIN_Z_DIAG→0.35')
    print('    Tree branches in   : lower GUY_MAX_XY_STD→0.18, raise GUY_MIN_LIN→0.72')
    print('    Shaft base caught  : raise SHAFT_EXCL_R→0.5')


km     = KMeans(n_clusters=2, random_state=0, n_init=10).fit(poles_df[['X']].values)
llabels = km.labels_
if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]:
    llabels = 1 - llabels
poles_df['Lane'] = llabels

print(f"Lane split at X ≈ {km.cluster_centers_.mean():.1f}")
print(f"  West lane (0): {(llabels==0).sum()} poles")
print(f"  East lane (1): {(llabels==1).sum()} poles")

pole_lane = {int(r['Pole_ID']): int(r['Lane']) for _, r in poles_df.iterrows()}

# ── STAGE 8 — Robust pole-top Z estimation (reuses tree_full_2d) ─────────────
# Before: rebuilt cKDTree(points[:, :2]) — wasted 10–30s on 10M point cloud.
# After:  convert pole XY to local coords → query tree_full_2d directly.

POLE_TOP_RADIUS = 1.2
POLE_TOP_PCT    = 98

for _, row in poles_df.iterrows():
    pid    = int(row['Pole_ID'])
    # Convert global pole centre → local coords for tree_full_2d
    px_loc = float(row['X']) - LOCAL_SHIFT[0]
    py_loc = float(row['Y']) - LOCAL_SHIFT[1]

    idxs = tree_full_2d.query_ball_point([px_loc, py_loc], r=POLE_TOP_RADIUS)
    if len(idxs) >= 10:
        # local Z → global Z
        top_z = float(np.percentile(local[idxs, 2], POLE_TOP_PCT)) + LOCAL_SHIFT[2]
    else:
        top_z = pole_tops[pid][2]   # fallback

    pole_tops[pid] = (float(row['X']), float(row['Y']), top_z)

top_zs = [t[2] for t in pole_tops.values()]
print(f'Pole tops — min={min(top_zs):.1f} median={float(np.median(top_zs)):.1f} max={max(top_zs):.1f} m (global Z)')


# ── Wire detection tuning ─────────────────────────────────────────────────────
MIN_SPAN        = 5.0    # m  minimum inter-pole distance to consider
MAX_SPAN        = 90.0   # m  maximum inter-pole distance

# Tube
TUBE_RADIUS     = 1.5    # m  half-width of initial 3-D search tube
Z_BELOW         = 3.5    # m  search below the lower pole top
Z_ABOVE         = 1.5    # m  search above the higher pole top

# Catenary
CAT_RESIDUAL_A  = 0.6    # m  strict pass
CAT_RESIDUAL_B  = 1.0    # m  loose pass
MIN_SPAN_COV    = 0.30   # fraction of span that must be covered

# Eigenvalue (linearity / planarity)
WIRE_MIN_LIN    = 0.65
WIRE_MAX_PLA    = 0.25
EIG_K           = 12     # k-nearest neighbours for local covariance

# Across-span spread
MAX_ACROSS_STD  = 0.35
# ─────────────────────────────────────────────────────────────────────────────


def local_eigenfeatures(pts_local, k=EIG_K):
    """Per-point linearity and planarity — fully vectorised (no Python per-point loop).
    Speedup: ~20-50x over the original loop version on typical wire segments.
    """
    n  = len(pts_local)
    if n < 4:
        return np.zeros(n, np.float32), np.ones(n, np.float32)

    lt  = cKDTree(pts_local)
    kk  = min(k + 1, n)

    # (n, kk) indices — one batch query
    _, nbr_idx = lt.query(pts_local, k=kk)

    # Build neighbour tensor (n, k, 3) — excludes self (col 0)
    nbrs = pts_local[nbr_idx[:, 1:]]          # (n, k, 3)

    # Vectorised covariance: centre then einsum
    mu      = nbrs.mean(axis=1, keepdims=True) # (n, 1, 3)
    c       = nbrs - mu                        # (n, k, 3)
    # covs[i] = c[i].T @ c[i] / (k-1)
    covs    = np.einsum('nki,nkj->nij', c, c) / max(kk - 2, 1)  # (n, 3, 3)

    # Batch eigenvalues — eigvalsh returns ascending order
    evals   = np.linalg.eigvalsh(covs)        # (n, 3)  L3 ≤ L2 ≤ L1
    L1, L2, L3 = evals[:, 2], evals[:, 1], evals[:, 0]

    safe = L1 > 1e-10
    lin  = np.where(safe, (L1 - L2) / np.where(safe, L1, 1), 0).astype(np.float32)
    pla  = np.where(safe, (L2 - L3) / np.where(safe, L1, 1), 1).astype(np.float32)
    return lin, pla


def project_span(pts, A_local, u):
    """Project pts (local 3-D) onto the span axis and perpendicular axis."""
    rel    = pts[:, :2] - A_local
    along  = rel @ u
    nv     = np.array([-u[1], u[0]])
    across = rel @ nv
    return along, across


def fit_parabola(s, z):
    M = np.stack([np.ones_like(s), s, s * s], axis=1)
    return np.linalg.lstsq(M, z, rcond=None)[0]


def fetch_tube(A_local, B_local, aZ_local, bZ_local, span):
    """
    Fetch candidate wire points using sample-along-span 2D queries.
    BEFORE: single 3D sphere of radius span/2+3.5 — for a 90m span that is
            a 49m sphere that pulls millions of points.
    AFTER:  N_SAMP small 2D queries (r=TUBE_RADIUS+0.5) along the span axis,
            then Z filter. Each query pulls ~30-80 pts → total ~500 pts max.
    """
    u = (B_local - A_local) / span

    # Sample points along the span (avoid endpoints where poles are)
    N_SAMP   = max(5, int(span / 8))
    ts       = np.linspace(0.05, 0.95, N_SAMP)
    samp_2d  = A_local[np.newaxis] + np.outer(ts, B_local - A_local)  # (N,2)

    # Batch 2D query — all samples in one call
    raw_lists = tree_full_2d.query_ball_point(samp_2d, r=TUBE_RADIUS + 0.5)

    # Union all results (avoids giant sphere entirely)
    parts = [np.array(x, dtype=np.int64) for x in raw_lists if len(x)]
    if not parts:
        return None
    cand_g = np.unique(np.concatenate(parts))

    cpts          = local[cand_g]
    along, across = project_span(cpts, A_local, u)

    low, high = min(aZ_local, bZ_local), max(aZ_local, bZ_local)
    mask = (
        (along  >  0.3) & (along < span - 0.3) &
        (np.abs(across) < TUBE_RADIUS) &
        (cpts[:, 2] > low  - Z_BELOW) &
        (cpts[:, 2] < high + Z_ABOVE)
    )
    if mask.sum() < 5:
        return None
    return cand_g[mask], cpts[mask], along[mask], across[mask]


def wire_pipeline(tube_result, span, cat_residual):
    """
    Apply catenary → spread → eigenvalue → coverage → re-expand.
    Returns global index array of wire points (into `local`), or empty.
    """
    cand_g, cpts, along, across = tube_result

    # 1. Catenary residual
    try:
        coef   = fit_parabola(along, cpts[:, 2])
        z_pred = coef[0] + coef[1]*along + coef[2]*along*along
        keep1  = np.abs(cpts[:, 2] - z_pred) < cat_residual
    except Exception:
        return np.array([], dtype=np.int64)
    if keep1.sum() < 5:
        return np.array([], dtype=np.int64)

    # 2. Across-span spread
    if np.std(across[keep1]) > MAX_ACROSS_STD:
        return np.array([], dtype=np.int64)

    # 3. Eigenvalue linearity + planarity
    lin, pla = local_eigenfeatures(cpts[keep1])
    keep2    = (lin >= WIRE_MIN_LIN) & (pla <= WIRE_MAX_PLA)
    if keep2.sum() < 4:
        return np.array([], dtype=np.int64)

    # 4. Span coverage
    al_ok = along[keep1][keep2]
    if (al_ok.max() - al_ok.min()) / span < MIN_SPAN_COV:
        return np.array([], dtype=np.int64)

    # 5. Re-expand using cleaner parabola
    try:
        coef2   = fit_parabola(al_ok, cpts[keep1][keep2][:, 2])
        z_pred2 = coef2[0] + coef2[1]*along + coef2[2]*along*along
        keep3   = (np.abs(cpts[:, 2] - z_pred2) < cat_residual) & \
                  (np.abs(across) < MAX_ACROSS_STD * 2.5)
    except Exception:
        keep3 = keep1.copy()
        keep3[keep1] &= keep2

    out = cand_g[keep3]
    return out if len(out) >= 4 else np.array([], dtype=np.int64)


def linear_fallback(A_local, B_local, aZ_local, bZ_local, span, u):
    """Pass C — same sample-along-span approach, linear Z model instead of catenary."""
    N_SAMP   = max(5, int(span / 8))
    ts       = np.linspace(0.05, 0.95, N_SAMP)
    samp_2d  = A_local[np.newaxis] + np.outer(ts, B_local - A_local)
    raw_lists = tree_full_2d.query_ball_point(samp_2d, r=TUBE_RADIUS + 0.5)
    parts = [np.array(x, dtype=np.int64) for x in raw_lists if len(x)]
    if not parts:
        return np.array([], dtype=np.int64)
    cand_g = np.unique(np.concatenate(parts))

    cpts          = local[cand_g]
    along, across = project_span(cpts, A_local, u)
    line_z = aZ_local + (bZ_local - aZ_local) * (along / span)
    mask = (
        (along  > 0.3) & (along < span - 0.3) &
        (np.abs(across) < TUBE_RADIUS) &
        (np.abs(cpts[:, 2] - line_z) < 1.5) &
        (cpts[:, 2] > min(aZ_local, bZ_local) - 3.5)
    )
    if mask.sum() < 5:
        return np.array([], dtype=np.int64)
    if np.std(across[mask]) > MAX_ACROSS_STD:
        return np.array([], dtype=np.int64)
    al = along[mask]
    if (al.max() - al.min()) / span < MIN_SPAN_COV:
        return np.array([], dtype=np.int64)
    return cand_g[mask]


print('Wire detection helpers ready.')

# ── Run detection across all pole pairs ──────────────────────────────────────
pole_ids = poles_df['Pole_ID'].values.astype(int)
pole_xys = poles_df[['X', 'Y']].values    # GLOBAL XY

# Build pairs
pairs = []
for i in range(len(pole_xys)):
    for j in range(i + 1, len(pole_xys)):
        d = float(np.linalg.norm(pole_xys[i] - pole_xys[j]))
        if MIN_SPAN < d < MAX_SPAN:
            pairs.append((i, j, d))
print(f'Candidate pairs in [{MIN_SPAN}, {MAX_SPAN}] m: {len(pairs)}')

wire_assignments = {}   # index-into-`local` → wire LAS class (70 or 71)
span_rows        = []

for (i, j, d) in pairs:
    pid_a, pid_b = int(pole_ids[i]), int(pole_ids[j])

    # Global XY pole positions
    Ag, Bg = pole_xys[i], pole_xys[j]

    # Convert to LOCAL for tree queries
    A_loc = Ag - LOCAL_SHIFT[:2]
    B_loc = Bg - LOCAL_SHIFT[:2]

    # Global top Z → local top Z
    aZ_loc = pole_tops[pid_a][2] - LOCAL_SHIFT[2]
    bZ_loc = pole_tops[pid_b][2] - LOCAL_SHIFT[2]

    vec = B_loc - A_loc
    u   = vec / d

    cross = pole_lane[pid_a] != pole_lane[pid_b]
    cls   = 71 if cross else 70
    kind  = 'CROSS_ROAD' if cross else 'SAME_LANE'

    # Pass A — strict catenary
    tube = fetch_tube(A_loc, B_loc, aZ_loc, bZ_loc, d)
    survivors = np.array([], dtype=np.int64)
    used_pass = None

    if tube is not None:
        survivors = wire_pipeline(tube, d, CAT_RESIDUAL_A)
        if len(survivors): used_pass = 'A'

    # Pass B — loose catenary
    if not len(survivors) and tube is not None:
        survivors = wire_pipeline(tube, d, CAT_RESIDUAL_B)
        if len(survivors): used_pass = 'B'

    # Pass C — linear fallback
    if not len(survivors):
        survivors = linear_fallback(A_loc, B_loc, aZ_loc, bZ_loc, d, u)
        if len(survivors): used_pass = 'C'

    if not len(survivors):
        continue

    # Vectorised: only assign points not yet claimed by another span
    existing = np.array(list(wire_assignments.keys()), dtype=np.int64) if wire_assignments else np.array([], dtype=np.int64)
    new_mask = ~np.isin(survivors, existing)
    wire_assignments.update({int(g): cls for g in survivors[new_mask]})

    span_rows.append({'pole_a': pid_a, 'pole_b': pid_b,
                      'span_m': round(d, 1), 'kind': kind,
                      'pass': used_pass, 'points': len(survivors)})
    print(f'  {kind:12s}  {pid_a:3d}↔{pid_b:3d}  {d:5.1f} m  pass={used_pass}  → {len(survivors):,} pts')

spans_df = pd.DataFrame(span_rows).sort_values(['kind', 'pole_a', 'pole_b'])
spans_df.to_csv(OUT_SPANS, index=False)

print(f'\nSpans detected: {len(spans_df)}')
if len(spans_df):
    for k in ['SAME_LANE', 'CROSS_ROAD']:
        n = (spans_df['kind'] == k).sum()
        if n: print(f"  {k}: {n}")
    for p in ['A','B','C']:
        n = (spans_df['pass'] == p).sum()
        if n: print(f"  Pass {p}: {n} spans")
print(f'Total unique wire points: {len(wire_assignments):,}')
print(f'{OUT_SPANS} written')

LAS_OFFSET = 64   # internal classes 1–9 → LAS classes 65–73

# Wire points: LOCAL coords → GLOBAL
if wire_assignments:
    wg       = np.array(list(wire_assignments.keys()), dtype=np.int64)
    wire_pts = local[wg] + LOCAL_SHIFT
    wire_cls = np.array([wire_assignments[int(g)] for g in wg], dtype=np.uint8)
else:
    wire_pts = np.zeros((0, 3))
    wire_cls = np.zeros(0, dtype=np.uint8)

# Component points already in global coords (shaft + crossarm + transformer + downguy)
final_pts = np.vstack([clean_pts, wire_pts])
final_cls = np.concatenate([
    clean_cls + LAS_OFFSET,    # 65=SHAFT 66=CROSSARM 72=TRANSFORMER 73=DOWNGUY
    wire_cls                   # 70=SAME_LANE 71=CROSS_ROAD (absolute already)
])
final_pid = np.clip(np.concatenate([
    clean_pid,
    np.zeros(len(wire_pts), dtype=np.int32)
]) + 1, 0, 255).astype(np.uint8)

print(f'Writing {len(final_pts):,} total points')
CLASS_PRINT = {
    65:'SHAFT', 66:'CROSSARM', 70:'SAME_LANE_WIRE',
    71:'CROSS_ROAD_WIRE', 72:'TRANSFORMER', 73:'DOWNGUY'
}
for cls_val, name in sorted(CLASS_PRINT.items()):
    n = (final_cls == cls_val).sum()
    if n:
        print(f'  {cls_val:>2}  {name:<18}  {n:>8,}')

header         = laspy.LasHeader(point_format=las.header.point_format,
                                 version=las.header.version)
header.offsets = las.header.offsets
header.scales  = las.header.scales
out            = laspy.LasData(header)
out.x, out.y, out.z = final_pts.T
out.classification   = final_cls
out.user_data        = final_pid
out.write(OUT_LAS)
print(f'{OUT_LAS} written')

# ── QML ───────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    65: (220, 80,  80,  'SHAFT'),
    66: (255, 200, 0,   'CROSSARM'),
    70: (60,  130, 220, 'SAME_LANE_WIRE'),
    71: (200, 50,  50,  'CROSS_ROAD_WIRE'),
    72: (120, 60,  200, 'TRANSFORMER'),
    73: (255, 0,   255, 'DOWNGUY'),
}
cats = ['    <category value="0" label="Unclassified" color="136,136,136,255" render="true"/>']
for cid, (r, g, b, name) in sorted(CLASS_COLORS.items()):
    cats.append(f'    <category value="{cid}" label="{name}" color="{r},{g},{b},255" render="true"/>')

qml = (
    "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
    '<qgis styleCategories="Symbology" version="3.34">\n'
    '  <renderer-v2 type="pointcloudclassified" attribute="Classification">\n'
    '    <categories>\n'
    + '\n'.join(cats) + '\n'
    '    </categories>\n'
    '  </renderer-v2>\n'
    '</qgis>\n'
)
with open(OUT_QML, 'w') as f:
    f.write(qml)
print(f'{OUT_QML} written')
print('\n✅ Done. QGIS → Layer Properties → Symbology → Load Style → poles_wires.qml')

