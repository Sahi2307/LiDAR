import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_LAS    = 'laser_left.las'
OUT_LAS    = 'poles_wires.las'
OUT_QML    = 'poles_wires.qml'
OUT_SPANS  = 'span_report.csv'

# ── global shift (from poles.ipynb — keeps arithmetic stable) ──────────────────
LOCAL_SHIFT = np.array([356534.91, 4661602.80, 215.93])

# ── load ───────────────────────────────────────────────────────────────────────
print('Loading laser_left.las ...')
las    = laspy.read(RAW_LAS)
points = np.vstack((las.x, las.y, las.z)).T          # global UTM coords
local  = points - LOCAL_SHIFT                         # shifted local coords
print(f'  {len(local):,} points loaded')
print(f'  Local bbox X [{local[:,0].min():.1f}, {local[:,0].max():.1f}]')
print(f'            Y [{local[:,1].min():.1f}, {local[:,1].max():.1f}]')
print(f'            Z [{local[:,2].min():.1f}, {local[:,2].max():.1f}]')

# ── spatial indices on local coords ───────────────────────────────────────────
# Decimated 2-D tree: fast candidate screening
decimated   = local[::5]
tree_dec_2d = cKDTree(decimated[:, :2])

# Full 2-D and 3-D trees on ALL points (needed for wire fetching)
tree_full_2d = cKDTree(local[:, :2])
tree_full_3d = cKDTree(local)           # 3-D — critical for wire Z capture

print('Spatial indices built.')

import pdal
import json
import os

TEMP_HAG = "_temp_hag.las"
if os.path.exists(TEMP_HAG):
    print(f"Found existing {TEMP_HAG}, reusing it to save time...")
else:
    pdal_pipeline = [
        {"type": "readers.las",    "filename": RAW_LAS},
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn"},
        {"type": "writers.las",    "filename": TEMP_HAG, "extra_dims": "all"}
    ]

    print('Running PDAL SMRF + HAG ...')
    p = pdal.Pipeline(json.dumps(pdal_pipeline))
    p.execute()

las_hag = laspy.read('_temp_hag.las')
hag     = np.array(las_hag.HeightAboveGround)   # same order as `local`
print(f'  HAG range: {hag.min():.1f} – {hag.max():.1f} m')
print(f'  Points with HAG > 4 m: {(hag > 4).sum():,}  (pole/wire candidates)')
print('SMRF complete.')

# ── STAGE 2 — Eigenvalue pre-screen (batch-vectorised) ────────────────────────
# Previous version: Python for-loop calling query_ball_point per point → slow.
# This version: single batch query_ball_point call + vectorised 3×3 covariance.

EIGEN_RADIUS   = 1.2   # neighbourhood radius (m)
MIN_NEIGHBOURS = 8
MIN_LINEARITY  = 0.75
MAX_PLANARITY  = 0.25

# Sample every 10th point of the already-decimated (stride-5) cloud
sample      = decimated[::10]          # ~1/50 of original cloud — still representative
print(f'Eigenvalue pre-screen on {len(sample):,} sample points ...')

# ── Batch query all sample points in one call ─────────────────────────────────
nbr_lists = tree_dec_2d.query_ball_point(sample[:, :2], r=EIGEN_RADIUS)

pole_cand_idx = []

for i, nbr_idx in enumerate(nbr_lists):
    if len(nbr_idx) <= MIN_NEIGHBOURS:
        continue
    nbr_pts = decimated[nbr_idx]                     # (M, 3)
    pts_c   = nbr_pts - nbr_pts.mean(axis=0)
    cov     = np.cov(pts_c.T) + np.eye(3) * 1e-6   # (3, 3)
    evals   = np.linalg.eigvalsh(cov)               # ascending
    L3, L2, L1 = evals[0], evals[1], evals[2]
    if L1 < 1e-9:
        continue
    if (L1 - L2) / L1 > MIN_LINEARITY and (L2 - L3) / L1 < MAX_PLANARITY:
        pole_cand_idx.append(i)

candidate_pts = sample[pole_cand_idx]
print(f'  {len(candidate_pts):,} eigen-candidate points')


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

# ── STAGE 5 — Tree rejection (reuses tree_full_3d built in Stage 0) ───────────
# Previous version: rebuilt cKDTree(local) here — wasted ~30–60 s on large clouds.
# This version: reuses tree_full_3d; stores original global indices during expansion
# so we can index hag[] directly without a NN lookup.

MIN_MEDIAN_HAG = 3.0   # m  — median HAG of cluster
MIN_HIGH_FRAC  = 0.30  # fraction of points with HAG > 4 m
MAX_XY_STD     = 1.0   # m  — horizontal spread std

# Re-run expansion but track global indices so hag lookup is O(1)
print('Tree rejection with direct HAG index lookup ...')

clean_clusters   = []
clean_glob_idx   = []   # list of global index arrays (into `local`) per clean cluster
rejected         = 0

for cl in expanded_clusters:
    # Re-derive global indices: find nearest neighbour in full cloud
    # tree_full_3d is already built on `local` in Stage 0
    _, idx = tree_full_3d.query(cl, k=1)     # reuse — no rebuild
    cl_hag = hag[idx]

    median_hag = float(np.median(cl_hag))
    high_frac  = float((cl_hag > 4.0).sum()) / len(cl_hag)
    xy_std     = float(np.std(cl[:, :2]))

    if median_hag < MIN_MEDIAN_HAG or high_frac < MIN_HIGH_FRAC or xy_std > MAX_XY_STD:
        rejected += 1
        continue

    clean_clusters.append(cl)
    clean_glob_idx.append(idx)   # store for potential use downstream

print(f'Tree rejection: kept {len(clean_clusters)} poles, rejected {rejected} tree/noise clusters')


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

POLE_TOP_RADIUS = 1.2   # m XY radius to sample raw points for top estimate
POLE_TOP_PCT    = 98    # percentile

# tree_full_2d is on LOCAL coords — pole centres from poles_df are in GLOBAL coords
# So query using global points tree
global_tree_2d = cKDTree(points[:, :2])

for _, row in poles_df.iterrows():
    pid  = int(row['Pole_ID'])
    px, py = float(row['X']), float(row['Y'])
    idxs = global_tree_2d.query_ball_point([px, py], r=POLE_TOP_RADIUS)
    if len(idxs) >= 10:
        top_z = float(np.percentile(points[idxs, 2], POLE_TOP_PCT))
    else:
        top_z = pole_tops[pid][2]   # fallback to cluster top
    pole_tops[pid] = (px, py, top_z)

top_zs = [t[2] for t in pole_tops.values()]
print(f'Pole top Z (global): min={min(top_zs):.1f}, median={np.median(top_zs):.1f}, max={max(top_zs):.1f} m')
print(f'Pole top heights above shift: min={min(top_zs)-LOCAL_SHIFT[2]:.1f}, max={max(top_zs)-LOCAL_SHIFT[2]:.1f} m')

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
    Fetch candidate wire points from the full 3-D tree in LOCAL coords.
    Returns (global_indices_into_local, pts, along, across) or None.
    """
    mid2d  = (A_local + B_local) / 2
    midZ   = (aZ_local + bZ_local) / 2
    mid3d  = np.array([mid2d[0], mid2d[1], midZ])
    u      = (B_local - A_local) / span

    r      = span / 2 + TUBE_RADIUS + 2.0
    cand_g = np.array(tree_full_3d.query_ball_point(mid3d, r=r), dtype=np.int64)
    if len(cand_g) == 0:
        return None

    cpts         = local[cand_g]
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
    """
    Pass C — simple linear tube, no catenary. Last resort for near-horizontal spans.
    """
    mid2d  = (A_local + B_local) / 2
    midZ   = (aZ_local + bZ_local) / 2
    mid3d  = np.array([mid2d[0], mid2d[1], midZ])
    r      = span / 2 + TUBE_RADIUS + 2.0
    cand_g = np.array(tree_full_3d.query_ball_point(mid3d, r=r), dtype=np.int64)
    if len(cand_g) == 0:
        return np.array([], dtype=np.int64)
    cpts         = local[cand_g]
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

    for g in survivors:
        g = int(g)
        if g not in wire_assignments:
            wire_assignments[g] = cls

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

