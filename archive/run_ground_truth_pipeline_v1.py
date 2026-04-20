"""
Ground Truth Pipeline (Adapted for Labeller V1)
Extracts component heights from V1 masks (dynamic per-pole scaling).
"""
import os, cv2
import numpy as np
import laspy
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# — CONFIGURATION —————————————————————————————————————————————————————————————
IN_LAS      = 'allwires_v1_with_ids.las'
LIDAR_DIR   = 'pole_masks'   # Sequential 001, 002...
GSV_DIR     = 'gsv_masks'
FUSED_DIR   = 'gt_fused_masks'
OUT_GT_CSV  = 'ground_truth.csv'
OUT_SUM_CSV = 'component_heights_summary.csv'

os.makedirs(FUSED_DIR, exist_ok=True)

# Labeller V1 Geometry (Must Match run_full_component_labeller.py)
IMG_W         = 1280
IMG_H         = 1024
RENDER_RADIUS = 30.0
Z_MARGIN      = 1.0

LABEL_NAMES = {
    1: 'SHAFT', 2: 'CROSSARM', 3: 'PRIMARY', 4: 'SECONDARY',
    5: 'NEUTRAL', 6: 'DOWNGUY', 7: 'ANCHOR', 8: 'TRANSFORMER'
}

# — LOAD DATA —————————————————————————————————————————————————————————————————
print(f"Loading {IN_LAS}...")
las = laspy.read(IN_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
cls = np.array(las.classification, dtype=np.uint8)
uid = np.array(las.user_data, dtype=np.int32)
print(f"  {len(pts):,} points loaded.")

# Build pole index
pole_mask = cls == 65
pole_ids  = np.unique(uid[pole_mask])
pole_ids  = pole_ids[pole_ids > 0]

pole_info = {}
for pid in pole_ids:
    m = pole_mask & (uid == pid)
    p = pts[m]
    pole_info[pid] = {
        'xy'   : p[:, :2].mean(axis=0),
        'top_z': float(p[:, 2].max()),
        'bot_z': float(p[:, 2].min()),
    }
print(f"  {len(pole_info)} poles indexed.")

full_tree = cKDTree(pts[:, :2])

# — FUSION & HEIGHT EXTRACTION ————————————————————————————————————————————————
rows = []
fusion_report = []

print(f"\nProcessing {len(pole_ids)} poles...")

for pid in sorted(pole_ids):
    info    = pole_info[pid]
    pole_xy = info['xy']
    bot_z   = info['bot_z']
    top_z   = info['top_z']
    pole_h  = top_z - bot_z
    z_lo    = bot_z - Z_MARGIN
    z_hi    = top_z + Z_MARGIN

    # 1. Fuse Masks
    lidar_path = os.path.join(LIDAR_DIR, f'pole_{pid:03d}_mask.png')
    gsv_path   = os.path.join(GSV_DIR,   f'pole_{pid:03d}_mask.png')
    
    lidar_mask = cv2.imread(lidar_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(lidar_path) else None
    gsv_mask   = cv2.imread(gsv_path,   cv2.IMREAD_GRAYSCALE) if os.path.exists(gsv_path)   else None

    if lidar_mask is None and gsv_mask is None:
        fusion_report.append({'pole_id': pid, 'source': 'NONE'})
        continue

    if lidar_mask is None:
        fused = gsv_mask; src = 'GSV_ONLY'
    elif gsv_mask is None:
        fused = lidar_mask; src = 'LIDAR_ONLY'
    else:
        fused = gsv_mask.copy()
        mask_nz = lidar_mask > 0
        fused[mask_nz] = lidar_mask[mask_nz] # LiDAR wins
        src = 'FUSED'

    cv2.imwrite(os.path.join(FUSED_DIR, f'pole_{pid:03d}_mask.png'), fused)
    fusion_report.append({'pole_id': pid, 'source': src})

    # 2. Back-Project to 3D Heights
    # Replicate V1 Projection Logic
    raw_local = np.array(full_tree.query_ball_point(pole_xy, r=RENDER_RADIUS), dtype=np.int64)
    if len(raw_local) == 0: continue
    
    near_pts = pts[raw_local]; near_cls = cls[raw_local]
    zm = (near_pts[:, 2] >= z_lo) & (near_pts[:, 2] <= z_hi)
    near_pts = near_pts[zm]; near_gidx = raw_local[zm]
    
    if len(near_pts) == 0: continue

    # PCA on XY for horizontal axis
    pca = PCA(n_components=2).fit(near_pts[:, :2])
    u = pca.transform(near_pts[:, :2])[:, 0]
    z = near_pts[:, 2]

    u_min, u_max = u.min(), u.max()
    u_range = max(u_max - u_min, 1e-3)
    z_range = max(z_hi - z_lo, 1e-3)

    # Pixel coords
    ux = np.clip(((u - u_min) / u_range * (IMG_W - 1)).astype(int), 0, IMG_W - 1)
    vx = np.clip(((1.0 - (z - z_lo) / z_range) * (IMG_H - 1)).astype(int), 0, IMG_H - 1)

    painted = fused[vx, ux]

    # Add Pole Base Reference
    rows.append({
        'pole_id': pid, 'component': 'POLE_BASE',
        'height_above_ground_m': 0.0, 'abs_z': round(bot_z, 3),
        'pole_height_m': round(pole_h, 2), 'n_points': int((uid==pid).sum())
    })

    # Calculate median height for each painted label
    found_comps = []
    for label_id, name in LABEL_NAMES.items():
        m = painted == label_id
        if not m.any(): continue
        
        comp_z = z[m]
        median_z = float(np.median(comp_z))
        rows.append({
            'pole_id': pid, 'component': name,
            'height_above_ground_m': round(median_z - bot_z, 2),
            'abs_z': round(median_z, 3),
            'pole_height_m': round(pole_h, 2),
            'n_points': int(m.sum())
        })
        found_comps.append(name)
    
    print(f"  Pole {pid:03d} [{src:10}]: h={pole_h:4.1f}m, labels=[{', '.join(found_comps)}]")

# — SAVE RESULTS ——————————————————————————————————————————————————————————————
df = pd.DataFrame(rows)
df.to_csv(OUT_GT_CSV, index=False)
print(f"\nSaved {OUT_GT_CSV} ({len(df)} rows)")

# Pivot for Summary
if not df.empty:
    pivot = df.pivot_table(
        index=['pole_id', 'pole_height_m'],
        columns='component',
        values='height_above_ground_m',
        aggfunc='median'
    ).reset_index()
    
    # Order columns predictably
    cols = ['pole_id', 'pole_height_m']
    for c in ['SHAFT','CROSSARM','TRANSFORMER','PRIMARY','SECONDARY','NEUTRAL','DOWNGUY','ANCHOR']:
        if c in pivot.columns: cols.append(c)
    
    pivot = pivot[cols]
    pivot.to_csv(OUT_SUM_CSV, index=False)
    print(f"Saved {OUT_SUM_CSV}")
    print("\nSummary Snapshot:")
    print(pivot.head(10).to_string(index=False))

pd.DataFrame(fusion_report).to_csv('fusion_report.csv', index=False)
