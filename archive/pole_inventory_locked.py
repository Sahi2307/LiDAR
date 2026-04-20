import os, cv2
import numpy as np
import pandas as pd
import laspy
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from pyproj import Transformer

# --- CONFIGURATION ---
IN_LAS      = 'allwires_v1.las'
LIDAR_DIR   = 'pole_masks'   # LiDAR side-elevation masks (1280x1024)
GSV_DIR     = 'gsv_masks'     # GSV masks (670x334)
OUT_CSV     = 'pole_inventory.csv'
OUT_KML     = 'pole_inventory.kml'

# LiDAR Labeller Params (must match how masks were generated)
LIDAR_IMG_W = 1280
LIDAR_IMG_H = 1024
LIDAR_RADIUS = 30.0
LIDAR_Z_MARGIN = 1.0

# GSV Labeller Params
GSV_SCALE_MPX = 0.02
GSV_ID_SHAFT   = 1
GSV_ID_PRIMARY = 3
GSV_ID_NEUTRAL = 5
GSV_ID_DOWNGUY = 6

# Label Mapping (Component ID -> Name)
LABEL_NAMES = {
    1:'SHAFT', 2:'CROSSARM', 3:'PRIMARY', 4:'SECONDARY',
    5:'NEUTRAL', 6:'DOWNGUY', 7:'ANCHOR', 8:'TRANSFORMER'
}

# Trust Logic
# LiDAR is used for these:
LIDAR_TRUSTED = {2, 4, 8}  # CROSSARM, SECONDARY, TRANSFORMER
# GSV is used for these (averaged across angles):
GSV_TRUSTED = {3, 5, 6}    # PRIMARY, NEUTRAL, DOWNGUY

# Hardcoded original IDs from notebook
ORIG_IDS = [1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 24, 25, 26, 28, 29, 
            30, 31, 32, 33, 34, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47]

# --- PROCESSING ---

def get_lidar_heights(mask_path, pole_xy, bot_z, top_z, pts, full_tree):
    """Extract heights from LiDAR masks using 3D-to-2D normalized projection."""
    if not os.path.exists(mask_path):
        return {}
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return {}

    z_lo = bot_z - LIDAR_Z_MARGIN
    z_hi = top_z + LIDAR_Z_MARGIN

    # Pull nearby points
    raw_local = np.array(full_tree.query_ball_point(pole_xy, r=LIDAR_RADIUS), dtype=np.int64)
    if len(raw_local) == 0: return {}
    near_pts = pts[raw_local]
    
    # Z-filter
    zm = (near_pts[:, 2] >= z_lo) & (near_pts[:, 2] <= z_hi)
    near_pts = near_pts[zm]
    if len(near_pts) == 0: return {}

    # Projection (Normalized PCA)
    pca = PCA(n_components=2).fit(near_pts[:, :2])
    u = pca.transform(near_pts[:, :2])[:, 0]
    z = near_pts[:, 2]
    
    u_min, u_max = u.min(), u.max()
    u_range = max(u_max - u_min, 1e-3)
    z_range = max(z_hi - z_lo, 1e-3)

    ux = np.clip(((u - u_min) / u_range * (LIDAR_IMG_W - 1)).astype(int), 0, LIDAR_IMG_W - 1)
    vx = np.clip(((1.0 - (z - z_lo) / z_range) * (LIDAR_IMG_H - 1)).astype(int), 0, LIDAR_IMG_H - 1)

    painted = mask[vx, ux]
    heights = {}
    for lid in LIDAR_TRUSTED:
        m = painted == lid
        if m.any():
            heights[lid] = round(float(np.median(z[m])) - bot_z, 2)
    return heights

def get_gsv_heights(new_id):
    """Extract heights from all available GSV masks and average them."""
    # Find all matching files (pole_N.png, pole_N_1.png, etc.)
    mask_paths = [
        os.path.join(GSV_DIR, f'pole_{new_id}_mask.png'),
        os.path.join(GSV_DIR, f'pole_{new_id}_1_mask.png')
    ]
    # Also check zero-padded variants just in case
    mask_paths.append(os.path.join(GSV_DIR, f'pole_{new_id:03d}_mask.png'))
    mask_paths.append(os.path.join(GSV_DIR, f'pole_{new_id:03d}_1_mask.png'))
    
    all_mask_heights = []
    
    # Unique paths that exist
    unique_paths = sorted(list(set([p for p in mask_paths if os.path.exists(p)])))
    
    for p in unique_paths:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        # Find Shaft Base (ground ref)
        y_shaft, _ = np.where(mask == GSV_ID_SHAFT)
        if len(y_shaft) == 0: continue
        base_y = y_shaft.max()
        
        m_heights = {}
        for lid in GSV_TRUSTED:
            y_comp, _ = np.where(mask == lid)
            if len(y_comp) > 0:
                m_heights[lid] = (base_y - np.median(y_comp)) * GSV_SCALE_MPX
        
        if m_heights:
            all_mask_heights.append(m_heights)
            
    if not all_mask_heights:
        return {}
        
    # Average across angles
    final_gsv = {}
    for lid in GSV_TRUSTED:
        vals = [h[lid] for h in all_mask_heights if lid in h]
        if vals:
            final_gsv[lid] = round(float(np.mean(vals)), 2)
            
    return final_gsv

def main():
    print(f"Loading {IN_LAS}...")
    las = laspy.read(IN_LAS)
    pts = np.vstack([las.x, las.y, las.z]).T
    cls = np.array(las.classification, dtype=np.uint8)
    uid = np.array(las.user_data, dtype=np.int32)
    
    utm2wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    full_tree = cKDTree(pts[:, :2])
    
    pole_mask = cls == 65
    
    rows = []
    print(f"Processing {len(ORIG_IDS)} poles...")
    
    for i, orig_pid in enumerate(ORIG_IDS, start=1):
        # Determine sequential new_id (match naming in folders)
        new_id = i
        
        m = pole_mask & (uid == orig_pid)
        if not m.any():
            print(f"  Pole {new_id:02d} (Orig {orig_pid:02d}): No POLE points found. Skipping.")
            continue
            
        p_pole = pts[m]
        pole_xy = p_pole[:, :2].mean(axis=0)
        bot_z = float(p_pole[:, 2].min())
        top_z = float(p_pole[:, 2].max())
        mid_z = float(p_pole[:, 2].mean())
        pole_h = round(top_z - bot_z, 2)
        
        # Coordinates
        lon, lat = utm2wgs.transform(pole_xy[0], pole_xy[1])
        
        # 1. LiDAR Heights
        lidar_p = os.path.join(LIDAR_DIR, f'pole_{new_id:03d}_mask.png')
        lidar_h = get_lidar_heights(lidar_p, pole_xy, bot_z, top_z, pts, full_tree)
        
        # 2. GSV Heights
        gsv_h = get_gsv_heights(new_id)
        
        # Merge
        merged = {}
        merged.update(gsv_h)
        merged.update(lidar_h) # LiDAR wins
        
        row = {
            'pole_id': new_id,
            'orig_id': orig_pid,
            'lat': round(lat, 7),
            'lon': round(lon, 7),
            'X': round(pole_xy[0], 3),
            'Y': round(pole_xy[1], 3),
            'Z_base': round(bot_z, 3),
            'pole_height_m': pole_h,
            'crossarm_h_m': merged.get(2, None),
            'secondary_h_m': merged.get(4, None),
            'transformer_h_m': merged.get(8, None),
            'primary_h_m': merged.get(3, None),
            'neutral_h_m': merged.get(5, None),
            'downguy_h_m': merged.get(6, None),
        }
        rows.append(row)
        
        comp_str = " ".join([f"{LABEL_NAMES[lid]}={v}m" for lid, v in merged.items() if lid in LABEL_NAMES])
        print(f"  Pole {new_id:02d}: H={pole_h}m  Lat={lat:.5f} [{comp_str}]")
        
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} with {len(df)} poles.")
    
    # Save KML
    generate_kml(df)

def generate_kml(df):
    placemarks = []
    for _, row in df.iterrows():
        pid = int(row['pole_id'])
        lat, lon = row['lat'], row['lon']
        z = row['Z_base'] + row['pole_height_m'] # Top of pole
        h = row['pole_height_m']
        
        # Balloon HTML
        comp_rows = ""
        for col, label in [('crossarm_h_m', 'Crossarm'), ('secondary_h_m', 'Secondary'), 
                           ('transformer_h_m', 'Transformer'), ('primary_h_m', 'Primary'), 
                           ('neutral_h_m', 'Neutral'), ('downguy_h_m', 'Downguy')]:
            val = row.get(col)
            if not pd.isna(val):
                comp_rows += f"<tr><td>{label}</td><b><td>{val:.2f} m</td></b></tr>"
        
        desc = f"""<![CDATA[
        <div style="font-family:Arial; min-width:150px">
          <h3>Pole {pid}</h3>
          <p>Height: <b>{h:.1f} m</b></p>
          <table border="1" cellpadding="3" style="border-collapse:collapse; width:100%">
            <tr bgcolor="#eee"><th>Component</th><th>Height (AGL)</th></tr>
            {comp_rows}
          </table>
        </div>
        ]]>"""
        
        pm = f"""  <Placemark>
    <name>ID:{pid} H:{h:.1f}m</name>
    <description>{desc}</description>
    <styleUrl>#poleStyle</styleUrl>
    <Point>
      <altitudeMode>absolute</altitudeMode>
      <coordinates>{lon:.7f},{lat:.7f},{z:.2f}</coordinates>
    </Point>
  </Placemark>"""
        placemarks.append(pm)
        
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Pole Inventory</name>
  <Style id="poleStyle">
    <IconStyle><color>ff00aaff</color><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>
    <LabelStyle><scale>0.7</scale></LabelStyle>
  </Style>
{''.join(placemarks)}
</Document>
</kml>"""
    with open(OUT_KML, 'w', encoding='utf-8') as f:
        f.write(kml)
    print(f"Saved {OUT_KML}")

if __name__ == "__main__":
    main()
