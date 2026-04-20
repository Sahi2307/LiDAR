"""
GSV Component Height Extraction
Extracts heights for Primary, Neutral, and Downguy wires from 670x334 GSV masks.
Uses a verified scale of 0.02 meters per pixel.
"""
import os, cv2
import numpy as np
import pandas as pd

# — CONFIGURATION —————————————————————————————————————————————————————————————
GSV_DIR   = 'gsv_masks'
IN_CSV    = 'pole_inventory.csv'
OUT_CSV   = 'pole_inventory.csv' # Overwrite with updated data

SCALE_MPX = 0.02

# Label IDs
ID_SHAFT    = 1
ID_PRIMARY  = 3
ID_NEUTRAL  = 5
ID_DOWNGUY  = 6

# — PROCESSING ————————————————————————————————————————————————————————————————
if not os.path.exists(IN_CSV):
    print(f"Error: {IN_CSV} not found. Run inventory script first.")
    exit(1)

df = pd.read_csv(IN_CSV)
print(f"Loaded inventory with {len(df)} poles.")

# Ensure primary/neutral/downguy columns exist or reset them if they were empty
for col in ['primary_h_m', 'neutral_h_m', 'downguy_h_m']:
    if col not in df.columns:
        df[col] = np.nan

count_updates = 0

print("\nProcessing GSV masks...")
for idx, row in df.iterrows():
    pid = int(row['pole_id'])
    
    # Try both naming conventions: zero-padded and plain
    paths = [
        os.path.join(GSV_DIR, f'pole_{pid:03d}_mask.png'), # pole_001
        os.path.join(GSV_DIR, f'pole_{pid}_mask.png'),    # pole_1
        os.path.join(GSV_DIR, f'pole_{pid}_1_mask.png')   # pole_1_1 (preferred if exists)
    ]
    
    mask = None
    # Prefer _1 version if multiple exist
    for p in reversed(paths):
        if os.path.exists(p):
            mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if mask is not None: 
                mask_file = os.path.basename(p)
                break
    
    if mask is None:
        continue
    
    # Find SHAFT (1) base as ground reference
    y_shaft, _ = np.where(mask == ID_SHAFT)
    if len(y_shaft) == 0:
        print(f"  Pole {pid:03d}: Mask found ({mask_file}) but no SHAFT label found. Skipping.")
        continue
    
    # Shaft base is the largest Y value (bottom of image)
    base_y = y_shaft.max()
    
    # Extract median heights for targets
    updates = []
    
    # Primary (3)
    y_pri, _ = np.where(mask == ID_PRIMARY)
    if len(y_pri) > 0:
        h_pri = round((base_y - np.median(y_pri)) * SCALE_MPX, 2)
        df.at[idx, 'primary_h_m'] = h_pri
        updates.append(f"PRI={h_pri}m")
        
    # Neutral (5)
    y_neu, _ = np.where(mask == ID_NEUTRAL)
    if len(y_neu) > 0:
        h_neu = round((base_y - np.median(y_neu)) * SCALE_MPX, 2)
        df.at[idx, 'neutral_h_m'] = h_neu
        updates.append(f"NEU={h_neu}m")
        
    # Downguy (6)
    y_guy, _ = np.where(mask == ID_DOWNGUY)
    if len(y_guy) > 0:
        h_guy = round((base_y - np.median(y_guy)) * SCALE_MPX, 2)
        df.at[idx, 'downguy_h_m'] = h_guy
        updates.append(f"GUY={h_guy}m")
        
    if updates:
        print(f"  Pole {pid:03d}: Extracted [{', '.join(updates)}] from {mask_file}")
        count_updates += 1

# — SAVE ——————————————————————————————————————————————————————————————————————
df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Updated {OUT_CSV} with GSV heights for {count_updates} poles.")

# Summary of filled columns
print("\nInventory Status (non-null counts/37):")
print(df[['primary_h_m', 'neutral_h_m', 'downguy_h_m']].count())
