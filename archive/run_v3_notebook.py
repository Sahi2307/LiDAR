import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Paths
WORKSPACE_DIR = r'e:\SAAHIE\poles-84376429-33333333'
DATA_DIR = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles'
NOTEBOOK_IN = os.path.join(WORKSPACE_DIR, "non_pro_v3_optimised_maybe.ipynb")
NOTEBOOK_OUT = os.path.join(DATA_DIR, "non_pro_v3_executed.ipynb")
SHIM_SRC = os.path.join(WORKSPACE_DIR, "pdal.py")
SHIM_DST = os.path.join(DATA_DIR, "pdal.py")

print(f"Copying shim to {DATA_DIR}...")
if os.path.exists(SHIM_SRC):
    shutil.copy(SHIM_SRC, SHIM_DST)

print(f"Loading {NOTEBOOK_IN}...")
with open(NOTEBOOK_IN, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Inject environment variables to prevent sklearn hang
print("Injecting thread limit environment variables...")
env_cell = nbformat.v4.new_code_cell(source="import os\nos.environ['OPENBLAS_NUM_THREADS'] = '1'\nos.environ['MKL_NUM_THREADS'] = '1'\nos.environ['OMP_NUM_THREADS'] = '1'\n")
nb.cells.insert(0, env_cell)

# Proactively update Stage 1 to check for existing HAG file to save time
for cell in nb.cells:
    if "Running PDAL SMRF + HAG" in cell.source:
        print("Injecting HAG file existence check into notebook Stage 1...")
        cell.source = """import os
import laspy
import numpy as np
import pdal
import json

TEMP_HAG = '_temp_hag.las'
if os.path.exists(TEMP_HAG):
    print(f'  Found existing {TEMP_HAG}, reusing it to save 5-10 minutes...')
    las_hag = laspy.read(TEMP_HAG)
else:
    pdal_pipeline = [
        {"type": "readers.las",    "filename": RAW_LAS},
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn"},
        {"type": "writers.las",    "filename": TEMP_HAG, "extra_dims": "all"}
    ]
    print('Running PDAL SMRF + HAG (this may take a few minutes)...')
    p = pdal.Pipeline(json.dumps(pdal_pipeline))
    p.execute()
    las_hag = laspy.read(TEMP_HAG)

hag = np.array(las_hag.HeightAboveGround)   # same order as `local`
print(f'  HAG range: {hag.min():.1f} – {hag.max():.1f} m')
print(f'  Points with HAG > 4 m: {(hag > 4).sum():,}  (pole/wire candidates)')
print('SMRF complete.')"""

# Run the notebook
print("Starting execution...")
ep = ExecutePreprocessor(timeout=3600, kernel_name='python3', startup_timeout=300)

try:
    ep.preprocess(nb, {'metadata': {'path': DATA_DIR}})
    print("Execution successful.")
except Exception as e:
    print(f"Error executing the notebook: {e}")
finally:
    with open(NOTEBOOK_OUT, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Executed notebook saved to {NOTEBOOK_OUT}")
