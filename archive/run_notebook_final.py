import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

# Paths
WORKSPACE_DIR = r'e:\SAAHIE\poles-84376429-33333333'
DATA_DIR = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles'
NOTEBOOK_IN = os.path.join(WORKSPACE_DIR, "non_pro_v2_with_downguy.ipynb")
NOTEBOOK_OUT = os.path.join(DATA_DIR, "non_pro_v2_with_downguy_executed.ipynb")
SHIM_SRC = os.path.join(WORKSPACE_DIR, "pdal.py")
SHIM_DST = os.path.join(DATA_DIR, "pdal.py")

print(f"Copying shim to {DATA_DIR}...")
if os.path.exists(SHIM_SRC):
    shutil.copy(SHIM_SRC, SHIM_DST)

print(f"Executing {NOTEBOOK_IN}...")
with open(NOTEBOOK_IN, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Modify the notebook to use absolute path for RAW_LAS or just run in DATA_DIR
# We will run in DATA_DIR.

ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')

try:
    # Run the notebook with the working directory set to DATA_DIR
    ep.preprocess(nb, {'metadata': {'path': DATA_DIR}})
    print("Execution successful.")
except Exception as e:
    print(f"Error executing the notebook: {e}")
finally:
    with open(NOTEBOOK_OUT, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Executed notebook saved to {NOTEBOOK_OUT}")
