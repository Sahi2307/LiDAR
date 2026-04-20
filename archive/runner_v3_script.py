import os
import sys
import subprocess
import shutil

WORKSPACE_DIR = r'e:\SAAHIE\poles-84376429-33333333'
DATA_DIR = r'e:\SAAHIE\poles-20260416T175107Z-3-001\poles'
SCRIPT = r'e:\SAAHIE\poles-84376429-33333333\non_pro_v5_cached.py'

SHIM_SRC = os.path.join(WORKSPACE_DIR, "pdal.py")
SHIM_DST = os.path.join(DATA_DIR, "pdal.py")

print(f"Copying shim to {DATA_DIR}...")
if os.path.exists(SHIM_SRC):
    shutil.copy(SHIM_SRC, SHIM_DST)

print(f"Executing {SCRIPT} in {DATA_DIR}...")
os.chdir(DATA_DIR)
sys.stdout.flush()

p = subprocess.Popen([sys.executable, "-u", SCRIPT])
p.communicate()
sys.exit(p.returncode)
