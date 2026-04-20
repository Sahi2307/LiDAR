"""
PDAL Python bindings shim — uses the QGIS-bundled pdal.exe CLI
to execute PDAL pipelines when the native Python bindings are unavailable.

Drop this file next to your notebooks and `import pdal` will find it.
"""

import json
import subprocess
import tempfile
import os

__version__ = "CLI-shim-2.9.0"

# Path to the QGIS-bundled pdal.exe
_PDAL_EXE = r"C:\Program Files\QGIS 3.44.4\bin\pdal.exe"

if not os.path.isfile(_PDAL_EXE):
    # Try fallback paths
    for candidate in [
        r"C:\Program Files\QGIS 3.40.12\bin\pdal.exe",
        r"C:\OSGeo4W64\bin\pdal.exe",
        r"C:\OSGeo4W\bin\pdal.exe",
    ]:
        if os.path.isfile(candidate):
            _PDAL_EXE = candidate
            break
    else:
        raise FileNotFoundError(
            f"Cannot find pdal.exe. Looked in:\n"
            f"  - C:\\Program Files\\QGIS 3.44.4\\bin\\pdal.exe\n"
            f"  - C:\\Program Files\\QGIS 3.40.12\\bin\\pdal.exe\n"
            f"  - C:\\OSGeo4W64\\bin\\pdal.exe\n"
            f"Install QGIS or set _PDAL_EXE in this file."
        )


class Pipeline:
    """Minimal Pipeline class that mimics pdal.Pipeline for CLI execution."""

    def __init__(self, pipeline_json: str):
        self._pipeline_json = pipeline_json
        self._pipeline = json.loads(pipeline_json)
        self._count = 0

    def execute(self):
        """Execute the pipeline using pdal.exe CLI."""
        # Write pipeline JSON to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="."
        ) as f:
            f.write(self._pipeline_json)
            pipeline_file = f.name

        try:
            result = subprocess.run(
                [_PDAL_EXE, "pipeline", pipeline_file],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for large point clouds
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pdal pipeline failed (exit code {result.returncode}):\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
            # Try to extract point count from output
            for line in result.stdout.split("\n"):
                if "point" in line.lower():
                    try:
                        self._count = int("".join(c for c in line if c.isdigit()))
                    except ValueError:
                        pass
            return self._count
        finally:
            try:
                os.unlink(pipeline_file)
            except OSError:
                pass

    @property
    def count(self):
        return self._count
