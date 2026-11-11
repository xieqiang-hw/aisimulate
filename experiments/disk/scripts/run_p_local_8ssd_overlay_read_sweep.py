"""Run the 8x overlay SSD read-bandwidth sensitivity cases."""
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

if __name__ == "__main__":
    project = Path.home() / "project"
    bundle = project / "data/freeinference_agentic_trace/derived/dynosim-openloop-v1/428f809eef5952012f9c-overlay-8x"
    output = project / "results/glm52-overlay-8x-p128gb-ssd-read-sweep"
    for bw in (16, 24):
        name = f"p_local_8ssd_p128gb_overlay_8x_read{bw}"
        print(f"Running {name}", flush=True)
        subprocess.run(
            [sys.executable, str(BASE / "scripts/run_p_g3_local.py"),
             "--config", str(BASE / f"configs/{name}.yaml"),
             "--bundle", str(bundle), "--output-root", str(output / name),
             "--speedup", "1"], check=True,
        )
