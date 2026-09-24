from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"

raise SystemExit(
    subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
        ],
        cwd=str(ROOT),
    )
)
