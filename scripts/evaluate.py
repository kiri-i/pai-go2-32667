"""Evaluate all predeclared starts, retaining JSON even when task completion fails."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--gap-cm", type=int, choices=[75, 80, 85, 90], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--python", default=str(ROOT / ".local/sim/bin/python"))
parser.add_argument("--view", choices=["original", "readable", "side"], default="readable")
a = parser.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
script = {"original": "probe_step.py", "readable": "probe_readable.py", "side": "probe_side.py"}[a.view]
results = []
for label, offset in [("near", .15), ("base", 0.), ("far", -.15)]:
    out = a.output / (label + ".json")
    cmd = [a.python, str(ROOT / "scripts" / script), "--rough", "--go2-backend", "newton", "--nominal-physics", "--heading-hold", "--checkpoint", str(a.checkpoint.resolve()), "--course-file", str(ROOT / "scripts/course_frontier.json"), "--suite-case", f"gap{a.gap_cm}", "--approach-offset", str(offset), "--start-after", "50", "--speed", "1", "--steps", "750", "--viz", "newton_gl", "--video-dir", str(a.output / label), "--output", str(out)]
    run = subprocess.run(cmd, cwd=ROOT)
    if not out.is_file():
        raise RuntimeError(f"Evaluation did not produce a record: {label}, exit={run.returncode}")
    result = json.loads(out.read_text())
    results.append({"start": label, "offset_m": offset, "exit_code": run.returncode, "course_outcome": result.get("course_outcome"), "checkpoint_sha256": result.get("checkpoint_sha256")})
(a.output / "summary.json").write_text(json.dumps(results, indent=2))
