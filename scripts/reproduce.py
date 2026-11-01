"""Print reproducible commands; execute only when --execute is explicitly given."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "baseline": None,
    "first": ("baseline", "first", "standard"),
    "second": ("first", "jump", "standard"),
    "stage3-continued": ("second", "jump", "standard"),
    "stage3-exposure": ("second", "hard", "standard"),
    "stage3-shaped": ("second", "hard", "jump_balance"),
    "stage4-standard": ("stage3-exposure", "hard", "standard"),
    "stage4-balance": ("stage3-exposure", "hard", "jump_balance"),
    "stage4-sequence": ("stage3-exposure", "hard", "sequence"),
}
def command(stage, python, runs):
    output = runs / stage
    if stage == "baseline":
        return [python, str(ROOT / "scripts/train_baseline.py"), "--output", str(output), "--viz", "none"]
    parent, terrain, variant = STAGES[stage]
    return [python, str(ROOT / "scripts/terrain_train.py"), "--kind", "gap", "--stage", terrain, "--variant", variant, "--checkpoint", str(runs / parent / "final.pt"), "--iterations", "400", "--num-envs", "1024", "--output", str(output), "--viz", "none"]
if __name__ == "__main__":
    import shlex
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--python", default=str(ROOT / ".local/sim/bin/python"))
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args()
    for stage in STAGES if a.stage == "all" else [a.stage]:
        cmd = command(stage, a.python, a.runs.resolve())
        print(shlex.join(cmd), flush=True)
        if a.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)
