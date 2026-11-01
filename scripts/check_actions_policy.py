"""Check this repository's self-hosted, no-remote-cache Actions contract."""
from pathlib import Path
import yaml

def check(root):
    errors = []
    for path in (root / ".github/workflows").glob("*.y*ml"):
        data = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        if set(data.get("on", {})) - {"push", "workflow_dispatch"}:
            errors.append("Unreviewed event may execute untrusted code")
        if "concurrency" not in data:
            errors.append("Missing concurrency cancellation")
        for name, job in data.get("jobs", {}).items():
            if "uses" in job:
                errors.append(f"{name}: reusable workflow is not audited")
            runner = job.get("runs-on", [])
            if not (runner == "self-hosted" or isinstance(runner, list) and "self-hosted" in runner):
                errors.append(f"{name}: self-hosted required")
            if "timeout-minutes" not in job:
                errors.append(f"{name}: timeout required")
            for step in job.get("steps", []):
                use = step.get("uses", "").split("@")[0]
                opts = step.get("with", {})
                if use.startswith("actions/cache") or use.startswith("actions/upload-artifact"):
                    errors.append("Remote caches and artifact uploads disabled")
                if use == "astral-sh/setup-uv" and opts.get("enable-cache") != "false":
                    errors.append("setup-uv cache must be explicitly disabled")
                if opts.get("cache") or "type=gha" in str(step):
                    errors.append("Implicit remote cache disabled")
    return errors

if __name__ == "__main__":
    errors = check(Path.cwd())
    print("\n".join(errors) if errors else "Actions policy: 0 violations")
    raise SystemExit(bool(errors))
