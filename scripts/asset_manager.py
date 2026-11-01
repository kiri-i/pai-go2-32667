"""Download only the two external robot assets; verify every use against SHA-256."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("GO2_ASSET_DIR", ROOT / ".local/assets")).resolve()

def manifest_entries():
    manifest = json.loads((ROOT / "sim/assets.lock.json").read_text())
    entries = {item["path"]: item["sha256"] for item in manifest["assets"]}
    for name, sha in entries.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or len(sha) != 64:
            raise ValueError("Invalid asset manifest")
    return manifest["base_url"], entries

def verify_assets():
    _, entries = manifest_entries()
    result = {}
    for name, sha in entries.items():
        path = ASSETS / name
        if any(parent.is_symlink() for parent in [path, *path.parents]):
            raise ValueError("Asset symlinks are not supported")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Missing or mismatched asset: {name}")
        result[name] = path
    return result

def prepare():
    base, entries = manifest_entries()
    for name, sha in entries.items():
        path = ASSETS / name
        if path.exists():
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Existing asset mismatch: {name}")
            continue
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError("Asset symlinks are not supported")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base + "/" + name, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"Download checksum mismatch: {name}")
        with path.open("xb") as stream:
            stream.write(data)
    verify_assets()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else verify_assets()
