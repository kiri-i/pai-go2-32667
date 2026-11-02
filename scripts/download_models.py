"""Download the published checkpoints and verify their size and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/manifest.json"
COMPARISON = ("stage4-standard", "stage4-balance")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]


def download(entry, destination):
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / entry["file"]
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not verify(target, entry):
            raise ValueError(f"Existing model differs; refusing to overwrite: {target}")
        return target
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".download-", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(entry["url"], timeout=120) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > entry["bytes"]:
                        raise ValueError(f"Download exceeds declared size: {entry['id']}")
                    output.write(chunk)
        if not verify(temporary, entry):
            raise ValueError(f"Size or SHA-256 mismatch: {entry['id']}")
        # Exclusive creation also protects against another writer appearing mid-download.
        with target.open("xb") as output, temporary.open("rb") as source:
            import shutil
            shutil.copyfileobj(source, output)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    entries = {item["id"]: item for item in json.loads(MANIFEST.read_text())["models"]}
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Include all six training stages")
    group.add_argument("--model", choices=entries)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/models")
    args = parser.parse_args()
    selected = entries if args.all else [args.model] if args.model else COMPARISON
    for name in selected:
        path = download(entries[name], args.output)
        print(f"Verified {name}: {path}")


if __name__ == "__main__":
    main()
