"""Apply the single recorded material alias compatibility patch, guarded by hashes."""
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
audit = json.loads((root / "evidence/baseline_source_audit.json").read_text())
for item in audit["files"]:
    package, relative = item["file"].split("/", 1)
    spec = importlib.util.find_spec(package)
    path = Path(next(iter(spec.submodule_search_locations))) / relative
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    if current == item["installed_sha256"]:
        continue
    if current != item["official_sha256"]:
        raise RuntimeError(f"Unexpected upstream source: {item['file']}")
    text = path.read_text().replace("sim_utils.PhysxRigidBodyMaterialCfg(", "sim_utils.RigidBodyMaterialCfg(")
    if hashlib.sha256(text.encode()).hexdigest() != item["installed_sha256"]:
        raise RuntimeError("Patch does not match recorded source")
    path.write_text(text)
print("Recorded official task source hashes verified")
