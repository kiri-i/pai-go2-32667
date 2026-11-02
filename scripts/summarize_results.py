"""Recompute the report summary from the published simulation records (CPU only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    manifest = json.loads((ROOT / 'evidence/evaluation_manifest.json').read_text())
    records = {}
    for item in manifest['records']:
        path = ROOT / item['file']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"Evaluation record checksum mismatch: {path}")
        records[path.stem] = json.loads(data)
    groups = {'standard': ['standard85-minus', 'standard85', 'standard85-plus'],
              'reward_modified': ['balance85-minus', 'balance85-close', 'balance85-plus']}
    result = {}
    for name, ids in groups.items():
        result[name] = {'successes': sum(records[i]['course_outcome'] == 'success' for i in ids),
                        'conditions': [{'offset_m': records[i]['approach_offset_m'],
                                        'outcome': records[i]['course_outcome']} for i in ids]}
    trace = records[manifest['contact_analysis_trial']]
    both = 0
    for frame in trace['samples']:
        contacts = [(p[0], f[2]) for p, f in zip(frame['feet_positions_m'], frame['feet_contact_forces_n'])]
        both += any(x < 1.50 and fz > 1 for x, fz in contacts) and any(x > 2.35 and fz > 1 for x, fz in contacts)
    result['both_banks_support'] = {'trial': manifest['contact_analysis_trial'],
                                    'seconds': round(both * trace['step_dt_s'], 6), 'steps': both}
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
