# Third-party notices

## Project code

The experimental adapters and evaluation code were exported from the author's Robot/Go2 experiment workspace. No license grant is added to this export. Public visibility does not itself grant a permissive software license. The source manifest identifies the imported files and their pre-export SHA-256 hashes.

## Isaac Lab and simulation dependencies

Isaac Lab is obtained as a version-pinned dependency, not redistributed in this repository. The task configuration audit references the official source at:
https://github.com/isaac-sim/IsaacLab/tree/7529e35ff98a851d62015a369b38e29827676f46

Upstream source and package licenses continue to apply. Isaac Sim, Newton, PyTorch, RSL-RL and other dependencies retain their respective licenses and distribution terms. The lock files contain distribution URLs and hashes; they do not replace upstream license terms.

## Robot assets

The two Go2 USD files are downloaded from NVIDIA's official Isaac asset distribution. They are not included in Git and are not relicensed by this project. Download and use are subject to the provider's applicable terms. `sim/assets.lock.json` records the URLs (base plus relative path) and SHA-256 values.
