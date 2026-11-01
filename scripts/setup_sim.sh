#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
findmnt -T . -no FSTYPE | grep -qv '^tmpfs$'
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.local/uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.local/sim"
if [[ ! -e "$UV_PROJECT_ENVIRONMENT" ]]; then
  PY="$(uv python find 3.12)"
  mkdir -p "$UV_PROJECT_ENVIRONMENT/lib64"
  "$PY" -m venv --copies --without-pip "$UV_PROJECT_ENVIRONMENT"
fi
uv sync --locked --link-mode copy --project sim
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/patch_material.py
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/asset_manager.py prepare
