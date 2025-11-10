#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

read_config_var() {
  local var_name="$1"
  python3 - <<'PYCODE'
import importlib
import sys

module = importlib.import_module("config")
value = getattr(module, sys.argv[1], None)
if value is None or not str(value).strip():
    sys.exit(1)
print(str(value).strip())
PYCODE
}

export HORUS_API_KEY="$(read_config_var HORUS_API_KEY)"
export ROOSTOO_API_KEY="$(read_config_var ROOSTOO_API_KEY)"
export ROOSTOO_SECRET_KEY="$(read_config_var ROOSTOO_SECRET_KEY)"

if POLL_INTERVAL=$(read_config_var POLL_INTERVAL_SECONDS 2>/dev/null); then
  export POLL_INTERVAL_SECONDS="$POLL_INTERVAL"
fi

python3 2.py
