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
  CONFIG_QUERY="$var_name" python3 - <<'PYCODE'
import importlib
import sys
import os

var_name = os.environ.get("CONFIG_QUERY")
if not var_name:
  sys.exit(1)

try:
  module = importlib.import_module("config")
except ModuleNotFoundError:
  sys.exit(1)

value = getattr(module, var_name, None)
if value is None:
    sys.exit(1)
value_str = str(value).strip()
if not value_str:
    sys.exit(1)
print(value_str)
PYCODE
}

export HORUS_API_KEY="$(read_config_var HORUS_API_KEY)"
export ROOSTOO_API_KEY="$(read_config_var ROOSTOO_API_KEY)"
export ROOSTOO_SECRET_KEY="$(read_config_var ROOSTOO_SECRET_KEY)"

if POLL_INTERVAL=$(read_config_var POLL_INTERVAL_SECONDS 2>/dev/null); then
  export POLL_INTERVAL_SECONDS="$POLL_INTERVAL"
fi

python3 2.py
