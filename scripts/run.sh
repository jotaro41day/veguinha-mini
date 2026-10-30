#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec python main.py --serve --body-model mk1 --start-paused "$@"
