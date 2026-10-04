#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Run sh setup_unix.sh first."
    exit 1
fi
exec .venv/bin/python portable.py launch
