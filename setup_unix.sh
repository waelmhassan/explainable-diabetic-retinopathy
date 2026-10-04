#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
python3.11 portable.py setup
