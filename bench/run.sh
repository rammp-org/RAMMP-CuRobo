#!/bin/bash
# One benchmark condition in the x86 bench image, from the repo root:
#   bench/run.sh <out-dir> <name> [sweep_bench.py run args, e.g. --set num_ik_seeds=512]
set -euo pipefail
OUT=$1
NAME=$2
shift 2
mkdir -p "$OUT"  # as the user: the container runs as root
exec docker run --rm --gpus all -v "$PWD":/work -v rammp-curobo-bench-cache:/root/.cache \
    rammp-curobo-bench:x86 python3 bench/sweep_bench.py run \
    --name "$NAME" --out "$OUT/$NAME.jsonl" "$@"
