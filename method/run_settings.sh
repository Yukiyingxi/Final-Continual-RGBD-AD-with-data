#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${METHOD_DIR}/.." && pwd)"
CODE_DIR="${METHOD_DIR}/code"
PYTHON="${METHOD_DIR}/env/bin/python"
DATA_DIR="${PACKAGE_DIR}/data/MVTec3D-processed"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: bash method/run_settings.sh {10-0|9-1|6-4|6-1x4|all}

Settings:
  10-0    Train all ten categories jointly in one stage.
  9-1     Train the first nine categories, then add tire.
  6-4     Train the first six categories, then add the remaining four.
  6-1x4   Train the first six, then add peach, potato, rope and tire one by one.
  all     Run all four settings sequentially from scratch.

Set DRY_RUN=1 to print the exact commands without creating files or training.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  10-0|9-1|6-4|6-1x4) SETTINGS=("$1") ;;
  all) SETTINGS=(10-0 9-1 6-4 6-1x4) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing tested Python environment: ${PYTHON}" >&2
  exit 3
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing processed MVTec3D dataset: ${DATA_DIR}" >&2
  exit 4
fi

output_for() {
  printf '%s/runs/%s_seed1' "${METHOD_DIR}" "$1"
}

for setting in "${SETTINGS[@]}"; do
  output_dir="$(output_for "${setting}")"
  if [[ -e "${output_dir}/args.json" || -e "${output_dir}/metrics.csv" ]]; then
    echo "Refusing to overwrite existing run: ${output_dir}" >&2
    exit 5
  fi
done

run_one() {
  local setting="$1"
  local output_dir
  output_dir="$(output_for "${setting}")"
  local -a command=(
    "${PYTHON}" train_mvtec3d_jitter_class_continual.py
    --setting "${setting}"
    --data-path "${DATA_DIR}"
    --output-dir "${output_dir}"
    --backbone dinov2reg_vit_base_14
    --device cuda:0
    --image-size 448 --crop-size 392 --metric-size 256
    --batch-size 16 --eval-batch-size 16 --collect-batch-size 4 --workers 4
    --base-iters 40000 --incremental-iters 5000
    --learning-rate 0.002 --dropout 0.4
    --class-loss-weight 1.0
    --feature-jitter-scale 20 --feature-jitter-prob 1
    --cfns-energy 0.995 --cfns-rank 128 --cfns-rows 512
    --dsns-energy 0.97 --dsns-rank 32 --dsns-rows 128
    --rows-per-call 32 --collect-batches 16 --log-every 100 --seed 1
  )

  printf 'setting=%s output=%s\n' "${setting}" "${output_dir}"
  printf 'cd %q &&' "${CODE_DIR}"
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${output_dir}"
  (
    cd "${CODE_DIR}"
    "${command[@]}"
  )
}

for setting in "${SETTINGS[@]}"; do
  run_one "${setting}"
done