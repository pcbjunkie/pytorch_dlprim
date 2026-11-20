#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# < 6 )); then
    echo 'Usage: run-radeon-yolo.sh GPU MODEL train|detect 256|512|1024|hd BATCH OUTPUT [ARG ...]' >&2
    echo 'GPU: r9-270, hd7970, s7150. MODEL: yolov8n, yolo26n, yolox-nano.' >&2
    exit 2
fi

gpu=$1 model=$2 workload=$3 resolution=$4 batch=$5 output=$6
shift 6
case "$gpu" in r9-270|hd7970|s7150) ;; *) echo "Unknown GPU preset: $gpu" >&2; exit 2;; esac
case "$model" in
    yolov8n) family=wildlife ;;
    yolo26n) family=yolo26 ;;
    yolox-nano) family=yolox ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
case "$resolution" in 256|512|1024|hd) ;; *) echo "Unknown resolution: $resolution" >&2; exit 2;; esac
if [[ ! $batch =~ ^[1-9][0-9]*$ ]]; then
    echo 'Batch must be a positive integer' >&2
    exit 2
fi

export OPENCL_POSE_ROOT=${OPENCL_POSE_ROOT:-$PWD}
export PYTHONPATH="$repo/examples${PYTHONPATH:+:$PYTHONPATH}"
export RUSTICL_ENABLE=${RUSTICL_ENABLE:-radeonsi}
cache=${RADEON_YOLO_CACHE_DIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/pytorch-dlprim/$gpu"}
mkdir -p -- "$cache"
flags=("DLPRIM_CACHE_DIR=$cache" "BENCH_RESOLUTION=$resolution" DLPRIM_CUSTOM_REDUCE=1)
args=(--batch "$batch" --output "$output")
preset="$repo/docs/presets"

case "$workload" in
    train)
        script="train-$model.py"
        args+=(--warmup 4 --steps 16 --check-loss)
        training_preset="$preset/training-common.env"
        if [[ $gpu == r9-270 && $model == yolov8n && $resolution == 256 ]]; then
            training_preset="$preset/training-yolov8n-r9-270-portable.env"
        fi
        mapfile -t training_flags < "$training_preset"
        flags+=("${training_flags[@]}")
        if [[ $model == yolov8n ]]; then
            args+=(--prefetch-input)
        else
            mapfile -t model_flags < "$preset/training-$model.env"
            flags+=("${model_flags[@]}")
            args+=(--family "$family")
        fi
        ;;
    detect)
        script=benchmark-hd-model.py
        args+=(--family "$family" --resolution "$resolution" --seconds 16 --forward-seconds 2
               --warmup 3 --check-images 8
               --copy-optimizations --small-k --pointwise-reduce --gemm-n balanced
               --fuse-conv-silu --fuse-sppf --winograd-mode small --input-packing batch
               --prefetch-input --gpu-detections --depthwise-stride2 --depthwise-layout 32x4
               --gemm-experiments loader,barrier,stores,tiny,narrow64,shapes,exact80)
        flags+=(OPENCL_DEFERRED_RELEASE_MB=256)
        if [[ $model == yolo26n ]]; then
            mapfile -t model_flags < "$preset/detection-yolo26n.env"
            flags+=("${model_flags[@]}")
            args+=(--postprocess-device cpu --cpu-head-layout channels-first)
        elif [[ $model == yolox-nano ]]; then
            args+=(--fuse-input-focus)
        fi
        ;;
    *) echo "Unknown workload: $workload" >&2; exit 2 ;;
esac

exec bash "$repo/tools/with-radeon-gcn-baseline.sh" env "${flags[@]}" \
    "${PYTHON:-python3}" "$repo/examples/radeon-yolo/$script" "${args[@]}" "$@"
