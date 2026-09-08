#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: bash run.sh cuda|hip IMAGE OUTPUT_DIR [capsule arguments...]" >&2
    exit 2
fi
backend=$1 image=$2 output=$3
shift 3
docker=${CAPSULE_DOCKER:-docker}
context=${CAPSULE_DOCKER_CONTEXT:-rootless}
mkdir -p "$output"
output=$(cd "$output" && pwd)
args=(--context "$context" run --rm --hostname fftm-poisson --shm-size=1g
      --mount "type=bind,src=$output,dst=/data")
case "$backend" in
    cuda)
        IFS=, read -ra devices <<< "${CAPSULE_CUDA_DEVICES:-0,1}"
        for device in "${devices[@]}"; do args+=(--device "nvidia.com/gpu=$device"); done
        args+=(-e NVIDIA_VISIBLE_DEVICES=void)
        ;;
    hip)
        args+=(--device /dev/kfd)
        IFS=, read -ra devices <<< "${CAPSULE_HIP_DEVICES:-/dev/dri/renderD128,/dev/dri/renderD129}"
        for device in "${devices[@]}"; do args+=(--device "$device"); done
        # ROCr may need all render nodes for initialization; select compute GPUs separately.
        if [[ -n ${CAPSULE_HIP_VISIBLE_DEVICES:-} ]]; then
            args+=(-e "ROCR_VISIBLE_DEVICES=$CAPSULE_HIP_VISIBLE_DEVICES")
        fi
        ;;
    *) echo "Backend must be cuda or hip" >&2; exit 2 ;;
esac
if [[ -n ${CAPSULE_UCX_TLS:-} ]]; then
    args+=(-e "CAPSULE_UCX_TLS=$CAPSULE_UCX_TLS")
fi
if (( $# == 0 )); then set -- verify; fi
printf 'Docker context: %s; output: %s\n' "$context" "$output"
"$docker" --context "$context" image inspect "$image" > "$output/image-inspect.json"
printf '%q ' "$docker" "${args[@]}" "$image" "$@"
printf '\n'
exec "$docker" "${args[@]}" "$image" "$@"
