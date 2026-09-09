#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: NGPU=N bash run.sh cuda|hip IMAGE OUTPUT_DIR [capsule arguments...]" >&2
    exit 2
fi
backend=$1 image=$2 output=$3
shift 3
docker=${CAPSULE_DOCKER:-docker}
context=${CAPSULE_DOCKER_CONTEXT:-rootless}
ngpu=${NGPU-1}
positive_mpi_count()
{
    [[ $1 =~ ^[1-9][0-9]*$ && ${#1} -le 10 ]] && (( $1 <= 2147483647 ))
}
if ! positive_mpi_count "$ngpu"; then
    echo 'NGPU must be a positive decimal integer within the MPI rank-count range.' >&2; exit 2
fi

if (( $# == 0 )); then set -- verify; fi
command=$1
case "$command" in
    verify|preflight|poisson3d|poisson4d|info|--help|-h) ;;
    *) echo 'Specify a capsule command first: verify, preflight, poisson3d, poisson4d, or info.' >&2; exit 2 ;;
esac

# Check explicit rank requests before Docker can allocate devices or create a container.
capsule_args=("$@")
ranks= ranks_set=0
for ((i=1; i<${#capsule_args[@]}; ++i)); do
    case "${capsule_args[i]}" in
        --ranks)
            if (( ranks_set || i+1 >= ${#capsule_args[@]} )); then
                echo '--ranks requires one value and may only be specified once.' >&2; exit 2
            fi
            ranks=${capsule_args[i+1]}
            ranks_set=1
            i=$((i+1))
            ;;
        --ranks=*)
            if (( ranks_set )); then
                echo '--ranks may only be specified once.' >&2; exit 2
            fi
            ranks=${capsule_args[i]#--ranks=}
            ranks_set=1
            ;;
    esac
done
validate_devices()
{
    local label=$1 value=$2 required_count=${3:-0} a b
    local -a entries
    if [[ -z $value || $value == ,* || $value == *, || $value == *,,* || $value == *[[:space:]]* ]]; then
        echo "$label must be a nonempty comma-separated device list." >&2; exit 2
    fi
    IFS=, read -ra entries <<< "$value"
    if (( required_count && ${#entries[@]} != required_count )); then
        echo "$label must select exactly NGPU=$ngpu devices." >&2; exit 2
    fi
    for ((a=0; a<${#entries[@]}; ++a)); do
        if [[ ${entries[a]} == all ]]; then
            echo "$label must select explicit devices, not all." >&2; exit 2
        fi
        for ((b=a+1; b<${#entries[@]}; ++b)); do
            if [[ ${entries[a]} == "${entries[b]}" ]]; then
                echo "$label contains a duplicate device." >&2; exit 2
            fi
        done
    done
}

if (( ranks_set )); then
    validate_devices --ranks "$ranks"
    IFS=, read -ra counts <<< "$ranks"
    for count in "${counts[@]}"; do
        if ! positive_mpi_count "$count" || (( count > ngpu )); then
            echo "--ranks must contain distinct positive counts not exceeding NGPU=$ngpu." >&2; exit 2
        fi
    done
    if [[ $command == poisson* && $ranks == *,* ]]; then
        echo 'An individual Poisson solve requires a single rank count.' >&2; exit 2
    fi
fi

device_args=()
case "$backend" in
    cuda)
        inventory=$(python3 "$(dirname "${BASH_SOURCE[0]}")/host_devices.py" cuda)
        IFS=, read -ra available <<< "$inventory"
        if (( ${#available[@]} < ngpu )); then
            echo "NGPU=$ngpu exceeds the ${#available[@]} NVIDIA GPUs reported by nvidia-smi." >&2; exit 2
        fi
        defaults=("${available[@]:0:ngpu}")
        default_devices=$(IFS=,; printf '%s' "${defaults[*]}")
        selected=${CAPSULE_CUDA_DEVICES-$default_devices}
        validate_devices CAPSULE_CUDA_DEVICES "$selected" "$ngpu"
        IFS=, read -ra devices <<< "$selected"
        for device in "${devices[@]}"; do device_args+=(--device "nvidia.com/gpu=$device"); done
        device_args+=(-e NVIDIA_VISIBLE_DEVICES=void)
        ;;
    hip)
        if [[ ${CAPSULE_HIP_DEVICES+x} ]]; then
            selected=$CAPSULE_HIP_DEVICES
        else
            selected=$(python3 "$(dirname "${BASH_SOURCE[0]}")/host_devices.py" hip)
        fi
        validate_devices CAPSULE_HIP_DEVICES "$selected"
        IFS=, read -ra devices <<< "$selected"
        if (( ${#devices[@]} < ngpu )); then
            echo "NGPU=$ngpu requires at least $ngpu AMD render nodes; found ${#devices[@]}." >&2; exit 2
        fi
        device_args+=(--device /dev/kfd)
        for device in "${devices[@]}"; do device_args+=(--device "$device"); done
        # ROCr can need every AMD render node even when only one compute GPU is selected.
        default_devices=0
        for ((i=1; i<ngpu; ++i)); do default_devices+=,$i; done
        visible=${CAPSULE_HIP_VISIBLE_DEVICES-$default_devices}
        validate_devices CAPSULE_HIP_VISIBLE_DEVICES "$visible" "$ngpu"
        device_args+=(-e "ROCR_VISIBLE_DEVICES=$visible")
        ;;
    *) echo "Backend must be cuda or hip" >&2; exit 2 ;;
esac
# Expand the default matrix only after checking the host's available device count.
if (( ! ranks_set )); then
    case "$command" in
        verify|preflight)
            ranks=1
            for ((i=2; i<=ngpu; ++i)); do ranks+=,$i; done
            ;;
        poisson3d|poisson4d) ranks=$ngpu ;;
    esac
    if [[ -n $ranks ]]; then capsule_args+=(--ranks "$ranks"); fi
fi
mkdir -p "$output"
output=$(cd "$output" && pwd)
args=(--context "$context" run --rm --hostname fftm-poisson --shm-size=1g
      --mount "type=bind,src=$output,dst=/data" -e "NGPU=$ngpu" "${device_args[@]}")
if [[ -n ${CAPSULE_UCX_TLS:-} ]]; then
    args+=(-e "CAPSULE_UCX_TLS=$CAPSULE_UCX_TLS")
fi
printf 'Docker context: %s; NGPU=%s; ranks=%s; output: %s\n' "$context" "$ngpu" "${ranks:-none}" "$output"
"$docker" --context "$context" image inspect "$image" > "$output/image-inspect.json"
printf '%q ' "$docker" "${args[@]}" "$image" "${capsule_args[@]}"
printf '\n'
exec "$docker" "${args[@]}" "$image" "${capsule_args[@]}"
