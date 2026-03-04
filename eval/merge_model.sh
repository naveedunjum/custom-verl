#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local_dir)
            local_dir="$2"
            shift 2
            ;;
        --target_dir)
            target_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Require both arguments
if [[ -z "$local_dir" || -z "$target_dir" ]]; then
    echo "Usage: $0 --local_dir <path> --target_dir <path>"
    exit 1
fi

apptainer exec verl_vllm012.latest.sif python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --target_dir "$target_dir" \