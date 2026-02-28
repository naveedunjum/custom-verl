apptainer exec ../verl_vllm012.latest.sif python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir outputs/grpo_bleu_20260225_002433/global_step_255/actor \
    --target_dir /path/to/merged_model