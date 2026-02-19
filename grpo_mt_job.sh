apptainer exec --nv \
    "${CONTAINER_PATH}" \
    python3 -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file_path} \
    data.val_files=${test_file_path} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=verl/utils/reward_score/bleu_reward.py \
    custom_reward_function.name=compute_score \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${exp_name} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=1000 \
    trainer.test_freq=200 \
    trainer.total_epochs=1 $@ 2>&1 | tee grpo_bleu.log
