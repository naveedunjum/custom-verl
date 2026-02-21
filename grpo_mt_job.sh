#!/bin/bash

# =============================================================================
# GRPO Training with BLEU Reward
# =============================================================================

# --- Paths & Container ---
CONTAINER_PATH=/hnvme/workspace/slcl100h-vllm/custom-verl/verl_vllm012.latest.sif
DOWNLOAD_DIR=/hnvme/workspace/slcl100h-vllm/hub/
export HF_HOME=$DOWNLOAD_DIR
mkdir -p $DOWNLOAD_DIR

# --- Proxy ---
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export SSL_CERT_FILE=../cacert.pem

# --- Model ---
MODEL_CHECKPOINT=Qwen/Qwen2.5-0.5B-Instruct

# --- Data ---
train_file_path=/hnvme/workspace/slcl100h-vllm/data/train_qwen_chat.parquet
test_file_path=/hnvme/workspace/slcl100h-vllm/data/test_qwen_chat.parquet

# --- Training hyperparameters ---
train_batch_size=512
rollout_num=8

# --- Experiment naming ---
datetime=$(date +"%Y%m%d_%H%M%S")
WANDB_PROJECT_NAME="reinforce-learning"
WANDB_RUN_NAME="grpo_bleu_${datetime}"
exp_name="outputs/grpo_bleu_${datetime}"
mkdir -p $exp_name

# --- Environment ---
HOST_IP=$(hostname -I | awk '{print $1}')
echo "Determined host IP: $HOST_IP"
unset ROCR_VISIBLE_DEVICES
source ~/.bash_profile

# =============================================================================
# Launch
# =============================================================================
apptainer exec --nv \
  "${CONTAINER_PATH}" \
  python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${train_file_path}" \
  data.val_files="${test_file_path}" \
  data.train_batch_size=${train_batch_size} \
  data.val_batch_size=128 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=${MODEL_CHECKPOINT} \
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
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.use_kl_in_reward=False \
  trainer.logger=['console','wandb'] \
  trainer.project_name=${WANDB_PROJECT_NAME} \
  trainer.experiment_name=${WANDB_RUN_NAME} \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.default_local_dir=${exp_name} \
  trainer.default_hdfs_dir=null \
  trainer.save_freq=1000 \
  trainer.test_freq=200 \
  trainer.total_epochs=1 $@ 2>&1 | tee ${exp_name}/grpo_bleu.log
