#!/bin/bash
# Path to .sif apptainer. Use your own container or the prebuilt container below
CONTAINER_PATH=/hnvme/workspace/slcl100h-vllm/custom-verl/verl_vllm012.latest.sif
# In case data is on project space define this such that apptainer binds the project space

# Prevent models from saving it /home/<user>/.cache
DOWNLOAD_DIR=/hnvme/workspace/slcl100h-vllm/hub/

export HF_HOME=$DOWNLOAD_DIR
mkdir -p $DOWNLOAD_DIR
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

MODEL_CHECKPOINT=Qwen/Qwen2.5-0.5B-Instruct


# Get the IP address of the compute node
HOST_IP=$(hostname -I | awk '{print $1}')
unset ROCR_VISIBLE_DEVICES


WANDB_PROJECT_NAME="reinforce-learning"
WANDB_RUN_NAME="grpo_run"
source ~/.bash_profile

# Log the IP address
echo "Determined host IP: $HOST_IP"

model_path=Qwen/Qwen2.5-0.5B-Instruct

datetime=$(date +"%Y%m%d_%H%M%S")
export SSL_CERT_FILE=../cacert.pem
echo "Here"
apptainer exec --nv "${CONTAINER_PATH}" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/data/gsm8k/train.parquet \
    data.val_files=$DATA_PATH/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k_record' \
    trainer.experiment_name='qwen2_7b_function_rm_re' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@ > fsdp2_re.log 2>&1 &
