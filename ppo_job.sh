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
WANDB_RUN_NAME="ppo_run"
source ~/.bash_profile

# Log the IP address
echo "Determined host IP: $HOST_IP"

model_path=Qwen/Qwen2.5-0.5B-Instruct

CONFIG_NAME=ppo_trainer_mt
datetime=$(date +"%Y%m%d_%H%M%S")
export SSL_CERT_FILE=../cacert.pem




# Training config

apptainer exec --nv \
	"${CONTAINER_PATH}" \
	python3 -m verl.trainer.main_ppo \
	data.train_files=data/gsm8k/train.parquet \
	data.val_files=data/gsm8k/test.parquet \
	data.train_batch_size=256 \
	data.max_prompt_length=512 \
	data.max_response_length=512 \
	actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=64 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
	actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
	critic.optim.lr=1e-5 \
	critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
	critic.ppo_micro_batch_size_per_gpu=4 \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.logger=console \
	trainer.val_before_train=False \
	trainer.n_gpus_per_node=4 \
	trainer.nnodes=1 \
	trainer.save_freq=100 \
	trainer.test_freq=10 \
	trainer.total_epochs=15 2>&1 | tee verl_demo.log
