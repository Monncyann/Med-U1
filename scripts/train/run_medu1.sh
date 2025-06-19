#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# # Set default model path if not provided
# if [ -z "$MODEL_PATH" ]; then
#     MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"
# fi

datetime=$(date +"%Y%m%d%H%M%S")
echo $datetime

dataset=medoption
base_model=Qwen2.5-3B-Instruct
train_batch_size=4
reward_metric=Merge # Only be effective in open-ended tasks
exp_name=${datetime}_bs${train_batch_size}_dataset${dataset}_model${base_model}_reward_metric${reward_metric}
# Train over a single node, 8 A100-80GB GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/Med-U1/scripts/data/processed_data_normal/train_${dataset}.parquet \
    data.val_files=/Med-U1/scripts/data/processed_data_normal/${dataset}.parquet \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/Med-U1/model/Qwen2.5-3B-Instruct\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    algorithm.reward_type='discrete' \
    algorithm.reward_metric=${reward_metric} \
    algorithm.reward_continuous_scale=100 \
    algorithm.check_think=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Med-U1' \
    trainer.experiment_name=${exp_name} \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${exp_name} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=450 \
    trainer.test_freq=350 \
    trainer.total_epochs=1 "${@:1}"