#!/usr/bin/env bash
# Step 1: Baseline SRT on the 100-sample DAPO test set.
# Verifies reward hacking: accuracy collapses while pseudo-reward spikes.
#
# Before running:
#   python scripts/extract_dapo_100.py   # creates train_100.parquet
#
# Usage:
#   bash experiment_scripts/srt_baseline_100.sh 2>&1 | tee baseline_100.txt

conda activate online_rl

# --- Batch sizes (smaller for 100 samples) ---
export FULL_BATCH_SIZE=8           # 8 prompts per batch × 16 rollouts = 128 responses
export PPO_MINI_BATCH_SIZE=8
export PER_GPU_MINI_BATCH_SIZE=16

# --- Rollouts (plan says 16 per prompt) ---
export NUM_PER_PROMPT_ROLLOUTS=16
export NUM_PER_PROMPT_ROLLOUTS_VALIDATION=16

# --- Sequence lengths ---
export MAX_RESPONSE_LENGTH=3072
export MAX_PROMPT_LENGTH=1024

# --- Optimizer ---
export LEARNING_RATE=1e-6
export KL_COEFF=0.001

# --- SRT (baseline: no entropy) ---
export REWARD_MANAGER='self_learning'
export LOG_THRESHOLD_PLOT=True
export SELF_CONSISTENCY_THRESHOLD=0.0
export SOFT_REWARD=False
export REMOVE_KL_LOSS_FROM_UNLABELLED_EXAMPLES=True
export OVERSAMPLING_KEEP_FRACTION=1.0

# --- Data: 100-sample subset ---
TRAIN_DATASET_PATH=$HOME/data/dapo_unlabeled/train_100.parquet
TEST_DATASET_PATH=$HOME/data/srt_test_dataset/test.parquet

# --- Run longer to observe collapse ---
TOTAL_EPOCHS=3

MODEL_PATH=Qwen/Qwen2.5-Math-7B
PROJECT_NAME=Qwen2.5_Math_7B_100samples
EXPERIMENT_NAME=SRT_baseline_100

SAVE_FREQ=-1
TEST_FREQ=20    # test every 20 steps (frequent — 100 samples trains fast)

CHECKPOINT_SAVE_PATH=$HOME/self_labeling_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
rm -rf $CHECKPOINT_SAVE_PATH

export VLLM_ATTENTION_BACKEND=XFORMERS
export SEED=42

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$TRAIN_DATASET_PATH \
    data.val_files=$TEST_DATASET_PATH \
    data.train_batch_size=$FULL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=$KL_COEFF \
    reward_model.reward_manager=$REWARD_MANAGER \
    reward_model.self_consistency_threshold=$SELF_CONSISTENCY_THRESHOLD \
    reward_model.soft_reward=$SOFT_REWARD \
    reward_model.remove_kl_loss_from_unlabeled_examples=$REMOVE_KL_LOSS_FROM_UNLABELLED_EXAMPLES \
    reward_model.oversampling_keep_fraction=$OVERSAMPLING_KEEP_FRACTION \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.log_threshold_plot=$LOG_THRESHOLD_PLOT \
    ray_dir='/tmp' $@
