#!/bin/bash

#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name srt_alpha0.5
#SBATCH --output job-%J.out
#SBATCH --gpus=1
#SBATCH --mail-user=yuvalzoh@post.bgu.ac.il
#SBATCH --mail-type=ALL

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda
source activate yuval_rl
cd ~/RL_Project/mini_entropy_srt

python -u lightweight_train.py \
    --alpha 0.5 --n_steps 300 --n_rollouts 8 \
    --eval_every 15 --n_eval_questions 20 \
    --train_parquet ~/data/dapo_unlabeled/train_curated.parquet \
    --output results/day3_alpha0.5.json
