#!/bin/bash
#SBATCH --job-name=convit_ensemble-fixed-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=/home/mila/d/diganta.misra/scratch/fair_moe/convit_fixed_seed_2.out
#SBATCH --error=/home/mila/d/diganta.misra/scratch/fair_moe/convit_fixed_seed_2.err
#SBATCH --no-requeue

# ulimit -Sn $(ulimit -Hn)
module load libffi
module load anaconda/3
conda activate /home/mila/d/diganta.misra/.conda/envs/ffcv_eg

wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

ulimit -Sn $(ulimit -Hn)

unset CUDA_VISIBLE_DEVICES
# unset LOCAL_RANK
WANDB_CACHE_DIR=$SCRATCH

pyfile=/home/mila/d/diganta.misra/projects/fair_ensemble_moe/CIFAR100/train_cifar100.py
seed=2
model_name=convit
yaml_pth=/home/mila/d/diganta.misra/projects/fair_ensemble_moe/CIFAR100/default_config_${model_name}.yaml

# Run fixed seeds for Model Init and Batch Order (enforce random seeds for Data Augmentation)
python $pyfile --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "MS_BS" --seed.DA $seed
