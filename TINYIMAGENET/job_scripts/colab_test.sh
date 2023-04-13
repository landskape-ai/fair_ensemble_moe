

# wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

# ulimit -Sn $(ulimit -Hn)

# unset CUDA_VISIBLE_DEVICES
# # unset LOCAL_RANK
# WANDB_CACHE_DIR=$SCRATCH

pyfile=/content/fair_ensemble_moe/TINYIMAGENET/train_tinyimagenet.py
seed=1
model_name=vit
yaml_pth=/content/fair_ensemble_moe/TINYIMAGENET/default_config_${model_name}.yaml

python $pyfile --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "RANDOM" --seed.DA $seed --seed.model_init $seed --seed.batch_order $seed
