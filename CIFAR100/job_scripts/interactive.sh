#!/bin/bash
echo arg 1 - 'Path of python script'
echo arg 2 - 'Name of model'
echo arg 3 - '(Optional) config yaml path, defaults to default_config_<model_name>.yaml'

pyfile={$1:-"train_cifar100.py"}
model_name=$2
default_yaml=default_config_${model_name}.yaml
yaml_pth=${3:-$default_yaml}
seed=${4:-"1"}

if [[ $model_name == *"moe"* ]]; then
    python $pyfile --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "RANDOM" --seed.DA $seed --seed.model_init $seed --seed.batch_order $seed
else
    # Run all random seeds (enforce random seeds for Data Augmentation, Model Init, and Batch Order)
    python $pyfile --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "RANDOM" --seed.DA $seed --seed.model_init $seed --seed.batch_order $seed
    # Run fixed seeds for Model Init and Batch Order (enforce random seeds for Data Augmentation)
    python $pyfile --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "MS_BS" --seed.DA $seed
fi
