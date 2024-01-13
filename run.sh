TASK=sentiment
SEED="123"
GPU=4
EPOCHS=10
BATCH_SIZE=16
BACKBONE=roberta-large

## Baseline
#CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type baseline --adv_eps 0.0 --coeff_sym 0.0 --model_lr 1e-5 --batch_size $BATCH_SIZE --epochs $EPOCHS --task $TASK --seed $SEED --backbone $BACKBONE

## Roast for Sentiment 
CUDA_VISIBLE_DEVICES=$GPU python train.py --roast --alpha 0.9  --beta 10 --unbiased_scale --train_type roast_test --adv_eps 1e-1 --model_lr 1e-5 --coeff_sym 0.01 --task $TASK --seed $SEED --backbone $BACKBONE

## Roast for Entailment 
#CUDA_VISIBLE_DEVICES=$GPU python train.py --roast --alpha 0.6  --beta 1 --unbiased_scale --train_type roast_test --adv_eps 1e-1 --model_lr 1e-5 --coeff_sym 0.01 --task $TASK --seed $SEED --backbone $BACKBONE
