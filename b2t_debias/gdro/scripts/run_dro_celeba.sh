DATA_ROOT=$1
SEED=$2

python gdro/group_dro.py --name gdro_resnet50_lr_1e-5_wd_1e-1_epoch_50_seed_$SEED \
--dataset celeba --image_size 224 \
--data_root $DATA_ROOT \
--model resnet50 --pretrained imagenet --num_classes 2 \
--epochs 50 --lr 1e-5 --weight-decay 1e-1 --seed $SEED