DATA_ROOT=$1
SEED=$2

python gdro/group_dro.py --name gdro_resnet50_lr_1e-5_wd_1_epoch_300_b2t_seed_$SEED \
--dataset cub --image_size 224 \
--data_root DATA_ROOT \
--model resnet50 --pretrained imagenet --num_classes 2 \
--epochs 300 --lr 1e-5 --weight-decay 1 --seed $SEED \
--pseudo_bias pseudo_bias/waterbirds.pt