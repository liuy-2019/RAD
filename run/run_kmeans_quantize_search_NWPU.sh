export CUDA_VISIBLE_DEVICES=1
python -W ignore rl_quantize.py     \
 --arch resnet50                    \
 --dataset NWPU-RESISC45       \
 --suffix ratio010                  \
 --preserve_ratio 0.1               \
 --n_worker 32                      \
 --data_bsize 32                    \
 --train_size 20000                 \
 --val_size 10000                   \
 --finetune_epoch 1                 \
 --debug                            \
 --prune_method prune               \
 --prune_level 0.99