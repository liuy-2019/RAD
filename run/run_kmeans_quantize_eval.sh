export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune.py     \
 -a resnet50                     \
 --data_name cifar10             \
 --workers 32                    \
 --test_batch 512                \
 --free_high_bit False           \
 --eval                          \
 --resume /home/liuyang/workspace/HAQ/haq/Chen2020Adversarial.pt_m0.pt        
