export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune.py     \
 -a qmobilenetv2                 \
 --data_name cifar10             \
 --workers 32                    \
 --test_batch 512                \
 --gpu_id 1                      \
 --free_high_bit False           \
 --linear_quantization           \
 --eval                          \


 #--resume checkpoints/mobilenetv2/qmobilenetv2_0.6_71.23.pth.tar        \