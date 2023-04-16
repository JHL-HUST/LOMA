# LOMA

## ImageNet

### Command
- LOMA_IF&FO:

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup python train_ddp_feature.py --net_type resnet --dataset imagenet --batch_size 256 --lr 0.1 --depth 50 --epochs 300 --expname ResNet50 --dist-url 'tcp://127.0.0.1:33395' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0  --feature_p 0.5 --flag 2 --scale_turn --offset_turn --off_pro 0.5 --data_path data_path > output.txt &`

## CIFAR

### ResNet-20 

- Baseline
`python main_feature.py --dataset cifar10 --arch resnet_feature  --depth 20  -c checkpoint_path  --p 0.0  --feature_p 0.0  --flag 2 --gpu gpu_id`
- LOMA_I
`python main_feature.py --dataset cifar10 --arch resnet_feature  --depth 20  -c checkpoint_path  --p 0.5  --feature_p 0.0  --flag 2 --gpu gpu_id`
- LOMA_F
`python main_feature.py --dataset cifar10 --arch resnet_feature  --depth 20  -c checkpoint_path  --p 0.0  --feature_p 0.5  --scale_turn --flag 2 --gpu gpu_id`
- FO
`python main_feature.py --dataset cifar10 --arch resnet_feature  --depth 20  -c checkpoint_path  --p 0.0  --feature_p 0.5  --offset_turn --off_pro 0.5 --flag 2 --gpu gpu_id`
- LOMA_IF&FO
`python main_feature.py --dataset cifar10 --arch resnet_feature  --depth 20  -c checkpoint_path  --p 0.5  --feature_p 0.5  --scale_turn --offset_turn --off_pro 0.5 --flag 2 --gpu gpu_id`


### WideResNet-28-10
- Baseline
`python main_feature.py --dataset cifar10 --arch wrn_feature  --depth 28 --widen-factor 10  -c checkpoint_path  --p 0.0  --feature_p 0.0  --flag 2 --gpu gpu_id`
- LOMA_IF&FO
`python main_feature.py --dataset cifar10 --arch wrn_feature  --depth 28 --widen-factor 10  -c checkpoint_path  --p 0.5  --feature_p 0.5  --scale_turn --offset_turn --off_pro 0.5 --flag 2 --gpu gpu_id`
- Others are similar to ResNet-20

### PyramidNet
- Baseline
`CUDA_VISIBLE_DEVICES=0,1  python train.py --net_type pyramidnet --dataset cifar100 --depth 200 --alpha 240 --batch_size 64 --lr 0.25 --expname PyraNet200 --epochs 300  --p 0.0 --feature_p 0.0`
- LOMA_IF&FO
`CUDA_VISIBLE_DEVICES=0,1  python train.py --net_type pyramidnet --dataset cifar100 --depth 200 --alpha 240 --batch_size 64 --lr 0.25 --expname PyraNet200 --epochs 300  --p 0.5 --feature_p 0.5  --scale_turn --offset_turn --off_pro 0.5 --flag 2`


## To change the preset deformation shape, modify in file LOMA_I.py and LOMA_F.py

