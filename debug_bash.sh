#!/bin/bash


# for i in {1..15}
# do  
#     python debug.py -data CIFAR10 -n_classes 10 -network ResNet -explainer IF --gpu -seed $i --scale --subsample
#     python debug.py -data CIFAR10 -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i --subsample
#     python debug.py -data CIFAR10 -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i --scale --subsample
#     python debug.py -data CIFAR10 -n_classes 10 -network ResNet -explainer RPS --gpu -seed $i --subsample
# done

# for i in {1..15}
# do  
#     python debug.py -data OCEA -n_classes 5 -network ResNet -explainer IF --gpu -seed $i --scale --subsample
#     python debug.py -data OCEA -n_classes 5 -network ResNet -explainer YADEA --gpu -seed $i --subsample
#     python debug.py -data OCEA -n_classes 5 -network ResNet -explainer YADEA --gpu -seed $i --scale --subsample
#     python debug.py -data OCEA -n_classes 5 -network ResNet -explainer RPS --gpu -seed $i --subsample
# done

#for i in {1..15}
#do
#    python debug.py -data MRI -n_classes 4 -network ResNet -explainer IF --gpu -seed $i --scale
#    python debug.py -data MRI -n_classes 4 -network ResNet -explainer YADEA --gpu -seed $i
#    python debug.py -data MRI -n_classes 4 -network ResNet -explainer YADEA --gpu -seed $i --scale
#    python debug.py -data MRI -n_classes 4 -network ResNet -explainer RPS --gpu -seed $i
#done

for i in {1..15}
do
    python debug.py -data SVHN -n_classes 10 -network ResNet -explainer IF --gpu -seed $i --scale
    python debug.py -data SVHN -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i
    python debug.py -data SVHN -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i --scale
    python debug.py -data SVHN -n_classes 10 -network ResNet -explainer RPS --gpu -seed $i
done