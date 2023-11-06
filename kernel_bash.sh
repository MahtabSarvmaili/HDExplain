#!/bin/bash


# for i in {1..30}
# do  
#     python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel RBF
#     python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel IMQ
#     python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel Linear
# done


# for i in {1..30}
# do  
#     python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel RBF
#     python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel IMQ
#     python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel Linear
# done

for i in {1..30}
do  
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel RBF
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel IMQ
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel Linear
done