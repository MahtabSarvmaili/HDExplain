#!/bin/bash


for i in {1..15}
do  
    python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel RBF
    python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel IMQ
    python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel Linear
    python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel RBF --scale
    python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel IMQ --scale
    python kernel.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -kernel Linear --scale
done


for i in {1..15}
do  
    python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel RBF
    python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel IMQ
    python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel Linear
    python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel RBF --scale
    python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel IMQ --scale
    python kernel.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -kernel Linear --scale
done

for i in {1..15}
do  
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel RBF
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel IMQ
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel Linear
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel RBF --scale
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel IMQ --scale
    python kernel.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -kernel Linear --scale
done