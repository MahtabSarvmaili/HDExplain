#!/bin/bash


for i in {1..15}
do
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.03
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.06
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.1
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.3
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.6
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 1.0
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 3.0
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 6.0
done


for i in {1..15}
do
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.03
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.06
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.1
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.3
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.6
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 1.0
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 3.0
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 6.0
done

for i in {1..15}
do
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.03
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.06
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.1
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.3
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.6
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 1.0
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 3.0
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 6.0
done

for i in {1..15}
do
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.03 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.06 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.1 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.3 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.6 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 1.0 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 3.0 --scale
    python temperature.py -data CIFAR10 -n_classes 10 -network ResNet --gpu -seed $i -temperature 6.0 --scale
done

for i in {1..15}
do
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.03 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.06 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.1 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.3 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 0.6 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 1.0 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 3.0 --scale
    python temperature.py -data OCEA -n_classes 5 -network ResNet --gpu -seed $i -temperature 6.0 --scale
done

for i in {1..15}
do
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.03 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.06 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.1 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.3 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 0.6 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 1.0 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 3.0 --scale
    python temperature.py -data MRI -n_classes 4 -network ResNet --gpu -seed $i -temperature 6.0 --scale
done

for i in {1..15}
do
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.03 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.06 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.1 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.3 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 0.6 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 1.0 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 3.0 --scale
    python temperature.py -data SVHN -n_classes 10 -network ResNet --gpu -seed $i -temperature 6.0 --scale
done