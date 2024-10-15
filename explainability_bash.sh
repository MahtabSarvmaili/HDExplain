#!/bin/bash


for i in {1..30}
do
    python explainability.py -data CIFAR10 -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i
    python explainability.py -data CIFAR10 -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i --scale
    python explainability.py -data CIFAR10 -n_classes 10 -network ResNet -explainer RPS --gpu -seed $i
    python explainability.py -data CIFAR10 -n_classes 10 -network ResNet -explainer IF --gpu -seed $i --scale
    python explainability.py -data CIFAR10 -n_classes 10 -network ResNet -explainer TracIn --gpu -seed $i --scale
done

for i in {1..30}
do
    python explainability.py -data OCEA -n_classes 5 -network ResNet -explainer YADEA --gpu -seed $i
    python explainability.py -data OCEA -n_classes 5 -network ResNet -explainer YADEA --gpu -seed $i --scale
    python explainability.py -data OCEA -n_classes 5 -network ResNet -explainer RPS --gpu -seed $i
    python explainability.py -data OCEA -n_classes 5 -network ResNet -explainer IF --gpu -seed $i --scale
    python explainability.py -data OCEA -n_classes 5 -network ResNet -explainer TracIn --gpu -seed $i --scale
done

for i in {1..30}
do
    python explainability.py -data MRI -n_classes 4 -network ResNet -explainer YADEA --gpu -seed $i
    python explainability.py -data MRI -n_classes 4 -network ResNet -explainer YADEA --gpu -seed $i --scale
    python explainability.py -data MRI -n_classes 4 -network ResNet -explainer RPS --gpu -seed $i
    python explainability.py -data MRI -n_classes 4 -network ResNet -explainer IF --gpu -seed $i --scale
    python explainability.py -data MRI -n_classes 4 -network ResNet -explainer TracIn --gpu -seed $i --scale
done

for i in {1..30}
do
    python explainability.py -data SVHN -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i
    python explainability.py -data SVHN -n_classes 10 -network ResNet -explainer YADEA --gpu -seed $i --scale
    python explainability.py -data SVHN -n_classes 10 -network ResNet -explainer RPS --gpu -seed $i
    python explainability.py -data SVHN -n_classes 10 -network ResNet -explainer IF --gpu -seed $i --scale
    python explainability.py -data SVHN -n_classes 10 -network ResNet -explainer TracIn --gpu -seed $i --scale
done