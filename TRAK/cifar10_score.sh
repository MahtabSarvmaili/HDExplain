#!/bin/bash

for i in {1..20}
do
    mkdir "iteration_$i"  # Creates a directory named "iteration_<current_iteration_number>"
    python cifar_train.py -data SVHN --resultpath "iteration_$i" -n_classes 10 --gpu -explainer YADEA
    python cifar_train.py -data SVHN --resultpath "iteration_$i" -n_classes 10 --gpu --scale -explainer YADEA
    python cifar_train.py -data SVHN --resultpath "iteration_$i" -n_classes 10 --gpu --scale -explainer IF
    python cifar_train.py -data SVHN --resultpath "iteration_$i" -n_classes 10 --gpu -explainer RPS
    python cifar_train.py -data SVHN --resultpath "iteration_$i" -n_classes 10 --gpu --scale -explainer TracIn
done