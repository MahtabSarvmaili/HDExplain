#!/bin/bash

python cifar10.py -data MRI -n_classes 4 --gpu -explainer YADEA
python cifar10.py -data MRI -n_classes 4 --gpu --scale -explainer YADEA
python cifar10.py -data MRI -n_classes 4 --gpu --scale -explainer IF
python cifar10.py -data MRI -n_classes 4 --gpu -explainer RPS
python cifar10.py -data MRI -n_classes 4 --gpu --scale -explainer TracIn