#!/bin/bash

 python explain_real.py -data CIFAR10 -n_classes 10 --gpu -explainer HDEXPLAIN
 python explain_real.py -data CIFAR10 -n_classes 10 --gpu --scale -explainer HDEXPLAIN
 python explain_real.py -data CIFAR10 -n_classes 10 --gpu --scale -explainer IF
 python explain_real.py -data CIFAR10 -n_classes 10 --gpu -explainer RPS
 python explain_real.py -data CIFAR10 -n_classes 10 --gpu --scale -explainer TracIn

 python explain_real.py -data MRI -n_classes 4 --gpu -explainer HDEXPLAIN
 python explain_real.py -data MRI -n_classes 4 --gpu --scale -explainer HDEXPLAIN
 python explain_real.py -data MRI -n_classes 4 --gpu --scale -explainer IF
 python explain_real.py -data MRI -n_classes 4 --gpu -explainer RPS
 python explain_real.py -data MRI -n_classes 4 --gpu --scale -explainer TracIn


python explain_real.py -data OCEA -n_classes 5 --gpu -explainer HDEXPLAIN
python explain_real.py -data OCEA -n_classes 5 --gpu --scale -explainer HDEXPLAIN
python explain_real.py -data OCEA -n_classes 5 --gpu --scale -explainer IF
python explain_real.py -data OCEA -n_classes 5 --gpu -explainer RPS
python explain_real.py -data OCEA -n_classes 5 --gpu --scale -explainer TracIn

python explain_real.py -data SVHN -n_classes 10 --gpu -explainer HDEXPLAIN
python explain_real.py -data SVHN -n_classes 10 --gpu --scale -explainer HDEXPLAIN
python explain_real.py -data SVHN -n_classes 10 --gpu --scale -explainer IF
python explain_real.py -data SVHN -n_classes 10 --gpu -explainer RPS
python explain_real.py -data SVHN -n_classes 10 --gpu --scale -explainer TracIn