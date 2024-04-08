conda activate ood
  527  train.py
  528  python train.py
  529  python train.py -r
  530  cd workspace/
  531  conda activate ood
  532  cd UBC-OCEA-Preprocess/
  533  cd ..
  534  cd Classifier_Explaination_via_KSD/
  535  python cifar10.py -data OCEA -n_classes 5
  536  python cifar10.py -data OCEA -n_classes 5 --gpu
  537  cd workspace/
  538  cd Classifier_Explaination_via_KSD/
  539  python cifar10.py -data OCEA -n_classes 5 --gpu
  540  conda activate ood
  541  python cifar10.py -data OCEA -n_classes 5 --gpu
  542  cd workspace/
  543  cd Classifier_Explaination_via_KSD/
  544  python cifar10.py -data OCEA -n_classes 5 --gpu
  545  conda activate ood
  546  python cifar10.py -data OCEA -n_classes 5 --gpu
  547  conda activate ood
  548  cd workspace/Classifier_Explaination_via_KSD/
  549  python cifar10.py -data OCEA -n_classes 5 --gpu
  550  python cifar10.py -data OCEA -n_classes 5 --gpu -explainer RPS
  551  python cifar10.py -data OCEA -n_classes 5 --gpu -explainer TracIN
  552  python cifar10.py -data OCEA -n_classes 5 --gpu -explainer TracIn
  553  python cifar10.py -data OCEA -n_classes 5 --gpu -explainer TracIn --scale
  554  python cifar10.py -data OCEA -n_classes 5 --gpu -explainer IF --scale
