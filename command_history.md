# Classifier_Explaination_via_KSD
===

  - Install the required packages from the environment.yml file
    - Change the prefix to your desired location
    - ensure that cuda version is compatible with the pytorch version
  - Optional: Train your model using the train.py file
  - Test the explanation methods for real-world datasets (CIFAR10 - SVHN) using explain_real.py
  - Test the explanation methods for synthetic datasets (Gaussian, Uniform, and Moons) using explain.py
  - Test the explanation methods for the HitRate and Coverate using explainability.py
  - Test the influence of temprature on the explanation methods using temperature.py
  - Test the impact of kernel methods on the explanation methods using kernel.py 

The bash file for every corresponding python file is provided. 
To replicate the results, run all of the bash files in the terminal and the plots, tables folders.
Finally run the jupyter notebooks (table_summary.ipynb, table_summary_noise.ipynb) to plot the results.



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

Some graph visualization tool
https://stackoverflow.com/questions/13513455/drawing-a-graph-or-a-network-from-a-distance-matrix
https://stackoverflow.com/questions/53967392/creating-a-graph-with-images-as-nodes
https://stackoverflow.com/questions/75110767/position-of-images-as-nodes-in-networkx-plot
https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html
https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/code
https://www.kaggle.com/datasets/shreyag1103/brain-mri-scans-for-brain-tumor-classification/data
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data