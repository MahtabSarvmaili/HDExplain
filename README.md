[]: # (start)

# Classifier Explanation via Kernelized Stein Discrepancy (KSD)

This repository provides the implementation for the paper:

**"Data-centric Prediction Explanation via Kernelized Stein Discrepancy"**  
by Mahtab Sarvmaili, Hassan Sajjad, and Ga Wu  
Published on [arXiv](https://arxiv.org/abs/2403.15576).  

If you use this work in your research, please cite:

```bibtex
@article{sarvmaili2024data,
  title={Data-centric Prediction Explanation via Kernelized Stein Discrepancy},
  author={Sarvmaili, Mahtab and Sajjad, Hassan and Wu, Ga},
  journal={arXiv preprint arXiv:2403.15576},
  year={2024}
}
```

---

## Overview

This repository provides tools to explain classifier predictions using the Kernelized Stein Discrepancy (KSD) approach. It supports experiments on both real-world datasets (e.g., CIFAR10, SVHN) and synthetic datasets (e.g., Gaussian, Uniform, Moons).

Key features include:
- Training custom models
- Explaining predictions with various kernel methods
- Visualizing the impact of hyperparameters
- Evaluating metrics like HitRate and Coverage

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Classifier_Explaination_via_KSD.git
   cd Classifier_Explaination_via_KSD
   ```

2. Create a Conda environment using the `environment.yml` file:

>> Attention: Adjust the prefix in environment.yml file according to your system.

   ```bash
   conda env create -f environment.yml
   conda activate ood
   ```
   - **Note**: Ensure that the CUDA version matches the PyTorch version in your environment.

---

## Usage

### 1. Training a Model
Train a classifier using:
```bash
python train.py
```
Add `-r` to resume training.

### 2. Explaining Predictions on Real-World Datasets
Test explanation methods on CIFAR10 or SVHN:
```bash
python explain_real.py
```

### 3. Explaining Predictions on Synthetic Datasets
Analyze explanations for synthetic datasets (Gaussian, Uniform, Moons):
```bash
python explain.py
```

### 4. Evaluating HitRate and Coverage
Run:
```bash
python explainability.py
```

### 5. Testing the Effect of Temperature
Evaluate the influence of temperature on explanations:
```bash
python temperature.py
```

### 6. Analyzing Kernel Methods
Explore the impact of kernel methods:
```bash
python kernel.py
```

### Bash Scripts
Predefined bash scripts are available for each experiment. Run them to replicate the results.

---

## Results

- **Plots**: All plots will be saved in the `plots/` directory.
- **Tables**: Generated tables are stored in the `tables/` directory.
>> Attention: Adjust the directory paths in the bash files to save the results in the desired location.
- **Summarizing Results**: Use the Jupyter notebooks:
  - `table_summary.ipynb`
  - `table_summary_noise.ipynb`

---

## Visualization Tools

To visualize the explanation graphs:
- [Distance Matrix-based Graphs](https://stackoverflow.com/questions/13513455/drawing-a-graph-or-a-network-from-a-distance-matrix)
- [Network Graphs with Images](https://stackoverflow.com/questions/53967392/creating-a-graph-with-images-as-nodes)

---

## Example Datasets

Below are some useful datasets for testing and experimentation:
- [Bank Marketing Dataset](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- [Heart Failure Clinical Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/code)
- [Brain MRI for Tumor Classification](https://www.kaggle.com/datasets/shreyag1103/brain-mri-scans-for-brain-tumor-classification/data)
- [Credit Card Approval Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data)

---

## Contributing

Contributions are welcome! Please create an issue or submit a pull request.

---

## License

This project is licensed under the Apache License 2.0. 

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[]: # (end)
