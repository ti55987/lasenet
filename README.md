
# Latent Variable Sequence Identification for Cognitive Models with Neural Bayes Estimation

This repository is the official implementation of LaseNet: Latent Variable Sequence Identification for Cognitive Models with Neural Bayes Estimation
<!-- add a paper link optional.  -->


## Requirements

To install requirements, first create a conda environment from the environment yaml file by running

```setup
conda env create -f environment.yml
```
Activate this environment by running
```
conda activate lasenet
```
### Optional setup
To experiment with GLM-HMM, we use version 0.0.1 of the Bayesian State Space Modeling framework from Scott Linderman's lab to perform GLM-HMM inference. Within the conda `lasenet` environment, install the forked version of the [ssm](https://github.com/zashwood/ssm) package. Install this version of ssm with the following commands:
```setup
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e .
```

We used MATLAB to implement MLE/MAP.  Please install the following toolkits before running the MATLAB code.
```matlab
% - Optimization Toolbox (fmincon)
% - Global Optimization Toolbox
% - Parallel Computing Toolbox (for parallel execution)
```

## Training LaseNet
1. Generate simulation data by running the notebook `simulation/simulate.ipynb`. The generated data will be saved to `data` folder. We also include sample data in `zip` file for Meta RL and GLM-HMM models.
2. Train the LastNet by running the notebook `training.ipynb`. The notebook shows how we train and fine-tune LaseNet for differet computational cognitive models. The generated model will be saved to `results` folder.


## Evaluation

Compare the LastNet and bechmark methods by running the notebook `eval.ipynb`. The notebook includes evaluation sample for 4-P RL and HRL.

### Real data
The experiments on real mice data in this work can be reproduced in `mice_fitting/figures_real_data.ipynb`. `mice_fitting` folder includes code and mice data.

## Pre-trained Models

You can find all the pretrained models under `results/models`.  You can also download additional pretrained models hosted on the Google Drive:

- [Model](https://drive.google.com/drive/folders/1--Ywm9IQbv0Z7B160Udi81y4uVKLWAJ-?usp=sharing) trained on 4-P RL with 9000 simulated agents and 500 trials per agent.
- [Model](https://drive.google.com/drive/folders/1-QzmG81fyu8hQWWfL3BwllzEA4l9f1ne?usp=sharing) trained on Meta RL with 9000 simulated agents and 720 trials per agent.
- [Model](https://drive.google.com/drive/folders/11CdQDc5JUvMCWhUA38zup4e2UTAgVjns?usp=sharing) trained on HRL with 9000 simulated agents and 720 trials per agent.
- [Model](https://drive.google.com/drive/folders/1-0kDCGjrppynMczjJt6Uq3Z-jCES-cTr?usp=sharing) trained on GLM-HMM with 9000 simulated agents and 720 trials per agent.


## Results
The following results are the expected output when running `eval.ipynb`
![](results/hrl_rpe.png)
![](results/hrl_cf.png)

## Contributing
If you'd like to contribute, or have any suggestions for these guidelines, open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
