
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

We used MATLAB to implement MLE/MAP.  Please make sure install the following toolkit before running the MATLAB code.
```matlab
% - Optimization Toolbox (fmincon)
% - Global Optimization Toolbox
% - Parallel Computing Toolbox (for parallel execution)
```

## Training LaseNet
- Generate simulation data by running the notebook `simulation/simulate.ipynb`. The generated data will be saved to `data` folder. We also include sample data in `zip` file for Meta RL and GLM-HMM models.
- Train the LastNet by running the notebook `training.ipynb`. The notebook shows how we train and fine-tune LaseNet for differet computational cognitive models. The generated model will be saved to `results` folder.


## Evaluation

- Generate results with benchmark methods (likelihood-dependent)
- Compare the LastNet and bechmark methods by running the notebook `eval.ipynb`

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models from the Google Drive here:

- [Model](https://drive.google.com/drive/folders/1--Ywm9IQbv0Z7B160Udi81y4uVKLWAJ-?usp=sharing) trained on 4-P RL with 9000 simulated agents and 720 trials per agent.
- [Model](https://drive.google.com/drive/folders/1-QzmG81fyu8hQWWfL3BwllzEA4l9f1ne?usp=sharing) trained on Meta RL with 9000 simulated agents and 720 trials per agent.
- [Model](https://drive.google.com/drive/folders/11CdQDc5JUvMCWhUA38zup4e2UTAgVjns?usp=sharing) trained on HRL with 9000 simulated agents and 720 trials per agent.
- [Model](https://drive.google.com/drive/folders/1-0kDCGjrppynMczjJt6Uq3Z-jCES-cTr?usp=sharing) trained on GLM-HMM with 9000 simulated agents and 720 trials per agent.


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
