
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

## Training
- Generate simulation data by running the notebook `simulation/simulate.ipynb`. The generated data will be saved to `data` folder. We also include sample data in `zip` file for Meta RL and GLM-HMM models.
- Train the LastNet by running the notebook `training.ipynb`. The notebook shows how we train and fine-tune LaseNet for differet computational cognitive models. The generated model will be saved to `results` folder.


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

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
