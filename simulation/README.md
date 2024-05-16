# Generate simulated data
This folder contains implementation to simulates data from cognive computaional models: 4-P RL, HRL, and GLM-HMM

We use version 0.0.1 of the Bayesian State Space Modeling framework from Scott Linderman's lab to perform GLM-HMM inference. Within the conda `lasenet` environment, install the forked version of the [ssm](https://github.com/zashwood/ssm) package. Install this version of ssm with the following commands:
```
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e .
```

To simulate data from meta RL with dyanmic noise, we use the open source [code](https://github.com/jl3676/dynamic_noise_estimation/tree/main/Dynamic_Foraging/code) from [Li et.al. 2024](https://doi.org/10.1016/j.jmp.2024.102842)

