# Benchmark methods - likelihood-dependent
This folder includes three commonly used methods to infer latent variables in computational cognitive modeling: e Maximum Likelihood Estimatio (MLE), Maximum a Posterior (MAP), and Expectation-Maximization (EM).

The procedures of recovering latent variables with the benchmark methods are as below. Note that there is no existing method for HRL.

## 4P-RL
1. Recover parameters by running either MLE `benchmarks/mle/fitPrlMLE.m` or MAP `benchmarks/map/fitPrlMAP.m`. We implemented MLE/MAP in MATLAB given its rich toolkits in statistical computation. 
2. Infer latent variables with `benchmarks/get_4prl_latent.py`

## Meta RL
The process for Meta RL is based on the [code](https://github.com/jl3676/dynamic_noise_estimation/tree/main/Dynamic_Foraging/code) from [Li et.al. 2024](https://doi.org/10.1016/j.jmp.2024.102842)
1. Recover parameters by running [recover_params.m](https://github.com/jl3676/dynamic_noise_estimation/blob/main/Dynamic_Foraging/code/recover_params.m). The MLE is adopted to recover the parameters here.
2. Infer latent variables by running [recover_latent_probs.m](https://github.com/jl3676/dynamic_noise_estimation/blob/main/Dynamic_Foraging/code/recover_latent_probs.m)

## GLM-HMM
`benchmarks/em/fit_glmhmm_em.py` fits a GLM-HMM with EM, that is based on the approximate EM algorithm used in [Ashwood et al. (2020)](https://www.biorxiv.org/content/10.1101/2020.10.19.346353v1.full.pdf). The original implementation can be found at this [notebook](https://github.com/lindermanlab/ssm/blob/master/notebooks/2b-Input-Driven-Observations-(GLM-HMM).ipynb).

`fit_glmhmm_em` function in `benchmarks/em/fit_glmhmm_em.py` performs the following steps:
1. Recover GLM weights and transition matrix. 
2. Infer latent variable with recovered parameters and input/choice via E-step.