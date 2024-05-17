import ssm
from ssm.util import find_permutation


def fit_glmhmm_em(prior_config: dict, n_session, inpts, true_choices, true_latent=[]):
    # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    N_iters = 200

    obs_dim = 1  # number of observed dimensions
    num_categories = len(true_choices.unique())  # number of categories for output
    input_dim = inpts.shape[-1]  # input dimensions

    # Instantiate GLM-HMM and set prior hyperparameters
    # Gaussian prior hyperparameter for the GLM weights
    prior_sigma = prior_config["prior_sigma"] if "prior_sigma" in prior_config else 2
    # Dirichlet prior hyperparameter for the transition matrix
    transition_alpha = (
        prior_config["transition_alpha"] if "transition_alpha" in prior_config else 1
    )
    num_state = prior_config["num_state"] if "num_state" in prior_config else 3

    all_recovered_weights, posterior_probs = [], []
    for i in range(n_session):
        map_glmhmm = ssm.HMM(
            num_state,
            obs_dim,
            input_dim,
            observations="input_driven_obs",
            verbose=0,
            observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
            transitions="sticky",
            transition_kwargs=dict(alpha=transition_alpha, kappa=0),
        )

        # Fit GLM-HMM with MAP estimation:
        choice, input = true_choices[i], inpts[i]
        _ = map_glmhmm.fit(
            choice, inputs=input, method="em", num_iters=N_iters, tolerance=10**-4
        )
        map_final_ll = map_glmhmm.log_likelihood(choice, inputs=input)
        print(i, " ll:", map_final_ll)
        # If there is true latent such as testing in simulated dataset, try to find the best state matched sequence.
        if len(true_latent) > 0:
            map_glmhmm.permute(
                find_permutation(
                    true_latent[i],
                    map_glmhmm.most_likely_states(choice, input=input),
                )
            )

        recovered_weights = -map_glmhmm.observations.params
        all_recovered_weights.append(recovered_weights)

        posterior_prob = map_glmhmm.expected_states(data=choice, input=input)[0]
        posterior_probs.append(posterior_prob)

    return posterior_probs, all_recovered_weights
