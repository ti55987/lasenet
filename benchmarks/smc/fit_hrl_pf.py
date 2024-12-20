import numpy as np
import scipy.special as special

class HRLParticleFilter:
    def __init__(self, alpha, beta, n_particles, n_states, initial_state_probs=None):
        """
        Initialize Particle Filter for HRLParticleFilter

        Args:
            alpha: Learning rate in HRL model
            beta: Softmax temperature in HRL model
            n_particles: Number of particles
            n_states: Number of hidden states/possible arrows
            initial_state_probs: Initial state probabilities
        """
        self.n_particles = n_particles
        self.n_states = n_states

        # Initialize initial state probabilities if not provided
        if initial_state_probs is None:
            self.initial_state_probs = np.ones(n_states) / n_states
        else:
            self.initial_state_probs = initial_state_probs

        # Initialize particles and weights
        self.particles = np.random.choice(n_states, size=n_particles,
                                        p=self.initial_state_probs)
        self.weights = np.ones(n_particles) / n_particles
        self.q_values = (1/n_states)*np.ones((n_particles, n_states))

        # Storage for state estimates
        self.state_estimates = []
        self.state_probabilities = []
        self.beta = beta
        self.alpha = alpha

    def predict_action(self, state, particle_idx):
        """Predict action probabilities for a given particle"""
        sftmx_p = special.softmax(self.beta * self.q_values[particle_idx])
        return sftmx_p

    def predict(self):
        """
        Predict step: propagate particles through transition model
        """
        new_particles = np.zeros(self.n_particles, dtype=int)
        for i in range(self.n_particles):
            # Sample new state from transition probabilities
            txp = self.predict_action(None, i)
            new_particles[i] = np.random.choice(self.n_states, p=txp)

        self.particles = new_particles

    def update(self, reward, action, stim):
        """
        Update step: update weights based on observation

        Args:
            observation: Current observation
        """
        # Find unique values, their first occurrence indices, and frequency counts
        unique_values, indices, counts = np.unique(stim, return_index=True, return_counts=True)

        # Filter to get values and indices that appear only once
        unique_action = unique_values[counts == 1]
        unique_arrow = indices[counts == 1]
        # Calculate emission probabilities for all particles
        for i in range(self.n_particles):
            bandit_probs = self.predict_action(None, i)
            arrow = self.particles[i]
            # not possible if chosen arrow direction is not the same as action
            if action != stim[arrow]:
              prob = 0
            elif action == unique_action: # following the unique side color
              prob = 1
            else:
              prob = bandit_probs[arrow]/(1-bandit_probs[unique_arrow])

            self.weights[i] *= prob

            # Calculate RPE for this particle
            rpe = reward - self.q_values[i, arrow]
            # Update Q-values using particle's learning rate
            self.q_values[i, arrow] += self.alpha * rpe
            # update the q value of other cues (counterfactual learning)
            others = np.array([x for x in list(np.arange(self.n_states)) if x != arrow ])
            self.q_values[i, others] += self.alpha*((1-reward)-self.q_values[i, others])

        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            # Handle numerical underflow
            self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate_state(self):
        """
        Estimate current state from particle distribution

        Returns:
            most_likely_state: Most likely current state
            state_probs: Probability distribution over states
        """
        state_counts = np.zeros(self.n_states)

        for i in range(self.n_particles):
            state_counts[self.particles[i]] += self.weights[i]

        state_probs = state_counts / np.sum(state_counts)
        most_likely_state = np.argmax(state_probs)

        return most_likely_state, state_probs

    def resample(self):
        """
        Resample particles based on their weights using systematic resampling
        """
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative_sum = np.cumsum(self.weights)

        # Handle numerical errors
        cumulative_sum[-1] = 1.0

        # Create new particles array
        new_particles = np.zeros(self.n_particles, dtype=int)

        i = 0  # positions index
        j = 0  # cumulative sum index

        while i < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1

        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def filter(self, reward, action, stimuli):
        """
        Perform one complete step of particle filtering

        Args:
            observation: Current observation

        Returns:
            most_likely_state: Most likely current state
            state_probs: Probability distribution over states
        """
        # Predict step
        self.predict()

        # Update step
        self.update(reward, action, stimuli)

        # Calculate effective sample size ESS = 1 / Σ(w_i^2)
        # This is a common heuristic used in particle filters to
        # determine when the particle diversity has become too low and
        # resampling is needed to maintain a good approximation of the
        # true state distribution.
        n_eff = 1.0 / np.sum(self.weights ** 2)

        # Resample if effective sample size is too low
        if n_eff < self.n_particles / 2:
          self.resample()

        # Estimate state
        most_likely_state, state_probs = self.estimate_state()

        # Store estimates
        self.state_estimates.append(most_likely_state)
        self.state_probabilities.append(state_probs)

        return most_likely_state, state_probs
    

def fit_hrl_pf(alpha, beta, actions, rewards, stimuli):
    """
    Fit a Hierarchical Reinforcement Learning Particle Filter (HRL-PF) to the given actions, rewards, and stimuli.

    Args:
        alpha (float): Learning rate in the HRL model.
        beta (float): Softmax temperature in the HRL model.
        actions (array-like): Array of actions taken.
        rewards (array-like): Array of rewards received.
        stimuli (array-like): Array of stimuli presented.

    Returns:
        estimated_states (list): List of estimated states from the particle filter.
        state_probabilities (list): List of probability distributions over states for each time step.
    """
    pf = HRLParticleFilter(alpha=alpha, beta=beta, n_particles=1000, n_states=3)
    # Run particle filter
    estimated_states = []
    state_probabilities = []
    for (action, reward, stim) in zip(actions, rewards, stimuli):
        est_state, state_probs = pf.filter(reward, action, stim)
        estimated_states.append(est_state)
        state_probabilities.append(state_probs)
    
    return estimated_states, state_probabilities
