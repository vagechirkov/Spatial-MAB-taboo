import numpy as np
from scipy.stats import beta as beta_dist

def forward_backward(log_emissions, log_p_init, log_p_trans):
    """
    Computes the smoothed state probabilities (gamma) using the standard Forward-Backward algorithm 
    in log-space for numerical stability.
    
    Reference for mathematical formulation:
    - Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. 
      Proceedings of the IEEE, 77(2), 257-286.
      
    Reference for implementation pattern in Python:
    - Closely mirrors the structure of `hmmlearn`'s internal passes: 
      https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/_hmmc.pyx
      Specifically the `_log_forward` and `_log_backward` functions.
    """
    n_steps, n_states = log_emissions.shape
    
    # Forward pass
    alpha = np.zeros((n_steps, n_states))
    alpha[0, :] = log_p_init + log_emissions[0, :]
    for t in range(1, n_steps):
        for j in range(n_states):
            # log sum exp
            max_val = np.max(alpha[t-1, :] + log_p_trans[:, j])
            alpha[t, j] = log_emissions[t, j] + max_val + np.log(np.sum(np.exp(alpha[t-1, :] + log_p_trans[:, j] - max_val)))

    # Backward pass
    beta = np.zeros((n_steps, n_states))
    # beta[-1, :] = 0 (log 1)
    for t in range(n_steps - 2, -1, -1):
        for i in range(n_states):
            max_val = np.max(log_p_trans[i, :] + log_emissions[t+1, :] + beta[t+1, :])
            beta[t, i] = max_val + np.log(np.sum(np.exp(log_p_trans[i, :] + log_emissions[t+1, :] + beta[t+1, :] - max_val)))

    # Compute smoothed probabilities
    gamma = alpha + beta
    # Normalize per time step
    for t in range(n_steps):
        max_val = np.max(gamma[t, :])
        gamma[t, :] = gamma[t, :] - (max_val + np.log(np.sum(np.exp(gamma[t, :] - max_val))))
        
    return np.exp(gamma)

def decode_states(jumps, max_jump, p_init, p_trans, alpha_j, beta_j):
    """
    Decodes the Hidden Markov Model states using two methods:
    1. Viterbi algorithm (finds the single most likely sequence of states)
    2. Forward-Backward algorithm (finds the marginal probability of being in each state at each time step)
    
    Reference for Viterbi decoding implementation:
    - Similar to `librosa.sequence.viterbi`: 
      https://librosa.org/doc/main/generated/librosa.sequence.viterbi.html
    - Standard dynamic programming approach computing the path with maximum log-likelihood.
    """
    n_steps = len(jumps)
    n_states = 2
    
    log_p_trans = np.log(p_trans + 1e-12)
    log_p_init = np.log(p_init + 1e-12)
    
    log_emissions = np.zeros((n_steps, n_states))
    
    jumps_norm = np.clip(jumps / max_jump, 1e-5, 1 - 1e-5)
    for i in range(n_states):
        log_emissions[:, i] += beta_dist.logpdf(jumps_norm, alpha_j[i], beta_j[i])
        
    # rgs_norm = np.clip(rgs / max_rg, 1e-5, 1 - 1e-5)
    # for i in range(n_states):
    #     log_emissions[:, i] += beta_dist.logpdf(rgs_norm, alpha_rg[i], beta_rg[i])
        
    # Viterbi
    T1 = np.zeros((n_states, n_steps))
    T2 = np.zeros((n_states, n_steps), dtype=int)
    
    T1[:, 0] = log_p_init + log_emissions[0, :]
    
    for i in range(1, n_steps):
        for j in range(n_states):
            prob = T1[:, i-1] + log_p_trans[:, j] + log_emissions[i, j]
            T1[j, i] = np.max(prob)
            T2[j, i] = np.argmax(prob)
            
    states = np.zeros(n_steps, dtype=int)
    states[-1] = np.argmax(T1[:, -1])
    
    for i in range(n_steps - 2, -1, -1):
        states[i] = T2[states[i+1], i+1]
        
    # Forward-Backward
    probs = forward_backward(log_emissions, log_p_init, log_p_trans)
        
    return states, probs
