import numpy as np
from scipy.stats import beta as beta_dist

def viterbi(jumps, rewards, max_jump, max_rew, p_init, p_trans, alpha_j, beta_j, alpha_r, beta_r):
    n_steps = len(jumps)
    n_states = 2
    
    j_norm = np.clip(jumps / max_jump, 1e-4, 1 - 1e-4)
    r_norm = np.clip(rewards / max_rew, 1e-4, 1 - 1e-4)
    
    log_p_trans = np.log(p_trans + 1e-12)
    log_p_init = np.log(p_init + 1e-12)
    
    log_emissions = np.zeros((n_steps, n_states))
    for k in range(n_states):
        log_j = beta_dist.logpdf(j_norm, a=alpha_j[k], b=beta_j[k])
        log_r = beta_dist.logpdf(r_norm, a=alpha_r[k], b=beta_r[k])
        log_emissions[:, k] = log_j + log_r
        
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
        
    return states
