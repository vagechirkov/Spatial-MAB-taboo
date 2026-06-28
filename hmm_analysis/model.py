import pymc as pm
import pytensor.tensor as pt
import numpy as np
import os

def logit(p):
    return pt.log(p / (1 - p))


def hmm_logp_func(jumps_norm, rewards_norm, mask, p_init, p_trans_seqs, alpha_j_seqs, beta_j_seqs, alpha_r_seqs, beta_r_seqs):
    """Forward algorithm for log-likelihood in PyMC using padded sequences"""
    import pytensor
    
    y_j = pt.shape_padright(jumps_norm) # (max_len, n_seqs, 1)
    y_r = pt.shape_padright(rewards_norm)
    mask_tensor = pt.as_tensor_variable(mask)
    
    logp_j = pm.logp(pm.Beta.dist(alpha=alpha_j_seqs, beta=beta_j_seqs), y_j)
    logp_r = pm.logp(pm.Beta.dist(alpha=alpha_r_seqs, beta=beta_r_seqs), y_r)
    emission_logp = logp_j + logp_r # (max_len, n_seqs, n_states)
    
    def step(logp_emission_t, mask_t, prev_log_alpha, p_trans_seqs):
        p_trans_safe = pt.clip(p_trans_seqs, 1e-10, 1.0)
        log_p_trans = pt.log(p_trans_safe)
        term = pt.shape_padright(prev_log_alpha) + log_p_trans
        max_term = pt.max(term, axis=1, keepdims=True)
        log_sum = max_term[:, 0, :] + pt.log(pt.sum(pt.exp(term - max_term), axis=1))
        
        curr_log_alpha = log_sum + logp_emission_t
        mask_t_expanded = pt.shape_padright(mask_t)
        curr_log_alpha = pt.switch(mask_t_expanded, curr_log_alpha, prev_log_alpha)
        return curr_log_alpha
    
    first_log_alpha = pt.log(p_init) + emission_logp[0]
    
    log_alpha, _ = pytensor.scan(
        fn=step,
        sequences=[emission_logp[1:], mask_tensor[1:]],
        outputs_info=[first_log_alpha],
        non_sequences=[p_trans_seqs],
        strict=True
    )
    
    final_log_alpha = pt.concatenate([pt.shape_padleft(first_log_alpha), log_alpha], axis=0)
    
    # Calculate log likelihood for each sequence at the last valid time step
    lengths = pt.sum(mask_tensor, axis=0, dtype='int32')
    indices = pt.arange(lengths.shape[0])
    seq_logp = pt.logsumexp(final_log_alpha[lengths - 1, indices, :], axis=1)
    
    return seq_logp

def build_hierarchical_model(data_dict):
    jumps_mat_norm = data_dict['jumps_mat_norm']
    rewards_mat_norm = data_dict['rewards_mat_norm']
    mask_mat = data_dict['mask_mat']
    agent_idx = data_dict['agent_idx']
    n_agents = data_dict['n_agents']
    
    with pm.Model() as hmm_model:
        # Initial probabilities
        p_init = pm.Dirichlet('p_init', a=np.ones(2))
        
        # Population Transitions
        p_trans_pop = pm.Dirichlet('p_trans_pop', a=np.ones((2, 2)), shape=(2, 2))
        # Parameterize transitions on logit scale for Exploit->Explore and Explore->Exploit
        logit_p01_pop = pm.Deterministic('logit_p01_pop', logit(p_trans_pop[0, 1]))
        logit_p10_pop = pm.Deterministic('logit_p10_pop', logit(p_trans_pop[1, 0]))
        
        sigma_trans = pm.HalfNormal('sigma_trans', sigma=0.5, shape=2)
        offset_trans_01 = pm.Normal('offset_trans_01', mu=0, sigma=1, shape=n_agents)
        offset_trans_10 = pm.Normal('offset_trans_10', mu=0, sigma=1, shape=n_agents)
        
        p01_subj = pm.Deterministic('p01_subj', pm.math.invlogit(logit_p01_pop + offset_trans_01 * sigma_trans[0]))
        p10_subj = pm.Deterministic('p10_subj', pm.math.invlogit(logit_p10_pop + offset_trans_10 * sigma_trans[1]))
        
        p_trans_subj = pm.Deterministic('p_trans_subj', pt.stack([
            pt.stack([1 - p01_subj, p01_subj], axis=1),
            pt.stack([p10_subj, 1 - p10_subj], axis=1)
        ], axis=1)) # (n_agents, 2, 2)
        
        # Population Emissions
        mu_jump_pop_0 = pm.Uniform('mu_jump_pop_0', 0.0, 0.5)
        mu_jump_pop_1 = pm.Uniform('mu_jump_pop_1', mu_jump_pop_0, 1.0)
        
        mu_rew_pop_1 = pm.Uniform('mu_rew_pop_1', 0.0, 0.5)
        mu_rew_pop_0 = pm.Uniform('mu_rew_pop_0', mu_rew_pop_1, 1.0)
        
        sigma_jump = pm.HalfNormal('sigma_jump', sigma=0.5, shape=2)
        offset_jump_0 = pm.Normal('offset_jump_0', mu=0, sigma=1, shape=n_agents)
        offset_jump_1 = pm.Normal('offset_jump_1', mu=0, sigma=1, shape=n_agents)
        
        mu_jump_subj_0 = pm.math.invlogit(logit(mu_jump_pop_0) + offset_jump_0 * sigma_jump[0])
        mu_jump_subj_1 = pm.math.invlogit(logit(mu_jump_pop_1) + offset_jump_1 * sigma_jump[1])
        mu_jump_subj = pm.Deterministic('mu_jump_subj', pt.stack([mu_jump_subj_0, mu_jump_subj_1], axis=1)) # (n_agents, 2)
        
        sigma_rew = pm.HalfNormal('sigma_rew', sigma=0.5, shape=2)
        offset_rew_0 = pm.Normal('offset_rew_0', mu=0, sigma=1, shape=n_agents)
        offset_rew_1 = pm.Normal('offset_rew_1', mu=0, sigma=1, shape=n_agents)
        
        mu_rew_subj_0 = pm.math.invlogit(logit(mu_rew_pop_0) + offset_rew_0 * sigma_rew[0])
        mu_rew_subj_1 = pm.math.invlogit(logit(mu_rew_pop_1) + offset_rew_1 * sigma_rew[1])
        mu_rew_subj = pm.Deterministic('mu_rew_subj', pt.stack([mu_rew_subj_0, mu_rew_subj_1], axis=1))
        
        log_kappa_jump_pop = pm.Normal('log_kappa_jump_pop', mu=np.log(10), sigma=1, shape=2)
        sigma_kappa_jump = pm.HalfNormal('sigma_kappa_jump', sigma=0.5, shape=2)
        offset_kappa_jump_0 = pm.Normal('offset_kappa_jump_0', mu=0, sigma=1, shape=n_agents)
        offset_kappa_jump_1 = pm.Normal('offset_kappa_jump_1', mu=0, sigma=1, shape=n_agents)
        kappa_jump_subj_0 = pt.exp(log_kappa_jump_pop[0] + offset_kappa_jump_0 * sigma_kappa_jump[0])
        kappa_jump_subj_1 = pt.exp(log_kappa_jump_pop[1] + offset_kappa_jump_1 * sigma_kappa_jump[1])
        kappa_jump_subj = pm.Deterministic('kappa_jump_subj', pt.stack([kappa_jump_subj_0, kappa_jump_subj_1], axis=1))
        
        log_kappa_rew_pop = pm.Normal('log_kappa_rew_pop', mu=np.log(10), sigma=1, shape=2)
        sigma_kappa_rew = pm.HalfNormal('sigma_kappa_rew', sigma=0.5, shape=2)
        offset_kappa_rew_0 = pm.Normal('offset_kappa_rew_0', mu=0, sigma=1, shape=n_agents)
        offset_kappa_rew_1 = pm.Normal('offset_kappa_rew_1', mu=0, sigma=1, shape=n_agents)
        kappa_rew_subj_0 = pt.exp(log_kappa_rew_pop[0] + offset_kappa_rew_0 * sigma_kappa_rew[0])
        kappa_rew_subj_1 = pt.exp(log_kappa_rew_pop[1] + offset_kappa_rew_1 * sigma_kappa_rew[1])
        kappa_rew_subj = pm.Deterministic('kappa_rew_subj', pt.stack([kappa_rew_subj_0, kappa_rew_subj_1], axis=1))
        
        alpha_j_subj = pm.Deterministic('alpha_j_subj', mu_jump_subj * kappa_jump_subj)
        beta_j_subj = pm.Deterministic('beta_j_subj', (1.0 - mu_jump_subj) * kappa_jump_subj)
        
        alpha_r_subj = pm.Deterministic('alpha_r_subj', mu_rew_subj * kappa_rew_subj)
        beta_r_subj = pm.Deterministic('beta_r_subj', (1.0 - mu_rew_subj) * kappa_rew_subj)
        
        # Map parameters to sequences
        a_idx = pt.as_tensor_variable(agent_idx)
        p_trans_seqs = p_trans_subj[a_idx]
        alpha_j_seqs = alpha_j_subj[a_idx]
        beta_j_seqs = beta_j_subj[a_idx]
        alpha_r_seqs = alpha_r_subj[a_idx]
        beta_r_seqs = beta_r_subj[a_idx]
        
        logp_seqs = hmm_logp_func(
            jumps_mat_norm, rewards_mat_norm, mask_mat, 
            p_init, p_trans_seqs, alpha_j_seqs, beta_j_seqs, alpha_r_seqs, beta_r_seqs
        )
            
        total_logp = pt.sum(logp_seqs)
        pm.Potential('total_hmm_logp', total_logp)
        pm.Deterministic('seq_logp', logp_seqs)
        
    return hmm_model

def sample_model(hmm_model):
    import multiprocessing
    with hmm_model:
        total_cores = int(multiprocessing.cpu_count())
        use_cores = max(1, total_cores - 5)
        
        print(f"Using {use_cores} cores and {use_cores} chains out of {total_cores} available.")
        
        trace = pm.sample(draws=3000, tune=1000, chains=use_cores, cores=use_cores, 
                          return_inferencedata=True, progressbar=True,
                          init='jitter+adapt_diag', target_accept=0.98)
    return trace
