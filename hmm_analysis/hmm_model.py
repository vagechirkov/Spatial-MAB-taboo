import pymc as pm
import pytensor.tensor as pt
import numpy as np
import os

def logit(p):
    return pt.log(p / (1 - p))


def hmm_logp_func(jumps_norm, mask, p_init, p_trans_seqs, alpha_j_seqs, beta_j_seqs):
    """
    Forward algorithm for log-likelihood in PyMC using padded sequences.
    
    Reference for implementation in PyMC/PyTensor:
    - This implements the standard forward algorithm in a tensor-based framework,
      similar to the approach in standard PyMC HMM tutorials:
      https://www.pymc.io/projects/examples/en/latest/time_series/hmm_intro.html
    - The use of `pytensor.scan` and `logsumexp` ensures numerical stability in log-space.
    """
    import pytensor
    
    y_j = pt.shape_padright(jumps_norm) # (max_len, n_seqs, 1)
    mask_tensor = pt.as_tensor_variable(mask)
    
    logp_j = pm.logp(pm.Beta.dist(alpha=alpha_j_seqs, beta=beta_j_seqs), y_j)
    # logp_rg = pm.logp(pm.Beta.dist(alpha=alpha_rg_seqs, beta=beta_rg_seqs), y_rg)
    emission_logp = logp_j # + logp_rg # (max_len, n_seqs, n_states)
    
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
    rg_mat_norm = data_dict['rg_mat_norm']
    mask_mat = data_dict['mask_mat']
    agent_idx = data_dict['agent_idx']
    n_agents = data_dict['n_agents']
    n_seqs = len(agent_idx)
    
    with pm.Model() as hmm_model:
        # Initial probabilities (fixed to remove unidentifiable parameters)
        p_init = np.array([0.5, 0.5])
        
        # Population Transitions (Biased towards persistence)
        p_trans_pop = pm.Dirichlet('p_trans_pop', a=np.array([[10.0, 2.0], [2.0, 10.0]]), shape=(2, 2))
        # Parameterize transitions on logit scale for Exploit->Explore and Explore->Exploit
        logit_p01_pop = pm.Deterministic('logit_p01_pop', logit(p_trans_pop[0, 1]))
        logit_p10_pop = pm.Deterministic('logit_p10_pop', logit(p_trans_pop[1, 0]))
        
        # Hierarchical p_trans
        sigma_trans = pm.HalfNormal('sigma_trans', sigma=0.5)
        offset_trans = pm.Normal('offset_trans', mu=0, sigma=1, shape=(n_agents, 2))
        
        logit_p_trans_subj_01 = logit_p01_pop + offset_trans[:, 0] * sigma_trans
        logit_p_trans_subj_10 = logit_p10_pop + offset_trans[:, 1] * sigma_trans
        
        p_trans_subj_01 = pm.math.invlogit(logit_p_trans_subj_01)
        p_trans_subj_10 = pm.math.invlogit(logit_p_trans_subj_10)
        
        p_trans_subj = pm.Deterministic('p_trans_subj', pt.stack([
            pt.stack([1 - p_trans_subj_01, p_trans_subj_01], axis=1),
            pt.stack([p_trans_subj_10, 1 - p_trans_subj_10], axis=1)
        ], axis=1)) # (n_agents, 2, 2)
        
        # Strict priors for better separation
        mu_jump_pop_0 = pm.Uniform('mu_jump_pop_0', 0.0, 0.35)
        mu_jump_pop_1 = pm.Uniform('mu_jump_pop_1', 0.40, 1.0)
        
        # Radius of gyration (Disabled for now)
        # mu_rg_pop_0 = pm.Uniform('mu_rg_pop_0', 0.0, 0.5)
        # mu_rg_pop_1 = pm.Uniform('mu_rg_pop_1', mu_rg_pop_0, 1.0)
        
        # Hierarchical jump means disabled to enforce universal state definitions
        # sigma_jump = pm.HalfNormal('sigma_jump', sigma=0.5, shape=2)
        # offset_jump_0 = pm.Normal('offset_jump_0', mu=0, sigma=1, shape=n_agents)
        # offset_jump_1 = pm.Normal('offset_jump_1', mu=0, sigma=1, shape=n_agents)
        # mu_jump_subj_0 = pm.math.invlogit(logit(mu_jump_pop_0) + offset_jump_0 * sigma_jump[0])
        # mu_jump_subj_1 = pm.math.invlogit(logit(mu_jump_pop_1) + offset_jump_1 * sigma_jump[1])
        # mu_jump_subj = pm.Deterministic('mu_jump_subj', pt.stack([mu_jump_subj_0, mu_jump_subj_1], axis=1)) # (n_agents, 2)
        
        mu_jump_pop = pt.stack([mu_jump_pop_0, mu_jump_pop_1])
        
        # sigma_rg = pm.HalfNormal('sigma_rg', sigma=0.5, shape=2)
        # offset_rg_0 = pm.Normal('offset_rg_0', mu=0, sigma=1, shape=n_agents)
        # offset_rg_1 = pm.Normal('offset_rg_1', mu=0, sigma=1, shape=n_agents)
        
        # mu_rg_subj_0 = pm.math.invlogit(logit(mu_rg_pop_0) + offset_rg_0 * sigma_rg[0])
        # mu_rg_subj_1 = pm.math.invlogit(logit(mu_rg_pop_1) + offset_rg_1 * sigma_rg[1])
        # mu_rg_subj = pm.Deterministic('mu_rg_subj', pt.stack([mu_rg_subj_0, mu_rg_subj_1], axis=1))
        
        # Global kappa for jumps (shared across states to prevent divergence/multimodality)
        log_kappa_jump_pop = pm.Normal('log_kappa_jump_pop', mu=np.log(10), sigma=1)
        kappa_jump_pop = pm.Deterministic('kappa_jump_pop', pt.exp(log_kappa_jump_pop))
        
        # Hierarchical kappa commented out for better convergence
        # sigma_kappa_jump = pm.HalfNormal('sigma_kappa_jump', sigma=0.5, shape=2)
        # offset_kappa_jump_0 = pm.Normal('offset_kappa_jump_0', mu=0, sigma=1, shape=n_agents)
        # offset_kappa_jump_1 = pm.Normal('offset_kappa_jump_1', mu=0, sigma=1, shape=n_agents)
        # kappa_jump_subj_0 = pt.exp(log_kappa_jump_pop[0] + offset_kappa_jump_0 * sigma_kappa_jump[0])
        # kappa_jump_subj_1 = pt.exp(log_kappa_jump_pop[1] + offset_kappa_jump_1 * sigma_kappa_jump[1])
        # kappa_jump_subj = pm.Deterministic('kappa_jump_subj', pt.stack([kappa_jump_subj_0, kappa_jump_subj_1], axis=1))
        
        # log_kappa_rg_pop = pm.Normal('log_kappa_rg_pop', mu=np.log(10), sigma=1, shape=2)
        # sigma_kappa_rg = pm.HalfNormal('sigma_kappa_rg', sigma=0.5, shape=2)
        # offset_kappa_rg_0 = pm.Normal('offset_kappa_rg_0', mu=0, sigma=1, shape=n_agents)
        # offset_kappa_rg_1 = pm.Normal('offset_kappa_rg_1', mu=0, sigma=1, shape=n_agents)
        # kappa_rg_subj_0 = pt.exp(log_kappa_rg_pop[0] + offset_kappa_rg_0 * sigma_kappa_rg[0])
        # kappa_rg_subj_1 = pt.exp(log_kappa_rg_pop[1] + offset_kappa_rg_1 * sigma_kappa_rg[1])
        # kappa_rg_subj = pm.Deterministic('kappa_rg_subj', pt.stack([kappa_rg_subj_0, kappa_rg_subj_1], axis=1))
        
        alpha_j_pop = pm.Deterministic('alpha_j_pop', mu_jump_pop * kappa_jump_pop)
        beta_j_pop = pm.Deterministic('beta_j_pop', (1.0 - mu_jump_pop) * kappa_jump_pop)
        
        # alpha_rg_subj = pm.Deterministic('alpha_rg_subj', mu_rg_subj * kappa_rg_subj)
        # beta_rg_subj = pm.Deterministic('beta_rg_subj', (1.0 - mu_rg_subj) * kappa_rg_subj)
        
        # Map parameters to sequences
        a_idx = pt.as_tensor_variable(agent_idx)
        # Use hierarchical transition matrix
        p_trans_seqs = p_trans_subj[a_idx]
        
        # Emissions are universal/global
        alpha_j_seqs = pt.repeat(pt.shape_padleft(alpha_j_pop), n_seqs, axis=0) # (n_seqs, 2)
        beta_j_seqs = pt.repeat(pt.shape_padleft(beta_j_pop), n_seqs, axis=0)
        # alpha_rg_seqs = alpha_rg_subj[a_idx]
        # beta_rg_seqs = beta_rg_subj[a_idx]
        
        logp_seqs = hmm_logp_func(
            jumps_mat_norm, mask_mat, 
            p_init, p_trans_seqs, alpha_j_seqs, beta_j_seqs
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
        
        trace = pm.sample(draws=2000, tune=1000, chains=use_cores, cores=use_cores, 
                          return_inferencedata=True, progressbar=True,
                          # nuts_sampler="numpyro",
                          # init='jitter+adapt_diag',
                          target_accept=0.95)
    return trace
