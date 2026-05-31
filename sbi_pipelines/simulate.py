import os
import torch
import numpy as np
from joblib import Parallel, delayed
from sbi.utils import BoxUniform

from sbi_pipelines.model import SBIModel
from abm.rewards import make_parent_and_children_cholesky, build_corr_matrix_bare_bones

def simulate_one_param_set(params, n_rounds=7, grid_size=11, n_agents=4):
    """
    Simulates exactly n_rounds for a given parameter set.
    Returns:
      - choices: (n_rounds, 15, n_agents, 2)
      - rewards: (n_rounds, 15, n_agents)
      - landscapes: (n_rounds, n_agents, grid_size, grid_size)
      - summary_stats: (n_rounds, n_agents, n_stats) or averaged over rounds/agents
    """
    length_scale, beta, tau, alpha = params
    
    all_choices = []
    all_rewards = []
    all_landscapes = []
    all_summaries = []
    
    for r in range(n_rounds):
        corr_matrix = build_corr_matrix_bare_bones(n_agents + 1, 0.6)
        parent, child_maps = make_parent_and_children_cholesky(
            rng=np.random.default_rng(),
            grid_size=grid_size,
            n_children=n_agents,
            length_scale=2.0,
            corr_matrix=corr_matrix
        )
        child_maps = [c - 0.5 for c in child_maps]
        all_landscapes.append(child_maps)
        
        model = SBIModel(
            child_maps=child_maps,
            rng=np.random.default_rng(),
            n=n_agents,
            grid_size=grid_size,
            length_scale=length_scale,
            beta=beta,
            tau=tau,
            alpha=alpha,
            observation_noise=0.0001
        )
        
        for _ in range(15):
            model.step()
            
        # summary stats for this round (averaged across agents by the reporter, for each of 15 steps)
        # results has 15 rows (for 15 steps), and the reporter columns
        results = model.datacollector.get_model_vars_dataframe()
        summary_cols = [
            "avg_reward",
            "last_choice_distance_private",
            "last_choice_distance_social",
            "nearest_choice_distance_private",
            "avg_choice_distance_private",
            "nearest_choice_distance_social",
            "avg_choice_distance_social",
        ]
        summary_round = results[summary_cols].values # shape (15, 7)
        all_summaries.append(summary_round)
        
        # Extract choices and rewards for each agent
        # agent_reporters are not easily extracted per step if we just want the full history,
        # but the agent itself holds X_observations and y_observations
        r_choices = []
        r_rewards = []
        for a in model.agents_list:
            r_choices.append(a.X_observations)
            r_rewards.append(a.y_observations)
            
        all_choices.append(r_choices)
        all_rewards.append(r_rewards)
        
    # Shapes:
    # all_choices: (7, 4, 15, 2) -> transpose to (7, 15, 4, 2)
    choices_arr = np.array(all_choices).transpose(0, 2, 1, 3) 
    rewards_arr = np.array(all_rewards).transpose(0, 2, 1) # (7, 15, 4)
    landscapes_arr = np.array(all_landscapes) # (7, 4, 11, 11)
    summaries_arr = np.array(all_summaries).mean(axis=0) # Mean over 7 rounds
    
    return choices_arr, rewards_arr, landscapes_arr, summaries_arr


def parallel_simulate(theta, n_jobs=8):
    theta_np = theta.numpy()
    
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(simulate_one_param_set)(p) for p in theta_np
    )
    
    choices = np.stack([x[0] for x in out])
    rewards = np.stack([x[1] for x in out])
    landscapes = np.stack([x[2] for x in out])
    summaries = np.stack([x[3] for x in out])
    
    return choices, rewards, landscapes, summaries

if __name__ == "__main__":
    out_dir = "/groups/romanczuk/Chirkov_mab/sbi/gp_ucb_vs_06_2026/sim_10k_1k"
    os.makedirs(out_dir, exist_ok=True)
    
    # Priors: lambda, beta, tau, alpha
    lb = [0.1, 0.01, 0.005, 0.0]
    ub = [5.0, 2.0,  0.1,   1.0]
    prior = BoxUniform(low=torch.tensor(lb), high=torch.tensor(ub))
    
    print("Simulating training data (10000)...")
    theta_train = prior.sample((10_000,))
    c_train, r_train, l_train, s_train = parallel_simulate(theta_train, n_jobs=96)
    
    torch.save({
        'theta': theta_train,
        'choices': torch.tensor(c_train, dtype=torch.float32),
        'rewards': torch.tensor(r_train, dtype=torch.float32),
        'landscapes': torch.tensor(l_train, dtype=torch.float32),
        'summaries': torch.tensor(s_train, dtype=torch.float32)
    }, os.path.join(out_dir, "train_data.pt"))
    
    # print("Simulating test data (1000)...")
    # theta_test = prior.sample((1000,))
    # c_test, r_test, l_test, s_test = parallel_simulate(theta_test, n_jobs=96)
    
    # torch.save({
    #     'theta': theta_test,
    #     'choices': torch.tensor(c_test, dtype=torch.float32),
    #     'rewards': torch.tensor(r_test, dtype=torch.float32),
    #     'landscapes': torch.tensor(l_test, dtype=torch.float32),
    #     'summaries': torch.tensor(s_test, dtype=torch.float32)
    # }, os.path.join(out_dir, "test_data.pt"))
    
    print("Done!")
