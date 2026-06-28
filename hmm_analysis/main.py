import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pymc as pm

from data import prepare_data
from model import build_hierarchical_model, sample_model
from decoding import viterbi
from plotting import plot_forest, plot_trajectories, plot_trace, plot_state_characteristics
from threshold_model import analyze_exploration_thresholds

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"hmm_analysis/results/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Output Directory: {out_dir} ---")
    
    data_dict = prepare_data()
    
    print("\nBuilding Hierarchical Model...")
    hmm_model = build_hierarchical_model(data_dict)
    
    print("\nSampling...")
    trace = sample_model(hmm_model)
    
    print("\nSaving Trace Summary...")
    summary = pm.summary(trace)
    summary.to_csv(os.path.join(out_dir, "trace_summary.csv"))
    print(summary)
    
    print("\nGenerating Trace Plot...")
    plot_trace(trace, out_dir)
    
    print("\nGenerating Forest Plot...")
    plot_forest(trace, out_dir)
    
    print("\nDecoding Hidden States (Viterbi)...")
    map_params = {
        'p_init': trace.posterior['p_init'].mean(dim=['chain', 'draw']).values,
        'p_trans_pop': trace.posterior['p_trans_pop'].mean(dim=['chain', 'draw']).values,
        'p_trans_subj': trace.posterior['p_trans_subj'].mean(dim=['chain', 'draw']).values,
        'mu_jump_pop': np.array([
            trace.posterior['mu_jump_pop_0'].mean(dim=['chain', 'draw']).values,
            trace.posterior['mu_jump_pop_1'].mean(dim=['chain', 'draw']).values
        ]),
        'mu_rew_pop': np.array([
            trace.posterior['mu_rew_pop_0'].mean(dim=['chain', 'draw']).values,
            trace.posterior['mu_rew_pop_1'].mean(dim=['chain', 'draw']).values
        ]),
        'mu_jump_subj': trace.posterior['mu_jump_subj'].mean(dim=['chain', 'draw']).values,
        'mu_rew_subj': trace.posterior['mu_rew_subj'].mean(dim=['chain', 'draw']).values,
        'kappa_jump_subj': trace.posterior['kappa_jump_subj'].mean(dim=['chain', 'draw']).values,
        'kappa_rew_subj': trace.posterior['kappa_rew_subj'].mean(dim=['chain', 'draw']).values,
        'alpha_j_subj': trace.posterior['alpha_j_subj'].mean(dim=['chain', 'draw']).values,
        'beta_j_subj': trace.posterior['beta_j_subj'].mean(dim=['chain', 'draw']).values,
        'alpha_r_subj': trace.posterior['alpha_r_subj'].mean(dim=['chain', 'draw']).values,
        'beta_r_subj': trace.posterior['beta_r_subj'].mean(dim=['chain', 'draw']).values,
        'max_jump': data_dict['max_jump'],
        'max_rew': data_dict['max_rew']
    }
    
    decoded_rows = []
    unique_agents = sorted(list(set([s['agent'] for s in data_dict['use_seqs']])))
    agent_to_idx = {agent: i for i, agent in enumerate(unique_agents)}
    
    for seq in data_dict['use_seqs']:
        a_idx = agent_to_idx[seq['agent']]
        states = viterbi(
            seq['jumps'], seq['rewards'], 
            map_params['max_jump'], map_params['max_rew'], 
            map_params['p_init'], 
            map_params['p_trans_subj'][a_idx],
            map_params['alpha_j_subj'][a_idx], map_params['beta_j_subj'][a_idx],
            map_params['alpha_r_subj'][a_idx], map_params['beta_r_subj'][a_idx]
        )
        for i in range(len(states)):
            decoded_rows.append({
                'group': seq['group'],
                'agent': seq['agent'],
                'round': seq['round'],
                'trial': seq['trial'][i],
                'env': seq['env'],
                'jump': seq['jumps'][i],
                'reward': seq['rewards'][i],
                'coord': seq['coords'][i],
                'hidden_state': states[i]
            })
            
    viterbi_df = pd.DataFrame(decoded_rows)
    viterbi_df.to_csv(os.path.join(out_dir, "state_predictions_viterbi.csv"), index=False)
    
    print("\nGenerating State Characteristics and Empirical Transitions...")
    plot_state_characteristics(viterbi_df, map_params, out_dir)
    
    print("\nPerforming Threshold Analysis...")
    analyze_exploration_thresholds(viterbi_df, out_dir)
    
    print("\nGenerating Trajectories...")
    plot_trajectories(data_dict['use_seqs'], map_params, out_dir)
    
    print(f"\nPipeline Complete! All results saved to {out_dir}")

if __name__ == '__main__':
    main()
