import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pymc as pm

from data import prepare_data
from hmm_model import build_hierarchical_model, sample_model
from decoding import decode_states
from plotting import plot_forest, plot_trajectories, plot_trace, plot_state_characteristics
from transition_prediction_model import predict_transitions
from foraging_analysis import analyze_foraging
from transition_performance_correlation import analyze_transition_performance_correlation
from draw_model import generate_model_diagram

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"hmm_analysis/results/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Output Directory: {out_dir} ---")
    
    data_dict = prepare_data()
    
    print("\nBuilding Hierarchical Model...")
    hmm_model = build_hierarchical_model(data_dict)
    
    # Generate the clear visual diagram of the model
    generate_model_diagram(hmm_model, out_dir)
    
    print("\nSampling...")
    trace = sample_model(hmm_model)
    
    print("\nSaving Trace Summary...")
    summary = pm.stats.summary(trace)
    summary.to_csv(os.path.join(out_dir, "trace_summary.csv"))
    print(summary)
    
    print("\nGenerating Trace Plot...")
    plot_trace(trace, out_dir)
    
    print("\nGenerating Forest Plot...")
    plot_forest(trace, out_dir)
    
    print("\nDecoding Hidden States (Viterbi + Forward-Backward)...")
    map_params = {
        'p_init': np.array([0.5, 0.5]),
        'p_trans_pop': trace.posterior['p_trans_pop'].mean(dim=['chain', 'draw']).values,
        'p_trans_subj': trace.posterior['p_trans_subj'].mean(dim=['chain', 'draw']).values,
        'mu_jump_pop': np.array([
            trace.posterior['mu_jump_pop_0'].mean(dim=['chain', 'draw']).values,
            trace.posterior['mu_jump_pop_1'].mean(dim=['chain', 'draw']).values
        ]),
        # 'mu_rg_pop': np.array([
        #     trace.posterior['mu_rg_pop_0'].mean(dim=['chain', 'draw']).values,
        #     trace.posterior['mu_rg_pop_1'].mean(dim=['chain', 'draw']).values
        # ]),
        # 'mu_jump_subj': trace.posterior['mu_jump_subj'].mean(dim=['chain', 'draw']).values,
        # 'mu_rg_subj': trace.posterior['mu_rg_subj'].mean(dim=['chain', 'draw']).values,
        'kappa_jump_pop': trace.posterior['kappa_jump_pop'].mean(dim=['chain', 'draw']).values,
        # 'kappa_jump_subj': trace.posterior['kappa_jump_subj'].mean(dim=['chain', 'draw']).values,
        # 'kappa_rg_subj': trace.posterior['kappa_rg_subj'].mean(dim=['chain', 'draw']).values,
        'alpha_j_pop': trace.posterior['alpha_j_pop'].mean(dim=['chain', 'draw']).values,
        'beta_j_pop': trace.posterior['beta_j_pop'].mean(dim=['chain', 'draw']).values,
        # 'alpha_rg_subj': trace.posterior['alpha_rg_subj'].mean(dim=['chain', 'draw']).values,
        # 'beta_rg_subj': trace.posterior['beta_rg_subj'].mean(dim=['chain', 'draw']).values,
        'max_jump': data_dict['max_jump'],
        'max_rg': data_dict['max_rg']
    }
    
    decoded_rows = []
    unique_agents = sorted(list(set([s['agent'] for s in data_dict['use_seqs']])))
    agent_to_idx = {agent: i for i, agent in enumerate(unique_agents)}
    
    for seq in data_dict['use_seqs']:
        a_idx = agent_to_idx[seq['agent']]
        states, probs = decode_states(
            seq['jumps'], 
            map_params['max_jump'], 
            map_params['p_init'], 
            map_params['p_trans_subj'][a_idx],
            map_params['alpha_j_pop'], map_params['beta_j_pop']
        )
        
        # Identify Exploitation (jump exactly 0, which was replaced by 1e-3 in data prep)
        states = states.astype(float)
        states[np.array(seq['jumps']) <= 1.01e-3] = 2
        
        for i in range(len(states)):
            decoded_rows.append({
                'group': seq['group'],
                'agent': seq['agent'],
                'round': seq['round'],
                'trial': seq['trial'][i],
                'env': seq['env'],
                'jump': seq['jumps'][i],
                'radius_of_gyration': seq['radius_of_gyration'][i],
                'reward': seq['rewards'][i],
                'coord': seq['coords'][i],
                'hidden_state': states[i],
                'prob_state_0': probs[i, 0],
                'prob_state_1': probs[i, 1]
            })
            
    viterbi_df = pd.DataFrame(decoded_rows)
    viterbi_df_path = os.path.join(out_dir, "state_predictions_viterbi.csv")
    viterbi_df.to_csv(viterbi_df_path, index=False)
    
    print("\nGenerating State Characteristics and Empirical Transitions...")
    plot_state_characteristics(viterbi_df, map_params, out_dir)
    
    print("\nGenerating Trajectories...")
    plot_trajectories(data_dict['use_seqs'], map_params, out_dir)
    
    print("\nPerforming Transition Prediction Analysis (Bayesian Logistic Regression)...")
    predict_transitions(viterbi_df_path)
    
    print("\nPerforming Optimal Foraging Analysis (Beta Regression)...")
    analyze_foraging(viterbi_df_path)
    
    print("\nPerforming Transition Performance Correlation Analysis...")
    analyze_transition_performance_correlation(out_dir)
    
    print(f"\nPipeline Complete! All results saved to {out_dir}")

if __name__ == '__main__':
    main()
