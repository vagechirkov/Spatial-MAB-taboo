import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def choice_to_coord(choice, grid_size=11):
    """Convert integer choice (0-120) to (row, col) coordinate"""
    return np.array([choice // grid_size, choice % grid_size])

def calc_rg(coords, window=3):
    """Calculate rolling Radius of Gyration with a given window size."""
    N = len(coords)
    rg = np.zeros(N - 1)
    for i in range(1, N):
        start_idx = max(0, i + 1 - window)
        points = coords[start_idx:i+1]
        center = np.mean(points, axis=0)
        rg_val = np.sqrt(np.mean(np.sum((points - center)**2, axis=1)))
        rg[i-1] = max(rg_val, 1e-3)
    return rg

def extract_sequences(df):
    """
    Extract jump lengths and rewards for each trial.
    Returns:
        sequences: list of dicts with 'jumps', 'rewards', 'coords', 'group', 'agent', 'trial', 'round'
    """
    sequences = []
    grouped = df.groupby(['group', 'agent', 'round'])
    
    for name, group_df in grouped:
        group_df = group_df.sort_values('trial')
        choices = group_df['choice'].values
        rewards = group_df['reward'].values
        coords = np.array([choice_to_coord(c) for c in choices])
        
        if len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            distances = np.where(distances == 0, 1e-3, distances)
            
            rg = calc_rg(coords, window=3)
            
            sequences.append({
                'group': name[0],
                'agent': f"{name[0]}_{name[1]}",
                'original_agent_id': name[1],
                'round': name[2],
                'env': int(group_df['env'].iloc[0]),
                'jumps': distances,
                'radius_of_gyration': rg,
                'rewards': rewards[1:],
                'coords': coords,
                'trial': group_df['trial'].values[1:]
            })
            
    return sequences

def prepare_data():
    human_data_path = os.environ.get('HUMAN_DATA_PATH')
    if not human_data_path or not os.path.exists(human_data_path):
        raise FileNotFoundError(f"Could not find data file at HUMAN_DATA_PATH: {human_data_path}. Please set it in your .env file.")
        
    print(f"Loading data from {human_data_path}")
    df = pd.read_csv(human_data_path)
    sequences = extract_sequences(df)
    print(f"Extracted {len(sequences)} valid trial sequences.")
    
    unique_groups = []
    for seq in sequences:
        if seq['group'] not in unique_groups:
            unique_groups.append(seq['group'])
    target_groups = unique_groups[:] # Using 3 groups
    use_seqs = [seq for seq in sequences if seq['group'] in target_groups]
    print(f"Using {len(use_seqs)} sequences ({len(target_groups)} full groups) for the HMM estimation.")
    
    # Map agents to integer indices
    unique_agents = sorted(list(set([s['agent'] for s in use_seqs])))
    agent_to_idx = {agent: i for i, agent in enumerate(unique_agents)}
    n_agents = len(unique_agents)
    
    max_len = max(len(s['jumps']) for s in use_seqs)
    n_seqs = len(use_seqs)
    
    jumps_mat = np.zeros((max_len, n_seqs))
    rg_mat = np.zeros((max_len, n_seqs))
    mask_mat = np.zeros((max_len, n_seqs), dtype=bool)
    agent_idx = np.zeros(n_seqs, dtype=int)
    
    for i, seq in enumerate(use_seqs):
        L = len(seq['jumps'])
        jumps_mat[:L, i] = seq['jumps']
        rg_mat[:L, i] = seq['radius_of_gyration']
        mask_mat[:L, i] = True
        agent_idx[i] = agent_to_idx[seq['agent']]
        
    max_jump = np.max(jumps_mat) + 1e-3
    max_rg = np.max(rg_mat) + 1e-3
    
    jumps_mat_norm = np.clip(jumps_mat / max_jump, 1e-4, 1.0 - 1e-4)
    rg_mat_norm = np.clip(rg_mat / max_rg, 1e-4, 1.0 - 1e-4)
    
    return {
        'use_seqs': use_seqs,
        'jumps_mat_norm': jumps_mat_norm,
        'rg_mat_norm': rg_mat_norm,
        'mask_mat': mask_mat,
        'agent_idx': agent_idx,
        'n_agents': n_agents,
        'max_jump': max_jump,
        'max_rg': max_rg
    }
