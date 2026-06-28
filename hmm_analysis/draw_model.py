import os
import pymc as pm
import graphviz

def generate_model_diagram(hmm_model, out_dir):
    print("Generating clean HMM model diagram...")
    try:
        # Generate base graphviz source
        g = pm.model_to_graphviz(hmm_model)
        dot_source = g.source
        
        # We manipulate the dot source directly to inject custom colors and readable labels 
        # without breaking the underlying node IDs and edges.
        replacements = {
            # Initial Probs
            'label="p_init': 'style="filled" fillcolor="#f0f0f0" label="Initial State Probs',
            
            # Transitions (Highlighted in Orange/Red)
            'label="p_trans_pop': 'style="filled" fillcolor="#f9d0c4" label="Population\\nTransition Matrix',
            'label="sigma_trans': 'label="Transition Variance',
            'label="offset_trans': 'label="Subject Trans. Offsets',
            'label="p_trans_subj': 'style="filled" fillcolor="#ffb347" label="Subject Transition Matrix\\n(Latent State Generator)',
            'label="logit_p01_pop': 'label="Logit(P_01)',
            'label="logit_p10_pop': 'label="Logit(P_10)',
            
            # Emissions (Highlighted in Blue)
            'label="mu_jump_pop_0': 'label="Local Exploration\\nJump Mean',
            'label="mu_jump_pop_1': 'label="Global Exploration\\nJump Mean',
            'label="kappa_jump_pop': 'label="Jump Concentration\\n(Kappa)',
            'label="log_kappa_jump_pop': 'label="Log(Kappa)',
            
            'label="alpha_j_pop': 'style="filled" fillcolor="#add8e6" label="Jump Emission Alpha\\n(Beta Dist)',
            'label="beta_j_pop': 'style="filled" fillcolor="#add8e6" label="Jump Emission Beta\\n(Beta Dist)',
            
            # Observed Log-Likelihood (Highlighted in Green)
            'label="seq_logp': 'style="filled,rounded" fillcolor="#d3f8d3" label="Observed Sequences\\n(HMM Forward Algorithm Logp)'
        }
        
        for old, new in replacements.items():
            dot_source = dot_source.replace(old, new)
            
        # Add graph-level attributes for cleaner look
        dot_source = dot_source.replace('digraph {', 'digraph { rankdir=TB; splines=ortho; nodesep=0.5; ranksep=0.7;')
            
        custom_g = graphviz.Source(dot_source)
        
        # Save as high-res PNG and PDF
        out_path = os.path.join(out_dir, "hmm_model_diagram")
        custom_g.render(out_path, format='png', cleanup=True)
        custom_g.render(out_path, format='pdf', cleanup=True)
        
        print(f"Model diagrams saved to {out_dir}/hmm_model_diagram.[png/pdf]")
    except Exception as e:
        print(f"Could not generate model diagram. Do you have 'graphviz' installed? Error: {e}")

if __name__ == "__main__":
    import os
    from datetime import datetime
    
    # Allows running as a standalone script to quickly regenerate the diagram
    # without having to run the massive MCMC sampling.
    try:
        from data import prepare_data
        from hmm_model import build_hierarchical_model
        
        print("Running standalone diagram generation...")
        
        # Create output directory
        out_dir = "hmm_analysis/results/standalone_diagram"
        os.makedirs(out_dir, exist_ok=True)
        
        # Quick data prep and model build
        data_dict = prepare_data()
        hmm_model = build_hierarchical_model(data_dict)
        
        # Generate diagram
        generate_model_diagram(hmm_model, out_dir)
        print(f"Standalone diagram successfully created in {out_dir}!")
        
    except ImportError as e:
        print(f"Error: Make sure you run this script from the root project directory: 'python hmm_analysis/draw_model.py'. Error: {e}")
