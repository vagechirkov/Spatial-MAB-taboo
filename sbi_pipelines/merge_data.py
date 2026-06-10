import os
import torch

def main():
    f1 = '/groups/romanczuk/Chirkov_mab/sbi/gp_ucb_vs_06_2026/sim_45k_1k/train_data.pt'
    f2 = '/groups/romanczuk/Chirkov_mab/sbi/gp_ucb_vs_06_2026/sim_50k_1k/train_data.pt'
    out_dir = '/groups/romanczuk/Chirkov_mab/sbi/gp_ucb_vs_06_2026/sim_95k_1k'
    out_f = os.path.join(out_dir, 'train_data.pt')

    print(f"Loading {f1}...")
    d1 = torch.load(f1)
    
    print(f"Loading {f2}...")
    d2 = torch.load(f2)

    merged = {}
    for k in d1.keys():
        print(f"Merging {k}...")
        merged[k] = torch.cat([d1[k], d2[k]], dim=0)

    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving to {out_f}...")
    torch.save(merged, out_f)
    print("Done!")

if __name__ == '__main__':
    main()
