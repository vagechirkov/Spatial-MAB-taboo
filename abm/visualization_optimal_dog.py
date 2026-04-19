import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_foraging_advantage_heatmap(t_gp, t_dog, max_T=150, max_R=3.0):
    T_vals = np.linspace(0, max_T, 300)
    R_vals = np.linspace(0, max_R, 300)
    T, R = np.meshgrid(T_vals, R_vals)
    
    V_gp = np.maximum(0, T - t_gp)
    V_dog = R * np.maximum(0, T - t_dog)
    
    Advantage = V_dog - V_gp
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    vmin, vmax = Advantage.min(), Advantage.max()
    if vmin >= 0:
        norm = plt.Normalize(vmin=0, vmax=vmax)
    elif vmax <= 0:
        norm = plt.Normalize(vmin=vmin, vmax=0)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    c = ax.pcolormesh(T, R, Advantage, cmap='RdBu_r', norm=norm, shading='auto')
    cb = fig.colorbar(c, ax=ax)
    cb.set_label('Advantage: E[V_DoG] - E[V_GP]', fontsize=12, fontweight='bold')
    
    if vmin < 0 < vmax:
        contour = ax.contour(T, R, Advantage, levels=[0], colors='black', linewidths=2.5, linestyles='solid')
        ax.clabel(contour, inline=True, fmt='Break-even (Advantage=0)', fontsize=10)
    
    ax.set_xlabel('Time Budget (T)', fontsize=12, fontweight='bold')
    ax.set_ylabel('DoG Reward Multiplier (R)', fontsize=12, fontweight='bold')
    ax.set_title(f'Exploration Advantage Heatmap\n(t_gp={t_gp}, t_dog={t_dog})', fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    t_gp = 5
    t_dog = 35
    fig, ax = plot_foraging_advantage_heatmap(t_gp=t_gp, t_dog=t_dog, max_T=150, max_R=3.0)

    output_path = "foraging_advantage_heatmap.png"
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")
    plt.close()
