import numpy as np
import matplotlib.pyplot as plt

def plot_reward_grid(grid, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.axis('off')
    plt.show()