import numpy as np

# Function to perform a Gaussian walk on a given grid 
def gaussian_walk(grid, steps, sigma=1.0, start='center', rng=np.random.default_rng()):    
    # Get the dimensions of the grid
    rows, cols = grid.shape
    
    if start == 'center':
        # Start at the center of the grid
        x, y = rows // 2, cols // 2
    else:
        # Start at a random position
        x, y = rng.integers(0, rows), rng.integers(0, cols)
    
    # Store the path taken
    path = np.array([[x, y]])

    for _ in range(steps):
        # Generate a random step from a Gaussian distribution
        dx = int(rng.normal(0, sigma))
        dy = int(rng.normal(0, sigma))
        
        # Update the position
        x = (x + dx) % rows  # Wrap around the grid
        y = (y + dy) % cols  # Wrap around the grid
        
        # Append the new position to the path
        path = np.vstack((path, [x, y]))
    
    return path, grid[path[:, 0], path[:, 1]]  # Return the path and the corresponding rewards from the grid