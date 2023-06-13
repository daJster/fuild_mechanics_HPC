import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from milestone1 import create_density_grid, probability_density, velocity, direction, plot_density_grid, animate
import random

W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

rho = lambda density_grid : probability_density(density_grid)
mu = lambda density_grid : velocity(density_grid)

#---------------------------------------------------------------------------------------------
def equilibruim_distribution(rho_grid, velocity_grid) :
    # Calculate the dot product between direction and the velocity
    dot_product = np.einsum('ji, jkl -> ikl', direction, velocity_grid)
    # Calculate the magnitude squared of the velocity
    magnitude_squared = np.linalg.norm(velocity_grid, axis=0)**2
    # Apply the formula
    result = W[:, np.newaxis, np.newaxis] * rho_grid[np.newaxis, :, :] * (1 + 3 * dot_product + (9/2) * dot_product**2 - (3/2) * magnitude_squared[np.newaxis, :, :])
    return result

def collision_term(density_grid, relax) :
    return density_grid + relax*(equilibruim_distribution(rho(density_grid), mu(density_grid)) - density_grid) 
#---------------------------------------------------------------------------------------------
    
def create_collision_grid() : 
    density_grid_animate = create_density_grid(x_shape=300, y_shape=300, uniform=True, rand=False)
    offset = 140 
    # filling the center of the grid
    for i in range(50) :
        for j in range(50) :
            density_grid_animate[:, i + offset, j + offset] = [density_grid_animate[k, i + offset, j + offset] + .1 for k in range(9)]
    return density_grid_animate

if __name__ == "__main__" :
    
    density_grid = create_collision_grid()
    print(rho(density_grid).shape, direction.shape, W.shape)
    collision_function = lambda density_grid : collision_term(density_grid, .8)
    plot_density_grid(density_grid, file='density_grid_collision.png')
    animate(file='density_grid_collision.mp4', frames=200, collision=collision_function, create_grid=create_collision_grid)