import numpy as np
import matplotlib.pyplot as plt
from milestone2 import collision_term
from milestone1 import direction, create_density_grid, animate, plot_density_grid
import random

top_channels = [5, 2, 6]
bottom_channels = [7, 4, 8]
    
def create_couette_grid_fixed() :
    density_grid_animate = create_density_grid(y_shape=102, x_shape=300, uniform=True, rand=False)
    offset_x = 60
    offset_y = 60
    # filling the center of the grid
    for i in range(20) :
        for j in range(100) :
            density_grid_animate[:, i + offset_x, j + offset_y] = [density_grid_animate[k, i + offset_x, j + offset_y] + .2 for k in range(9)]

    return density_grid_animate


def create_couette_grid(x_shape=300, y_shape=300, epsilon=.01, rho0=1) :
    density_grid_animate = create_density_grid(y_shape=102, x_shape=300, uniform=True, rand=False)
    offset_x = 60
    offset_y = 0
    # for i in range(20) :
    #     for j in range(300) :
    #         density_grid_animate[:, i + offset_x, j + offset_y] = [density_grid_animate[k, i + offset_x, j + offset_y] + .2 for k in range(9)]

    return density_grid_animate



def set_couette_boundary_fixed(probability_density_grid) :
    for y in range(probability_density_grid.shape[2]) :
        cell_top = np.copy(probability_density_grid[:, 0, y])
        cell_bottom = np.copy(probability_density_grid[:, -1, y])
        # top boundary
        probability_density_grid[bottom_channels, 0, y] = cell_top[top_channels]
        # bottom boundary
        probability_density_grid[top_channels, -1, y] = cell_bottom[bottom_channels]

    return probability_density_grid

def set_couette_boundary(probability_density_grid) :
    Un = 3
    for y in range(probability_density_grid.shape[2]) :
        cell_top = np.copy(probability_density_grid[:, 0, y])
        cell_bottom = np.copy(probability_density_grid[:, -1, y])
        Rhon = cell_top[0] + cell_top[1] + cell_top[3] + 2*(cell_top[2] + cell_top[6] + cell_top[5])
        # top boundary
        probability_density_grid[bottom_channels, 0, y] = [cell_top[5] + (cell_top[1] - cell_top[3])/2 - Rhon*Un/2, \
            cell_top[2], \
            cell_top[6] - (cell_top[1] - cell_top[3])/2 + Rhon*Un/2]
        # bottom boundary
        probability_density_grid[top_channels, -1, y] = cell_bottom[bottom_channels]

    return probability_density_grid


if __name__ == "__main__" :
    #plot_density_grid(create_couette_grid_fixed(), file="couette_grid_fixed.png")
    collision_function = lambda density_grid : collision_term(density_grid, .7)
    #animate(file="couette_boundary_fixed.mp4", collision=collision_function, create_grid=create_couette_grid_fixed, frames=200, interval=100, boundary=set_couette_boundary_fixed)
    animate(file="couette_boundary.mp4", collision=collision_function, create_grid=create_couette_grid, frames=200, interval=100, boundary=set_couette_boundary)