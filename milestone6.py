import numpy as np
import matplotlib.pyplot as plt
from milestone2 import collision_term, W, mu
from milestone1 import direction, create_density_grid, result_repo, streaming2D, animate
from milestone4 import top_channels, bottom_channels, right_channels, left_channels, create_couette_grid_fixed


def create_sliding_lid_grid() :
    probability_density_grid = create_density_grid(y_shape=302, x_shape=302, rand=False, uniform=True)
    return probability_density_grid

def create_sliding_lid_boundaries(west=True, east=True, north=True, south=True) :
    
    def create_sliding_lid_boundaries_inner(probability_density_grid) :
        wall_speed = [1.6, 0]
        # Top and bottom boundaries
        if not west and not east and not north and not south :
            return probability_density_grid
        
        for x in range(probability_density_grid.shape[2]) :
            cell_top = np.copy(probability_density_grid[:, -1, x])
            cell_bottom = np.copy(probability_density_grid[:, 0, x])
            proba_cell_top = cell_top.sum()
            
            # new moving densities
            new_cell_top_8 = cell_top[6] - 2*W[6]*proba_cell_top*np.dot(direction[:, 6], wall_speed)/np.dot(direction[:, 6], direction[:, 6])
            new_cell_top_4 = cell_top[2] - 2*W[2]*proba_cell_top*np.dot(direction[:, 2], wall_speed)/np.dot(direction[:, 2], direction[:, 2])
            new_cell_top_7 = cell_top[5] - 2*W[5]*proba_cell_top*np.dot(direction[:, 5], wall_speed)/np.dot(direction[:, 5], direction[:, 5])
           
            if north : 
                # top boundary
                probability_density_grid[bottom_channels, -1, x] = [new_cell_top_7, new_cell_top_4, new_cell_top_8]
            if south :
                # bottom boundary
                probability_density_grid[top_channels, 0, x] = cell_bottom[bottom_channels]
                
            # now change axis
            y = x
            cell_right = np.copy(probability_density_grid[:, y, -1])
            cell_left = np.copy(probability_density_grid[:, y, 0])
            
            if east :
                # right boundary
                probability_density_grid[left_channels, y, -1] = cell_right[right_channels]
            if west :
                # left boundary
                probability_density_grid[right_channels, y, 0] = cell_left[left_channels]
            
        return probability_density_grid
    
    return create_sliding_lid_boundaries_inner

def velocity_streamplot():
    framestop = 200000
    omega = 1.7
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    density_grid_plot = create_sliding_lid_grid()
    x, y = np.arange(0, density_grid_plot.shape[2]), np.arange(0, density_grid_plot.shape[1])
    x, y = np.meshgrid(x, y)
    
    plot_arr_norm = np.zeros(300)
    
    plt.figure()
    for i in range(framestop) :
        print(i, '/', framestop, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, boundary=create_sliding_lid_boundaries(), test=True)
        plot_arr_norm = mu(density_grid_plot)
        u, v = plot_arr_norm[0], plot_arr_norm[1]
    
    plt.streamplot(x, y, u, v, density=1)

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.savefig(result_repo+'velocity_streamplot_sliding_lid.png')
    plt.close()

if __name__ == "__main__" :
    collision_function = lambda density_grid : collision_term(density_grid, .53)
    
    animate(file='sliding_lid.mp4', frames=2000, velocity_active='norm', cmap='viridis', collision=collision_function, create_grid=create_sliding_lid_grid, boundary=create_sliding_lid_boundaries)
    animate(file='sliding_lid_test_boundaries.mp4', frames=1000, collision=collision_function, create_grid=create_couette_grid_fixed, boundary=create_sliding_lid_boundaries)
    
    velocity_streamplot()