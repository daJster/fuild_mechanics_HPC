import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema
from milestone2 import collision_term, equilibruim_distribution, rho, mu
from milestone1 import streaming2D, result_repo, direction, plot_density_grid
import random



#---------------------------------------------------------------------------------------------
def create_sinus_density(x_shape=300, y_shape=300, epsilon=.01, rho0=1) :
    velocity_grid = np.zeros((2, y_shape, x_shape))
    Lx = x_shape  # Length of the x-axis
    x = np.arange(Lx)
    
    density_variation = rho0 + epsilon * np.sin(2   * np.pi * x / Lx) # array 300
    reshaped_array = density_variation.reshape(300, 1)
    rho_grid = np.tile(reshaped_array, (1, 300))
    return rho_grid, velocity_grid
            
def create_sinus_velocity(x_shape=300, y_shape=300, epsilon=.1) :
    rho_grid = np.full((y_shape, x_shape), 1)
    Ly = y_shape  # Length of the y-axis
    y = np.arange(y_shape)
    velocity_variation = epsilon * np.sin(2 * np.pi * y / Ly)
    velocity_grid = np.expand_dims(velocity_variation, axis=(0, 2))
    velocity_grid = np.repeat(velocity_grid, repeats=300, axis=2)
    velocity_grid = np.repeat(velocity_grid, repeats=2, axis=0)
    return rho_grid, velocity_grid
#---------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------
def animate(file=None, frames=200, interval=100, collision=None, cmap="Blues", create_grid=None, boundary=False) :
    rho_grid_animate, velocity_grid_animate = create_grid()
    density_grid_animate = equilibruim_distribution(rho_grid_animate, velocity_grid_animate)
    print(rho_grid_animate.shape, velocity_grid_animate.shape, density_grid_animate.shape)

    total_density = density_grid_animate.sum(axis=0)
    fig = plt.figure()
    im = plt.imshow(total_density, animated=True, cmap=cmap)
    i = 0 
    def updatefig(frame) :
        nonlocal density_grid_animate, i
        i += 1
        density_grid_animate = streaming2D(density_grid_animate, direction, collision=collision, test=True, boundary=boundary)
        frame = density_grid_animate.sum(axis=0)
        im.set_array(frame)
        print('frame :', i, "/", frames, end='\r')
        return im,

    ani = FuncAnimation(fig, updatefig, blit=True, frames=frames, interval=interval)
    cbar = fig.colorbar(im)
    plt.gca().invert_yaxis()
    ani.save(result_repo+file, writer='ffmpeg')

def plot_rho(rho, file=None) :
    plt.figure()
    plt.imshow(rho, cmap='Blues')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(result_repo+file)
    plt.close()
#---------------------------------------------------------------------------------------------

def plot_density_on_axis() :
    omega = .7
    tmax = 10000

    fig, ax = plt.subplots()
    
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    rho_grid_plot, velocity_grid_plot = create_sinus_density()
    density_grid_plot = equilibruim_distribution(rho_grid_plot, velocity_grid_plot)
    rho_grid_plot_1D = rho_grid_plot.sum(axis=1)
    max_index = np.argmax(rho_grid_plot_1D)

    plot_arr = np.zeros(tmax)
    
    for i in range(tmax) :
        print(i, '/', tmax, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, test=True)
        plot_arr[i] = rho_grid_plot[max_index].sum()
        rho_grid_plot = rho(density_grid_plot)
        
    plot_arr = ((plot_arr/plot_arr.mean()) - 1)
    x = np.arange(300)
    plot_axis_arr = []
    
    for amplitude in plot_arr :
        plot_axis_arr.append(amplitude*np.sin(x*2*np.pi/len(x)))
    
    plot_axis_arr = np.array(plot_axis_arr)
    period = len(plot_axis_arr)//8
    chosen_indices = [period*i for i in range(8)]
    plot_axis_arr = plot_axis_arr[chosen_indices]
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i / len(plot_axis_arr)) for i in range(len(plot_axis_arr))]
    
    for idx, arr in enumerate(plot_axis_arr) :
        ax.plot(x, arr, color=colors[idx])
        
    
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    
    plt.xlabel('Y axis')
    plt.ylabel('Amplitude')
    plt.savefig(result_repo+'density_axis_plot.png')
    plt.close()

def plot_density_on_time() :
    omegas = [.3, .5, .7, 1, 1.3, 1.5, 1.7]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    tmax = 5000
    plt.figure()
    
    for idx, omega in enumerate(omegas) :
        collision_function = lambda density_grid : collision_term(density_grid, omega)
        rho_grid_plot, velocity_grid_plot = create_sinus_density()
        density_grid_plot = equilibruim_distribution(rho_grid_plot, velocity_grid_plot)
        rho_grid_plot_1D = rho_grid_plot.sum(axis=1)
        max_index = np.argmax(rho_grid_plot_1D)

        plot_arr = np.zeros(tmax)
        
        for i in range(tmax) :
            print(i, '/', tmax, end='\r')
            density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, test=True)
            plot_arr[i] = rho_grid_plot[max_index].sum()
            rho_grid_plot = rho(density_grid_plot)
            
        plot_arr = (plot_arr/plot_arr.mean()) - 1  
        
        x = np.arange(tmax)
        local_max_indices = argrelextrema(plot_arr, np.greater)[0]


        plt.plot(x[local_max_indices], plot_arr[local_max_indices], color=colors[idx], label=str(omega))

    plt.xlabel('Time')
    plt.ylabel('Maximal density')
    plt.legend()
    plt.savefig(result_repo+'density_time_plot_with_omegas.png')
    plt.close()
    
def plot_velocity_on_time() :
    omegas = [.3, .5, .7, 1, 1.3, 1.5, 1.7]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    tmax = 5000
    plt.figure()
    for idx, omega in enumerate(omegas) :
        collision_function = lambda density_grid : collision_term(density_grid, omega)
        rho_grid_plot, velocity_grid_plot = create_sinus_velocity(epsilon=.01)
        density_grid_plot = equilibruim_distribution(rho_grid_plot, velocity_grid_plot)
        mu_grid_plot = np.linalg.norm(velocity_grid_plot, axis=0)
        mu_grid_plot_1D = mu_grid_plot.sum(axis=0)
        max_index = np.argmax(mu_grid_plot_1D)

        plot_arr = np.zeros(tmax)
        
        for i in range(tmax) :
            print(i, '/', tmax, end='\r')
            density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, test=True)
            plot_arr[i] = np.linalg.norm(mu(density_grid_plot), axis=0)[:, max_index].sum()
            
        plot_arr = (plot_arr/plot_arr.mean())
        
        x = np.arange(tmax)
        local_max_indices = argrelextrema(plot_arr, np.greater)[0]

        plt.plot(x[local_max_indices], plot_arr[local_max_indices], color=colors[idx], label=str(omega))

    plt.xlabel('Time')
    plt.ylabel('Maximal velocity')
    plt.legend()
    plt.savefig(result_repo+'velocity_time_plot_with_omegas.png')
    plt.close()

def plot_velocity_on_viscosity() :
    omegas = np.arange(0, 2, 0.1)
    vis = np.zeros(omegas.shape)
    eps = .01
    tmax = 5000
    plt.figure()
    for idx, omega in enumerate(omegas) :
        collision_function = lambda density_grid : collision_term(density_grid, omega)
        rho_grid_plot, velocity_grid_plot = create_sinus_velocity(epsilon=eps)
        density_grid_plot = equilibruim_distribution(rho_grid_plot, velocity_grid_plot)
        rho_grid_plot_1D = rho_grid_plot.sum(axis=1)
        max_index = np.argmax(rho_grid_plot_1D)

        plot_arr = np.zeros(tmax)
        
        for i in range(tmax) :
            print(i, '/', tmax, end='\r')
            density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, test=True)
            plot_arr[i] = rho_grid_plot[max_index].sum()
            rho_grid_plot = rho(density_grid_plot)
            
        plot_arr = (plot_arr/plot_arr.mean()) - 1  
        x = np.arange(tmax)
        local_max_indices = argrelextrema(plot_arr, np.greater)[0]
        
        viscosities = ((-1)*np.log((np.array(plot_arr[local_max_indices]) / (plot_arr[local_max_indices][0]))) * (velocity_grid_plot.shape[1] / (2*np.pi))**2) / x[local_max_indices]
        viscosity = viscosities[1:].mean()
        print(omega, viscosity)
        vis[idx] = viscosity
    
    vis_theo = ((1/omegas) - 1/2)/3
    plt.plot(omegas, vis, color='blue', label='experiment')
    plt.plot(omegas, vis_theo, color='black', label='theory')
    plt.xlabel('omegas')
    plt.ylabel('viscosities')
    plt.legend()
    plt.savefig(result_repo+'velocity_viscosity_plot_with_omegas_2.png')
    plt.close()
    return
   
if __name__ == "__main__" :
         
    omega = 1
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    plot_rho(create_sinus_density(epsilon=.1)[0], file="sinus_density.png")
    animate(file="sinus_density.mp4", collision=collision_function, create_grid=create_sinus_density, frames=200, interval=100)
    animate(file="sinus_velocity.mp4", collision=collision_function, create_grid=create_sinus_velocity, frames=400, interval=100)
    plot_density_on_time()
    plot_velocity_on_viscosity()
    plot_velocity_on_time()
    plot_density_on_axis()