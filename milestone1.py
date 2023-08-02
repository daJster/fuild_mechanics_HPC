import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# repository for results and plots
result_repo = 'result_xy/'

# seeding for same random values
np.random.seed(1234)
random.seed(12)

# direction array for direction channels
direction = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                      [0, 0, 1, 0, -1, 1, 1, -1, -1]])



def probability_density(f) :
    # calculates the probability density of the grid f, returns rho of f
    return f.sum(axis=0)

def velocity(f) :
    # caluclates the velocity profile of the probability density grid f, the result is a 2 by x_shape by y_shape numpy array
    return np.einsum("ij,jkl->ikl", direction, f)/probability_density(f)


def create_density_grid(y_shape=15, x_shape=10, v_shape=9, rand=.2, uniform=False, uniform_value=(1/9)) :
    # creates a probability density grid with specific options
    
    if uniform : 
        # creates a uniform grid, with value equal to uniform_value 
        density_grid = np.full((v_shape, x_shape, y_shape), uniform_value)
    else :
        # creates randomly values in the grid
        density_grid = np.random.rand(v_shape, x_shape, y_shape)
        density_grid /= density_grid.sum(axis=0)
        
    # create holes in the matrix
    if rand :
        for y in range(y_shape) :
            for x in range(x_shape) :
                if np.random.random() > rand :
                    # if random value is more than rand, the cell becomes empty
                    density_grid[:, x, y] = np.zeros(v_shape)
                    
    return density_grid


def plot_density_grid(density_grid, file=None) :
    # given the probability density grid f and a file name it plots rho of f 
    plt.figure()
    total_density = probability_density(density_grid)
    plt.imshow(total_density, cmap='Blues')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(result_repo+file)
    plt.close()


#---------------------------------------------------------------------------------------------
def streaming2D(arr, direction, test=False, collision=False, boundary=False, pressure=False) :
    # given the options runs a streaming process rolling the density grid automatons
    
    # deep copy the arr (probablity density grid) in an accumulator 
    acc = np.copy(arr)
    
    if pressure :
        # if pressure operator is given apply pressure on the liquid
        arr = pressure(arr)
    
    # propagates the cells in all 9 channels using np.roll
    for i in range(direction.shape[1]) :
        arr[i, :, :] = np.roll(arr[i, :, :], shift=(direction[1, i], direction[0, i]), axis=(0, 1))
  
    if boundary :
        # if boundary opperator is given apply boundary on the grid
        arr = boundary(arr)      
        
    if collision :
        # if collision operator is given apply collision on the grid cells
       arr = collision(arr)
       
    if test :
        # test if the mass is conserved through each streaming
        try :
            # compare the accumulator with the modified array
            assert np.isclose(arr.sum(), acc.sum(), rtol=10**(-2)) # testing if the mass is conserved after one stream
        except :
            print('AssertionError : the mass is not conserved during the stream.\n probability before : ', np.sum(arr), ' probablity after :',  np.sum(acc))
            raise AssertionError
        
    
    # return the modified array
    return arr
#---------------------------------------------------------------------------------------------

def animate(file=None, frames=200, interval=100, velocity_active=False, collision=None, pressure=False, cmap="Blues", create_grid=(lambda : create_density_grid(x_shape=30, y_shape=30)), boundary=False) :
    # given the proper options we run the animation process on a created_grid, and use streaming2D every frame to make an .mp4 animation.
    
    # create probability density grid
    density_grid_animate = create_grid()
    total_density = probability_density(density_grid_animate)
    
    # initiates figure
    fig = plt.figure()
    
    if velocity_active :
        # if activated gives a first image of the velocity
        if velocity_active == 'norm' :
            # saves norm of the two directions x and y
            im  = plt.imshow(np.linalg.norm(velocity(density_grid_animate), axis=0)[:, :], animated=True, cmap=cmap)
        else :
            # otherwise saves only the x axis velocity
            im = plt.imshow(velocity(density_grid_animate)[0, :, :], animated=True, cmap=cmap)
    else :
        # if not activated gives a first image of density
        im = plt.imshow(total_density, animated=True, cmap=cmap)
        
    # initiate counter i
    i = 0
    
    def updatefig(frame) :
        # loop function to update the figure
        nonlocal density_grid_animate, i
        i += 1
        
        # run the streaming and save it
        density_grid_animate = streaming2D(density_grid_animate, direction, collision=collision, test=True, boundary=boundary, pressure=pressure)
        
        # saving the frame in the wanted format
        if velocity_active :
            if velocity_active == 'norm' :
                frame  = np.linalg.norm(velocity(density_grid_animate), axis=0)[:, :]
            else  :
                frame  = velocity(density_grid_animate)[0, :, :]
        else :
            frame = density_grid_animate.sum(axis=0)
       
        im.set_array(frame)
        print('frame :', i, "/", frames, end='\r')
        return im,

    # initiate the animation object and run it with the right options, number of rames, interval (speed), color, ...
    ani = FuncAnimation(fig, updatefig, blit=True, frames=frames, interval=interval)
    cbar = fig.colorbar(im)
    plt.gca().invert_yaxis()
    # save the animation
    ani.save(result_repo+file, writer='ffmpeg')



if __name__ == "__main__":
    # create a randomized grid
    density_grid = create_density_grid(x_shape=30, y_shape=30)
    plot_density_grid(density_grid, file='density_grid.png')
    
    # stream once the created grid
    density_grid = streaming2D(density_grid, direction, test=True)
    plot_density_grid(density_grid, file='density_grid_plusone.png')
    
    # run animation on the grid
    animate(file='density_animation.mp4', frames=200)
    
    # create a uniform grid and specifiy direction channels by color
    test_directions_grid = create_density_grid(x_shape=5, y_shape=5, uniform=True, rand=False)
    for i in range(9) :
        test_directions_grid[i, 2, 2] = i

    test_directions_grid = streaming2D(test_directions_grid, direction, test=True)
    plot_density_grid(test_directions_grid, file='test_directions.png')
    
    
    # create uniform grid
    density_grid_animate = create_density_grid(x_shape=300, y_shape=300, uniform=True, rand=False)
    offset_x = 10
    offset_y = 0 
    # filling the center of the grid (square)
    for x in range(50) :
        for y in range(50) :
            density_grid_animate[:, y + offset_y, x + offset_x] = [1 for k in range(9)]
    plot_density_grid(density_grid_animate, file='testing_xy.png')