import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import argparse
import time
from milestone1 import streaming2D, direction, create_density_grid, velocity, result_repo
from milestone2 import collision_term
from milestone6 import create_sliding_lid_boundaries


def save_mpiio(comm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array_like
        Portion of the array on this MPI processes. This needs to be a
        two-dimensional array.
    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)
    
    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (nx.item(), ny.item())
                        })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    offsety = np.zeros_like(local_ny)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()
    
    
def send_cell_boundary(comm, probability_density_grid, direction_dict) :
    """
    Send and receive boundary values of the probability density grid.

    Parameters:
    - comm (MPI.Comm): MPI communicator object.
    - probability_density_grid (numpy.ndarray): Probability density grid.
    - direction_dict (dict): Dictionary containing communication directions.

    Returns:
    numpy.ndarray: Updated probability density grid with boundary values exchanged.
    """
    
    # send to north, receive from south
    recv_buffer = np.copy(probability_density_grid[:, 0, :])
    send_buffer = np.copy(probability_density_grid[:, -2, :])
    comm.Sendrecv(send_buffer, direction_dict["north"]["to"], recvbuf=recv_buffer, source= direction_dict["north"]["from"])
    probability_density_grid[:, 0, :] = recv_buffer

    # send to south, receive from north
    recv_buffer = np.copy(probability_density_grid[:, -1, :])
    send_buffer = np.copy(probability_density_grid[:, 1, :])
    comm.Sendrecv(send_buffer, direction_dict["south"]["to"], recvbuf=recv_buffer, source= direction_dict["south"]["from"])
    probability_density_grid[:, -1, :] = recv_buffer

    # send to west, receive from east
    recv_buffer = np.copy(probability_density_grid[:, :, -1])
    send_buffer = np.copy(probability_density_grid[:, :, 1])
    comm.Sendrecv(send_buffer, direction_dict["west"]["to"], recvbuf=recv_buffer, source= direction_dict["west"]["from"])
    probability_density_grid[:, :, -1] = recv_buffer

    # send to east, receive from west
    recv_buffer = np.copy(probability_density_grid[:, :, 0])
    send_buffer = np.copy(probability_density_grid[:, :, -2])
    comm.Sendrecv(send_buffer, direction_dict["east"]["to"], recvbuf=recv_buffer, source= direction_dict["east"]["from"])
    probability_density_grid[:, :, 0] = recv_buffer

    return probability_density_grid


def parallel_sliding_lid(frames, node_shape, grid_size, omega, debug, save):
    """
    Simulates fluid dynamics using the lattice Boltzmann method in a parallelized manner.

    Parameters:
    - frames (int): Number of simulation frames.
    - node_shape (tuple): Shape of individual processing nodes in the Cartesian grid (x, y).
    - grid_size (tuple): Total dimensions of the simulation grid (x, y).
    - omega (float): Relaxation parameter for collision term.

    Returns:
    None
    """
    
    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    # Extract node shape dimensions
    node_shape_x = node_shape[0]
    node_shape_y = node_shape[1]
    
    # Calculate dimensions of each node
    node_dim_x = grid_size[0] // node_shape_x
    node_dim_y = grid_size[1] // node_shape_y
    
    # Create Cartesian communicator
    cartcomm = comm.Create_cart(dims=[node_shape_x, node_shape_y],  # Dimensions of the Cartesian grid
                            periods=(False, False),            # Whether the grid is periodic in each dimension
                            reorder=False)                    # Whether ranks can be reordered for optimization

    
    # Get coordinates of the current process
    coords = cartcomm.Get_coords(rank)

    # Define dictionary for communication directions
    direction_dict = {
        "north": {
            "from": cartcomm.Shift(0, -1)[0],
            "to": cartcomm.Shift(0, -1)[1]
         },
        "south": {
            "from": cartcomm.Shift(0, 1)[0],
            "to": cartcomm.Shift(0, 1)[1]
         },
        "west": {
            "from": cartcomm.Shift(1, -1)[0],
            "to": cartcomm.Shift(1, -1)[1]
         },
        "east": {
            "from": cartcomm.Shift(1, 1)[0],
            "to": cartcomm.Shift(1, 1)[1]
         }
    }
    
    # Define process positions for boundary handling
    proc_position = {
        "east": direction_dict["east"]["to"] < 0,
        "west": direction_dict["west"]["to"] < 0,
        "north": direction_dict["north"]["to"] < 0,
        "south": direction_dict["south"]["to"] < 0
    }
    
    # Define the collision function
    collision_function = lambda density_grid: collision_term(density_grid, omega)
    
    # Create initial probability density grid
    probability_density_grid = create_density_grid(y_shape=node_dim_y, x_shape=node_dim_x, rand=False, uniform=True)
    
    # Synchronize processes
    cartcomm.Barrier()
    if rank == 0 and debug :
        print("-------------- start")
        start_time = time.time()
        
    # Iterate over frames
    for frame in range(frames):
        if debug :
            print('proc ', rank, ' : ', frame, '/', frames, "\t coords : ", coords[0], ',', coords[1], "\t shape : ", node_dim_x, ',', node_dim_y)
        
        # Send boundary values
        probability_density_grid = send_cell_boundary(cartcomm, probability_density_grid, direction_dict)
        
        # Calculate velocity
        u = velocity(probability_density_grid)
        
        # Perform streaming and collision operations
        probability_density_grid = streaming2D(probability_density_grid, direction, collision=collision_function, \
            boundary=create_sliding_lid_boundaries(
                north=proc_position["north"],
                south=proc_position["south"],
                east=proc_position["east"],
                west=proc_position["west"]), 
                test=True)
        
    # Extract velocity components for saving
    v = u[:, 1:-1, 1:-1]
    
    if save :
        # Save velocity components using MPI I/O
        save_mpiio(cartcomm, result_repo + "velocity_x.npy", v[0])
        save_mpiio(cartcomm, result_repo + "velocity_y.npy", v[1])
    
    # Synchronize processes
    cartcomm.Barrier()
    if rank == 0 and debug : 
            print("-------------- End")
            end_time = time.time()
            print("duration : ", end_time - start_time, " seconds")
    
    return


if __name__ == "__main__" :
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("-w", "--omega", type=float, default=0.7, help="Relaxation parameter for collision term")
    parser.add_argument('-gs', '--grid_size', nargs='+', type=int, default=(300, 300), help="Dimensions of the simulation grid (x y)")
    parser.add_argument('-ns', '--node_shape', nargs='+', type=int, default=(2, 2), help="Shape of individual processing nodes (x y)")
    parser.add_argument("-f", "--frames", type=int, default=400, help="Number of simulation frames")
    parser.add_argument("-debug", "--debug", action='store_true', default=False, help="enables debugging")
    parser.add_argument("-save", "--save", action='store_true', default=False, help="save .npy files")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the parallel_sliding_lid function with the parsed arguments
    parallel_sliding_lid(frames=args.frames,
                        node_shape=args.node_shape,
                        grid_size=args.grid_size,
                        omega=args.omega,
                        debug=args.debug,
                        save = args.save)