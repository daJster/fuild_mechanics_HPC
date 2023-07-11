import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import argparse
from milestone1 import streaming2D, direction
from milestone2 import collision_term
from milestone6 import create_sliding_lid_boundaries, create_sliding_lid_grid

def send_cell_boundary(comm, probability_density_grid) :
    north_from, north_to = comm.Shift(1, 1)
    south_from, south_to = comm.Shift(1, -1)
    east_from, east_to = comm.Shift(0, 1)
    west_from, west_to = comm.Shift(0, -1)
    
    
    return


def parallel_sliding_lid(frames, node_shape, grid_size, omega) :
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    node_shape_x = node_shape[0]
    node_shape_y = node_shape[1]
    
    node_dim_x = grid_size[0] // node_shape_x
    node_dim_y = grid_size[1] // node_shape_y
    
    cartcomm = comm.Create_cart(dims=[node_shape_x, node_shape_y],
                                periods=(False, False),
                                reorder=False
                                )
    
    coords = cartcomm.Get_coords(rank)
    
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    
    probability_density_grid = create_sliding_lid_grid(x_shape=node_dim_x, y_shape=node_dim_y)
    
    for frame in range(frames) :
        print('proc ', rank, ' : ', frame, '/', frames, "\t shape : ", node_dim_x, ',', node_dim_y)
        send_cell_boundary(cartcomm, probability_density_grid)
        probability_density_grid = streaming2D(probability_density_grid, direction, collision=collision_function, boundary=create_sliding_lid_boundaries, test=True)
    
    return


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("-w", "--omega", type=float, default=0.7)
    parser.add_argument('-gs', '--grid_size', nargs='+', type=int, default=(300, 300))
    parser.add_argument('-ns', '--node_shape', nargs='+', type=int, default=(2, 2))
    parser.add_argument("-f", "--frames", type=int, default=400)

    args = parser.parse_args()
    
    parallel_sliding_lid(frames=args.frames,
                         node_shape=args.node_shape,
                         grid_size=args.grid_size,
                         omega=args.omega)