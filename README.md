# Milestone 7 Simulation

This is a simulation script for Milestone 7. It uses MPI to run simulations in parallel and provides options to run and plot the results.

## Prerequisites

- Python 3.8 or higher
- mpi4py library
- numpy library
- matplotlib library
- seaborn library
- pandas library 

## How to Run

1. Clone the repository to your local machine.

2. Ensure you have the required dependencies installed. You can install them using the following commands:

   ```bash
   pip3 install -r requirements.txt
   ```
3. Make sure the simulation script milestone7.py is in the same directory as the script you provided. To run the simulation with default settings (4 processes), use the following command:
   ```bash
   ./milestone7
   ```

4. To specify a different number of processes, use the -np flag followed by the desired number. For example, to run with 8 processes:
    ```bash
   ./milestone7 -np 8
   ```

5. Additional command-line arguments can be used to customize the simulation:

    - -w, --omega: Relaxation parameter for collision term (default: 0.7)
    - -gs, --grid_size: Dimensions of the simulation grid (x y) (default: 300 300)
    - -ns, --node_shape: Shape of individual processing nodes (x y) (default: 2 2)
    - -f, --frames: Number of simulation frames (default: 400)
    - -debug, --debug: Enable debugging (default: False)
    - -save, --save: Save .npy files (default: False)

## Example Usages

- Run the simulation with custom grid size, node shape, and omega:

    ```bash
    ./milestone7 -gs 500 500 -ns 3 3 -w 1.2
    ```

- Run the simulation with debugging enabled and save .npy files:

    ```bash
    ./milestone7 -debug -save
    ```

## How to Run (milestone7_wrapper)

1. Clone the repository to your local machine.

2. Ensure you have the required dependencies installed. You can install them using the following commands:

   ```bash
   pip install mpi4py
   ```

3. To run the simulation with default settings (4 processes), use the following command:

    ```bash
   ./milestone7 -run
   ```

4. To run the simulation with default settings (4 processes), use the following command:

    ```bash
   ./milestone7 -run -np 8
   ```

5. Additional command-line arguments can be used to customize the simulation: (same as milestone7)

6. To plot the results of the simulation, use the following command:

    ```bash
   ./milestone7 -plot
   ```

## Notes
- The parallel simulation (milestone7*) depend on other milestones function and utilities. Please keep the other milestones in the same folder for compatible imports.

- The simulation script milestone7.py should be in the same directory as the milestone7 shell script and milestone7_wrapper.py for proper execution.