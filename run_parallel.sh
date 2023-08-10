#!/bin/bash -x
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --mem=6gb
#SBATCH --time=00:30:00
#SBATCH --partition=dev_multiple
#SBATCH -J HPC_WITH_PYTHON
#SBATCH --output=run_parallel.out
#SBATCH --error=run_parallel.err
#SBATCH --export=ALL

# This script follows this guide : https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues

# modules loaded
module load compiler/gnu/10.2   
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
python3 -m pip install --force-reinstall --no-binary :all: mpi4py matplotlib numpy

cd ${SLURM_SUBMIT_DIR}
echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."
echo "${SLURM_NTASKS} tasks have been succesfully allocated"

# Running the wrapper of parallel code
python3 milestone7_wrapper.py -run >> output_parallel.txt