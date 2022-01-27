#!/bin/bash

#SBATCH --job-name=mpi_exec
#SBATCH --output=mpi_%j.txt
#SBATCH -N 1 # Num nodes
#SBATCH -n 4 # Num COREs
#SBATCH --partition=nodo.q
#SBATCH --exclusive

hostname

#module load gcc/10.2.0
#module load openmpi/3.0.0

#mpicc prova.c -o prova

mpirun -np $2 ./$1


