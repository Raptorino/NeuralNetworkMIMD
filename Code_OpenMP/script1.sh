#!/bin/bash 
#SBATCH -N 1 		#N num nodos
#SBATCH -n 1
#SBATCH --exclusive


export OMP_NUM_THREADS=12
perf stat ./$1 $2 $3 $4

# How to use:
#module add gcc/10.2.0 
# Compile: 		gcc -fopenmp -std=c99 -Ofast -lm common.c nn-main.c -o nnmain
# Slurn Execution: 	sbatch -o code_Output.txt ./script1.sh nnmain


