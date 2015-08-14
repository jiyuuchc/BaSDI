#! /bin/bash

#SBATCH --mail-user=jyu@uchc.edu
#SBATCH -N 1
#SBATCH --mem=24g
#SBATCH --qos=general
#SBATCH --partition=gpu

#SBATCH -o jupyter.log

unset LD_LIBRARY_PATH
# cd chioso

# source /home/FCAM/jyu/work/sg/bin/activate

#XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 jupyter-lab --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/home/FCAM/jyu/work/
poetry run jupyter-lab --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/home/FCAM/jyu/work

