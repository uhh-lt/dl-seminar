# one is no fit for the job! You can do that either in the sbatch
# command line or here with the other settings.
#SBATCH --job-name=test_tensorflow
#SBATCH --partition=gpu --qos=gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --time=00:10:00
# Never forget that! Strange happenings ensue otherwise.
#SBATCH --export=NONE

set -e # Good Idea to stop operation on first error.

source /sw/batch/init.sh

# Load environment modules for your application here.

module unload env

#module load /sw/BASE/env/2017Q1-gcc-openmpi /sw/BASE/env/cuda-8.0.44_system-gcc
module load env/cuda-9.0.176_gcc-6.4.0
module load python/3.6.1

# Actual work starting here. You might need to call
# srun or mpirun depending on your type of application
# for proper parallel work.
# Example for a simple command (that might itself handle
# parallelisation).
echo "Hello World! I am $(hostname -s) greeting you!"
echo "Also, my current TMPDIR: $TMPDIR"

echo "nvidia-smi:"

srun bash -c 'nvidia-smi'

# Let's pretend our started processes are working on a
# predetermined parameter set, looking up their specific
# parameters using the set number and the process number
# inside the batch job.
export PARAMETER_SET=42
# Simplest way to run an identical command on all allocated
# cores on all allocated nodes. Use environment variables to
# tell apart the instances.
srun bash -c 'echo "process $SLURM_PROCID \
(out of $SLURM_NPROCS total) on $(hostname -s) \
parameter set $PARAMETER_SET"'

srun bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_PROCID LD_LIBRARY_PATH=/sw/compiler/cuda-9.0.176/lib64:$HOME/cuda/lib64/ python3 $HOME/convolutional.py'