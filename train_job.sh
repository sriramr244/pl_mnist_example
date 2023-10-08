#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --account=rrg-pfieguth
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10      # CPU cores/threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.8 cuda cudnn

# Activate your enviroment
source ~/envs/hello/bin/activate

# Variables for readability
logdir=/home/kkaai/scratch/saved
datadir=/home/kkaai/scratch/data
# datadir=$SLURM_TMPDIR

tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
    python ~/workspace/pl_mnist_example/train.py \
    --model Conv \
    --dataloader MNIST \
    --batch_size 32 \
    --epoch 10 \
    --num_workers 10 \
    --logdir ${logdir} \
    --data_dir  ${datadir}
