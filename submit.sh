#!/bin/bash
#BSUB -q gpuv100
#BSUB -J action_recognition[5]
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o hpc_outputs/unet_%J_%I.out
#BSUB -e hpc_outputs/unet_%J_%I.err
#BSUB -B
#BSUB -N

# Load environment
module load python3/3.12.11
module load cuda/12.1

# (Optional) Activate your conda/venv
# source ~/venvs/torch/bin/activate

# Move to your working directory
cd $SLURM_SUBMIT_DIR

# Run the training
python3 train.py \
  --epochs 25 \
  --batch-size 2 \
  --img-size 256 \
  --lr 1e-4 \
  --loss wbce \
  --pos-weight 3.0 \
  --output-dir ./hpc_outputs
