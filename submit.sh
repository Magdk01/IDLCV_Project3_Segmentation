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
module load python3/3.11.9
module load cuda/12.1

source ~/02516/IDLCV_Project3_Segmentation/venv/bin/activate

# Move to your working directory
cd ~/02516/IDLCV_Project3_Segmentation


# --- Model selection: unet or simple ---
MODEL=${1:-unet}

python3 train.py \
  --model $MODEL \
  --epochs 25 \
  --batch-size 2 \
  --img-size 256 \
  --lr 1e-4 \
  --loss wbce \
  --pos-weight 3.0 \
  --output-dir ./hpc_outputs