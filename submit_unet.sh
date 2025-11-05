#!/bin/bash
#BSUB -q c02516
#BSUB -J segmentation_models_unet
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o hpc_outputs/unet_%J.out
#BSUB -e hpc_outputs/unet_%J.err
#BSUB -B
#BSUB -N

module load python3/3.11.9
module load cuda/12.1

cd ~/Desktop/IDLCV_Project3_Segmentation/

source .venv/bin/activate

python3 train.py \
  --model unet \
  --epochs 25 \
  --batch-size 2 \
  --img-size 256 \
  --lr 1e-4 \
  --loss focal \
  --pos-weight 3.0 \
  --output-dir ./hpc_outputs