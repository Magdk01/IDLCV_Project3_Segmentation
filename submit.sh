#!/bin/bash
#BSUB -q gpuv100
#BSUB -J segmentation_models
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -o hpc_outputs/%J.out
#BSUB -e hpc_outputs/%J.err
#BSUB -B
#BSUB -N

# Load environment
module load python3/3.11.9
module load cuda/12.1

source ~/02516/IDLCV_Project3_Segmentation/venv/bin/activate

# Move to your working directory
cd ~/02516/IDLCV_Project3_Segmentation

echo "[INFO] Starting SimpleEncoder training..."
python3 train.py \
  --model simple \
  --epochs 25 \
  --batch-size 2 \
  --img-size 256 \
  --lr 1e-4 \
  --loss wbce \
  --pos-weight 3.0 \
  --output-dir ./hpc_outputs

# rename result file (if it exists)
if [ -f "./hpc_outputs/results.txt" ]; then
    mv ./hpc_outputs/results.txt ./hpc_outputs/results_simple.txt
fi

echo "[INFO] Starting UNet training..."
python3 train.py \
  --model unet \
  --epochs 25 \
  --batch-size 2 \
  --img-size 256 \
  --lr 1e-4 \
  --loss wbce \
  --pos-weight 3.0 \
  --output-dir ./hpc_outputs

# rename result file (if it exists)
if [ -f "./hpc_outputs/results.txt" ]; then
    mv ./hpc_outputs/results.txt ./hpc_outputs/results_unet.txt
fi

echo "[INFO] Both models finished successfully!"
