#!/bin/bash

# Generic options:

#SBATCH --account=bdncl03  # Run job under project <project>
#SBATCH --time=1:0:0         # Run for a max of 1 hour

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
#SBATCH --nodes=1          # Resources from a single node
#SBATCH --gres=gpu:1       # One GPU per node (plus 25% of node CPU and RAM per GPU)

# Run commands:

# assume IBM Watson Machine Learning Community Edition is installed
# in conda environment "tf-gpu")

echo Job running at `hostname` and starting at `date`
echo classification script
nvidia-smi  # Display available gpu resources
source /users/cparton/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu

# Place other commands he@re
python face_regression.py --b 512 --layers 8 --lr 0.001
python face_regression.py --b 512 --layers 8 --lr 0.0001
python face_regression.py --b 512 --layers 8 --lr 0.00001
python face_regression.py --b 512 --layers 8 --lr 0.000001
python face_regression.py --b 512 --layers 8 --lr 0.0000001
python face_regression.py --b 512 --layers 8 --lr 0.00000001
echo end of job, results in ./logs_bede
