#!/bin/bash

# List of directories
# dirs=("asphalt" "bike" "clinkers"  "grass" "mosaic" "tiles")
# dirs=("grass" "mosaic" "tiles")
dirs=("asphalt" "bike" "clinkers")

# Get the current working directory
output_dir=$(pwd)

time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting at ${time}" &>> "${output_dir}/output.log"

for dir in "${dirs[@]}"; do
    echo "Submitting job for ${dir}" &>> "${output_dir}/output.log"
    cd /home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET/${dir}
    sbatch ${dir}_slurm.job &>> output.log
done

echo "All jobs submitted" &>> "${output_dir}/output.log"