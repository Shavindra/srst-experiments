#!/bin/bash

# List of directories
dirs=("asphalt" "bike" "clinkers" "grass" "mosaic" "tiles")

# Get the current working directory
output_dir=$(pwd)

for dir in "${dirs[@]}"; do
    echo "Submitting job for ${dir}" &>> "${output_dir}/output.log"
    cd /home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET/${dir}
    sbatch ${dir}_slurm.job &>> output.log
done

echo "All jobs submitted" &>> "${output_dir}/output.log"