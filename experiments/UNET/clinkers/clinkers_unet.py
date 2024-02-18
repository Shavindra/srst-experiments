# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append('/home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET')  # This is /home/sfonseka/dev/SRST/srst-dataloader

from train import train_unet

train_unet('clinkers', mask_count=500, epochs=20)
