# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append('../../..')  # This is /home/sfonseka/dev/SRST/srst-dataloader
sys.path.append('/home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET')  # This is /home/sfonseka/dev/SRST/srst-dataloader/experiments/UNET

from train import train_unet

train_unet('bike-asphalt', mask_count=10, epochs=5)