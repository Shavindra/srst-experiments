# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append('../../..')  # This is /home/sfonseka/dev/SRST/srst-dataloader

from train import train_unet

train_unet('bike', mask_count=20)