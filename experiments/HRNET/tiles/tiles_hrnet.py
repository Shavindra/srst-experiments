# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append(f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/HRNET')  # This is /home/sfonseka/dev/SRST/srst-dataloader

from train import train_hrnet as trainer

trainer('tiles', mask_count=500, epochs=25)