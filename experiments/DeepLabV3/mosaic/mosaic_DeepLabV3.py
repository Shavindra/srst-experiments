# %%

import sys
import os

## SET UP PATHS
import sys
sys.path.append(f'/home/sfonseka/dev/SRST/srst-dataloader/experiments/DeepLabV3')  # This is /home/sfonseka/dev/SRST/srst-dataloader

from train import train_dlv3 as trainer

trainer('mozaik', mask_count=10, epochs=3)