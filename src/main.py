"""
Script        : main.py
Description   : Image analysis for detecting fibers in samples of a composite material called HexTool.
                The script detects the fibres in a sample of this specific material
                and then extracts the regions where the fibres have the same orientation. 

Author        : WEIBEL Hugo, MA Oscar
Date          : 01/01/2025
Version       : 1.0.0
"""

__author__  = "WEIBEL Hugo, MA Oscar"
__copyright__ = "Copyright 2025, Hugo WEIBEL & Oscar MA"
__credits__ = ["M. FORTE DA CRUZ", "MME. LO FUEDO"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "WEIBEL Hugo, MA Oscar"
__email__   = ["hugo03.weibel@gmail.com", "oscarma230@email.com"]
__status__  = "Development" 

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import time
import os
import cv2 as cv
import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

import Sample
import setup as stp
import tools
import data

from UNet import HexToolDataset,UNetOrientation, MSELoss


#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def run(n_config : str):

    """
    
    """

    #------------------------------
    START = time.perf_counter()

    #---------------
    sample = Sample.init(config_path=n_config, n_fret=False)

    sample.process_sample()
    sample.group_regions()
    sample.compute_shapes()

    #---------------
    sample.save_config()
    sample.save()
    sample.print(region=False)

    #---------------
    regions_img = sample.render(n_render=stp.RENDER)
    cv.imwrite(sample.output_path + sample.name + "_regions_all.png", regions_img)

    #---------------
    END  = time.perf_counter()
    TIME = END - START
    print(f"\nExecution time : {TIME:.4f} seconds")

#================================================================================#
def train_UNet(dataset_dir="./data/dataset/UNet", epochs=20, batch_size=4, learning_rate=1e-4):
    
    #------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on : {device}")

    #------------------------------
    print("Loading Dataset...", end="", flush="True")
    dataset     = HexToolDataset(data_dir=dataset_dir)
    dataloader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Loading Dataset ... Done")
    
    #------------------------------
    model       = UNetOrientation(in_channels=1, out_channels=2).to(device)
    criterion   = MSELoss().to(device)
    optimizer   = optim.Adam(model.parameters(), lr=learning_rate)

    #------------------------------
    os.makedirs("./models", exist_ok=True)
    
    for epoch in range(epochs):

        model.train() 
        running_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:

            for images, targets in pbar:
                
                images  = images.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
            
                preds   = model(images)
                loss    = criterion(preds, targets)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        #---------------
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} done | Epoch loss : {epoch_loss:.4f}\n")

        torch.save(model.state_dict(), f"./models/unet_hextool_epoch_{epoch+1}.pth")

#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
# config = tools.arg_parse()

# run(n_config=config)

# sample = Sample.Sample.load(filepath = "./output/hxtl_p25_pre/hxtl_p25_pre.pkl")

# data.build_dataset(n_sample     = sample,
#                    patch_size   = 512,
#                    export_dir   = "./data/datasets/UNet")

train_UNet(dataset_dir="./data/datasets/UNet/",
           epochs=10, 
           batch_size=4,
           learning_rate=1e-4)