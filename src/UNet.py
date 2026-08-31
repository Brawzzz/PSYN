#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class HexToolDataset(Dataset):

    """
    dataset preparation for HexTool 
    """

    def __init__(self, data_dir : str = "./data/datasets/UNet/"):

        """
        :params data_dir: path to dataset directory
        """

        #------------------------------
        self.img_dir    = os.path.join(data_dir, "images")
        self.mask_dir   = os.path.join(data_dir, "masks")
        
        self.filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]

    #================================================================================#
    def __len__(self):

        """
        return the number of files found in the datasets
        """

        #------------------------------
        return len(self.filenames)

    #================================================================================#
    def __getitem__(self, idx : int) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        """
        load and prepare the input for the network

        :params idx: index of the pacth to load  

        :return (tensor_img, tensor_target): the input image and the mask as a tensor 
        """

        #------------------------------
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        #------------------------------
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0 
        img = np.expand_dims(img, axis=0) 
        
        #------------------------------
        mask_name = img_name.replace('.png', '.npy')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        mask_angles = np.load(mask_path)
        (h, w)      = mask_angles.shape

        #------------------------------
        target = np.zeros((3, h, w), dtype=np.float32)

        valid_pixels    = mask_angles != -100.0
        angles_rad      = np.deg2rad(mask_angles[valid_pixels])
        
        target[0][valid_pixels] = np.cos(2 * angles_rad)
        target[1][valid_pixels] = np.sin(2 * angles_rad)
        target[2][valid_pixels] = 1.0 
        
        #------------------------------
        tensor_img      = torch.from_numpy(img)
        tensor_target   = torch.from_numpy(target)
        
        return(tensor_img, tensor_target)


#========================================================================================================#
#========================================================================================================#
class MSELoss(nn.Module):

    """
    define the loss function of the network, here we adapt the MSE function 
    to ignore the error link to the pixels wich are not corresponding to fibers 
    """
    
    def __init__(self):

        super().__init__()
        self.mse = nn.MSELoss(reduction='none') 

    #================================================================================#
    def forward(self, preds, targets) -> float:

        """
        :params preds:      [Batch, 2, Height, Width] -> Prediction (Vx, Vy)
        :params targets:    [Batch, 3, Height, Width] -> Truth (Vx, Vy, Masque)
        """

        #------------------------------
        true_val  = targets[:, :2, :, :]
        mask      = targets[:, 2:3, :, :] 
        
        loss_pixel  = self.mse(preds, true_val) 
        masked_loss = loss_pixel * mask
        
        num_valid_pixels = mask.sum() + 1e-8 

        final_loss = masked_loss.sum() / (num_valid_pixels * 2)
        
        return final_loss


#========================================================================================================# 
#========================================================================================================#
class DoubleConv(nn.Module):

    """
    Apply two succesive 2D-Convolution with a batch normalisation and a ReLU activation :

    (Conv2d -> BatchNorm -> ReLU)
    --> -->
    (Conv2d -> BatchNorm -> ReLU)

    """

    def __init__(self, in_channels, out_channels):

        super().__init__()
        
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    #================================================================================#
    def forward(self, x):
        return self.conv(x)


#========================================================================================================#
#========================================================================================================#
class UNetOrientation(nn.Module):

    """
    UNet architecture to predict pixels orientation
    """

    def __init__(self, in_channels=1, out_channels=2):

        super().__init__()
        
        #---------- ENCODEUR
        self.down_1 = DoubleConv(in_channels, 64)
        self.pool_1 = nn.MaxPool2d(2)

        self.down_2 = DoubleConv(64, 128)
        self.pool_2 = nn.MaxPool2d(2)

        self.down_3 = DoubleConv(128, 256)
        self.pool_3 = nn.MaxPool2d(2)
        
        #---------- BOTTLENECK
        self.bottleneck = DoubleConv(256, 512)
        
        #---------- DECODEUR
        self.upConv3    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_3        = DoubleConv(512, 256)
        
        self.upConv2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_2        = DoubleConv(256, 128)
        
        self.upConv1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_1        = DoubleConv(128, 64)
        
        #---------- OUTPUT
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    #------------------------------
    def forward(self, x):

        #---------- ENCODEUR        
        x1 = self.down_1(x)
        x2 = self.down_2(self.pool_1(x1))
        x3 = self.down_3(self.pool_2(x2))
        
        #---------- BOTTLENECK
        bot = self.bottleneck(self.pool_3(x3))
        
        #---------- DECODEUR
        u3 = self.upConv3(bot)
        u3 = self.up_3(torch.cat([u3, x3], dim=1))
        
        u2 = self.upConv2(u3)
        u2 = self.up_2(torch.cat([u2, x2], dim=1))
        
        u1 = self.upConv1(u2)
        u1 = self.up_1(torch.cat([u1, x1], dim=1))
        
        #---------- OUTPUT
        out = self.outc(u1)
        out = nn.functional.normalize(out, p=2, dim=1)
        
        return out


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def train_UNet(dataset_dir="./data/dataset/UNet", 
               epochs=20, 
               batch_size=4, 
               learning_rate=1e-4,
               model_dir : str = "./models/UNet_hxtl.pth") -> None:

    """
    function to train UNet model

    Parameters
    ----------
    dataset_dir     : path to the directory containing the data
    epoch           : number of epochs for the training phase
    batch_size      : size of the batch use for one epoch of training
    learning_rate   : learning rate set for training phase
    """

    #------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on : {device}")

    #------------------------------
    dataset     = HexToolDataset(data_dir=dataset_dir)
    dataloader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("Dataset loaded")
    
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

    torch.save(model.state_dict(), model_dir)

    
#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
if __name__ == "__main__":
    
    dataset = HexToolDataset(data_dir="./data/datasets/UNet")
    print(f"Dataset loaded with : {len(dataset)} patchs")
    
    (image, target) = dataset[0]
    
    img_display = image.squeeze().numpy() 

    vx_display      = target[0].numpy()
    vy_display      = target[1].numpy()
    valid_display   = target[2].numpy()
    
    (fig, axs) = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(img_display, cmap='gray')
    axs[0].set_title("Image (Entrée U-Net)")
    
    axs[1].imshow(valid_display, cmap='gray')
    axs[1].set_title("Masque de Validité (1=Fibre, 0=Fond)")
    
    axs[2].imshow(vx_display, cmap='coolwarm', vmin=-1, vmax=1)
    axs[2].set_title("Composante Vx : cos(2θ)")
    
    axs[3].imshow(vy_display, cmap='coolwarm', vmin=-1, vmax=1)
    axs[3].set_title("Composante Vy : sin(2θ)")
    
    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()