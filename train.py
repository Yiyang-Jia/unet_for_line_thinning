import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import unet_model

### Train/val data preparation

# transform images to a standard size
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = unet_model.LineDataset('thick_lines_synthetic/', 'thin_lines_synthetic/', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for both splits
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

## Training loops

# Initialize model, loss, and optimizer
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model = unet_model.UNet().to(device)
model_path = 'unet_line_thinning_model_black.pth' 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
train_loss_list = [100]
val_loss_list = [100]
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
            val_losses=[]
            for (val_data, val_target) in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_losses.append(criterion(val_output, val_target).item())
            val_loss = np.mean(val_losses)
            val_loss_list[0] = val_loss


# Training loop
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = unet_model.UNet().to(device)
model_path = 'unet_line_thinning_model.pth' 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
train_loss_list = [100]
val_loss_list = [100]
if os.path.exists(model_path): #continue training if a weight file already exists
    model.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
            val_losses=[]
            for (val_data, val_target) in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_losses.append(criterion(val_output, val_target).item())
            val_loss = np.mean(val_losses)
            val_loss_list[0] = val_loss


# Training loop
num_epochs = 1

for epoch in range(1, num_epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = criterion(output, target)
        train_loss_list.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            val_losses=[]
            for (val_data, val_target) in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_losses.append(criterion(val_output, val_target).item())
            val_loss = np.mean(val_losses)
        
        if val_loss < min(val_loss_list):
           torch.save(model.state_dict(), model_path)
        val_loss_list.append(val_loss )
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_loss.item():.6f} ' 
                  f'  validation loss: {val_loss:.6f}') 
