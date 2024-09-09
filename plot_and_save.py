import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import unet_model

def load_model(model_path):

    device = torch.device('cpu')
    
    model = unet_model.UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to perform inference
def predict(model, image):
    with torch.no_grad():
        output = model(image)
    return output

# Function to post-process the output
def postprocess_output(output):
    output = torch.sigmoid(output)  # Apply sigmoid to get probability map
    output = output.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy array
    return (output > 0.95).astype(np.uint8) * 255  # Threshold and convert to binary image


 # Path to your saved model
image_path = 'nature_imag_cropped.png'  # Path to a test image


# Load the model
model_path = 'unet_line_thinning_model.pth'  #shoule be the same as defined in the training block
model = load_model(model_path)

# Preprocess the image
input_image = preprocess_image(image_path)


# Perform inference
output = predict(model, input_image)

# Post-process the output
result = postprocess_output(output) #filtered output



# %%
hand_drawn_line =  preprocess_image('hand_drawn_cropped.png')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
im1 = input_image.squeeze(0).squeeze(0) #original image
im2 = result #filtered inference image
im3 = hand_drawn_line.squeeze(0).squeeze(0) #ground truth
im4 = output.squeeze(0).squeeze(0) #unfiltered inference image


# %%
import matplotlib.pyplot as plt
import numpy as np

def save_three_arrays_side_by_side(im1, im2, im3, save_path, titles=None, fig_size=(18, 6), dpi=300):
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size)
    
    # Function to display a single image
    def display_image(ax, im, title):
        im_display = ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.axis('off')  # Hide axes
        if title:
            ax.set_title(title)
        #if im.ndim == 2:
            #fig.colorbar(im_display, ax=ax)
    
    # Display the three images
    display_image(ax1, im1, titles[0] if titles else None)
    display_image(ax2, im2, titles[1] if titles else None)
    display_image(ax3, im3, titles[2] if titles else None)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close(fig)
    
    print(f"Side-by-side image saved to {save_path}")

# Example usage
# im1, im2, and im3 are your NumPy arrays
save_three_arrays_side_by_side(im1, im3, im2, 'three_images_comparison.png', titles=['original','ground truth',
                                                                                     'unet-processed' ])

# %% unfiltered output plot
save_three_arrays_side_by_side(im1, im3, im4, 'three_images_comparison_unfiltered.png', titles=['original','ground truth',
                                                                                     'unet-processed' ])