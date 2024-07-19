import matplotlib.pyplot as plt
import numpy as np
from preprocessing import images, depth_maps, reflectance_maps, labels

# Function to display an image, depth map, reflectance map, and label
def display_sample(image, depth_map, reflectance_map, label, index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Sample {index} - Label: {label}')
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image')
    
    axes[1].imshow(depth_map, cmap='gray')
    axes[1].set_title('Depth Map')
    
    axes[2].imshow(reflectance_map, cmap='gray')
    axes[2].set_title('Reflectance Map')
    
    plt.show()

# Randomly select a few samples to visualize
num_samples_to_display = 5
sample_indices = np.random.choice(len(images), num_samples_to_display, replace=False)

for idx in sample_indices:
    display_sample(images[idx], depth_maps[idx], reflectance_maps[idx], labels[idx], idx)
