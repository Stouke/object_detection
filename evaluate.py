import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained fusion model
fusion_model = tf.keras.models.load_model(r'C:\Users\n2309064h\Desktop\Multimodal_code\object_detector_(yolov3_w_fusion)\preprocessed_data\trained_fusion_model.h5')

# Load the validation data
X_val_camera = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\object_detector_(yolov3_w_fusion)\preprocessed_data\val_images.npy')
X_val_depth = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\object_detector_(yolov3_w_fusion)\preprocessed_data\val_depth_maps.npy')
X_val_reflectance = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\object_detector_(yolov3_w_fusion)\preprocessed_data\val_reflectance_maps.npy')
y_val = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\object_detector_(yolov3_w_fusion)\preprocessed_data\val_labels.npy')  # Make sure you have labels for validation

# Visual Inspection Function
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
sample_indices = np.random.choice(len(X_val_camera), num_samples_to_display, replace=False)

for idx in sample_indices:
    display_sample(X_val_camera[idx], X_val_depth[idx], X_val_reflectance[idx], y_val[idx], idx)

# Predict on the validation data
y_pred_prob = fusion_model.predict([X_val_camera, X_val_depth, X_val_reflectance])
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print the performance metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
