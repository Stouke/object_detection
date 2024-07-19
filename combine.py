import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import os

# Load the trained fusion model
fusion_model = tf.keras.models.load_model(r'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/trained_fusion_model.h5')

# Dummy function for YOLOv3 predictions (replace with actual YOLOv3 code)
def yolo_predict(image):
    # Replace with actual YOLOv3 prediction code
    return [{'class': 'car', 'bbox': [50, 50, 150, 150]}, {'class': 'pedestrian', 'bbox': [200, 200, 250, 250]}]

# Function to predict using fusion model
def fusion_model_predict(image, depth_map, reflectance_map):
    input_data = [np.expand_dims(image, axis=0), np.expand_dims(depth_map, axis=0), np.expand_dims(reflectance_map, axis=0)]
    prediction = fusion_model.predict(input_data)
    return prediction

# Function to combine YOLOv3 and fusion model predictions
def combine_predictions(yolo_predictions, fusion_predictions):
    combined_predictions = []
    for yolo_pred in yolo_predictions:
        obj_class = yolo_pred['class']
        bbox = yolo_pred['bbox']
        combined_pred = {
            'class': obj_class,
            'bbox': bbox,
            'fusion_score': fusion_predictions[0][0]  # Replace with appropriate fusion score logic
        }
        combined_predictions.append(combined_pred)
    return combined_predictions

# Load the validation data
X_val_camera = np.load(r'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/val_images.npy')
X_val_depth = np.load(r'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/val_depth_maps.npy')
X_val_reflectance = np.load(r'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/val_reflectance_maps.npy')
y_val = np.load(r'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/val_labels.npy')  # Make sure you have labels for validation

# Loop through validation data and combine predictions
for i in range(len(X_val_camera)):
    image = X_val_camera[i]
    depth_map = X_val_depth[i]
    reflectance_map = X_val_reflectance[i]
    
    # YOLOv3 predictions
    yolo_predictions = yolo_predict(image)
    
    # Fusion model predictions
    fusion_predictions = fusion_model_predict(image, depth_map, reflectance_map)
    
    # Combine predictions
    combined_predictions = combine_predictions(yolo_predictions, fusion_predictions)
    
    # Print or store combined predictions
    print(f"Combined Predictions for sample {i}: {combined_predictions}")
