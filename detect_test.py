import cv2
import numpy as np
import tensorflow as tf
import os

# Function to preprocess the input images
def preprocess_image(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(image.shape) < 3 or image.shape[2] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

# Function to resize the input arrays to (416, 416)
def resize_array(array, target_shape=(416, 416)):
    print(f"Resizing array with shape: {array.shape}")
    if len(array.shape) == 4 and array.shape[0] == 1:
        array = np.squeeze(array, axis=0)
    if len(array.shape) == 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    return cv2.resize(array, target_shape, interpolation=cv2.INTER_AREA)

# Function to perform YOLOv3 detection
def detect_objects(image):
    height, width = image.shape[:2]
    image = preprocess_image(image)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    detections = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, boxes, class_ids, confidences

# Define base directory
base_dir = 'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data'
base_dir_yolov = 'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)'

# Load the YOLOv3 model
net = cv2.dnn.readNet(os.path.join(base_dir_yolov, 'yolov3/yolov3.weights'), os.path.join(base_dir_yolov, 'yolov3/yolov3.cfg'))

# Load the classes
classes = []
with open(os.path.join(base_dir_yolov, 'yolov3/coco.names'), 'r') as f:
    classes = f.read().splitlines()

# Load the early fusion model
fusion_model = tf.keras.models.load_model(os.path.join(base_dir, 'trained_fusion_model.h5'))

# Load your input images (e.g., from depth_maps.npy and reflectance_maps.npy)
images = np.load(os.path.join(base_dir, 'val_images.npy'))
depth_maps = np.load(os.path.join(base_dir, 'val_depth_maps.npy'))
reflectance_maps = np.load(os.path.join(base_dir, 'val_reflectance_maps.npy'))

# Define class labels for the fusion model
fusion_classes = ['pedestrian', 'car']  # Only two classes

# Perform object detection for each pair of input images
for i in range(len(images)):
    print(f"Processing sample {i}")
    
    # YOLOv3 detection
    processed_image, boxes, class_ids, confidences = detect_objects(images[i])
    
    # Debugging: Print shapes of arrays
    print(f"Image shape: {images[i].shape}")
    print(f"Depth map shape: {depth_maps[i].shape}")
    print(f"Reflectance map shape: {reflectance_maps[i].shape}")

    # Resize arrays to (416, 416)
    try:
        resized_depth_map = resize_array(depth_maps[i], (416, 416))
        resized_reflectance_map = resize_array(reflectance_maps[i], (416, 416))
        resized_camera_image = resize_array(images[i], (416, 416))
    except cv2.error as e:
        print(f"Error resizing arrays: {e}")
        continue

    # Ensure the resized_camera_image has one channel
    if len(resized_camera_image.shape) == 2:
        resized_camera_image = np.expand_dims(resized_camera_image, axis=-1)
    elif resized_camera_image.shape[2] == 3:
        resized_camera_image = cv2.cvtColor(resized_camera_image, cv2.COLOR_RGB2GRAY)
        resized_camera_image = np.expand_dims(resized_camera_image, axis=-1)
    
    # Early fusion model prediction
    depth_map = resized_depth_map.reshape((1, 416, 416, 1))
    reflectance_map = resized_reflectance_map.reshape((1, 416, 416, 1))
    camera_image = resized_camera_image.reshape((1, 416, 416, 1))
    fusion_prediction = fusion_model.predict([camera_image, depth_map, reflectance_map])
    
    # Map fusion model predictions to class labels
    fusion_predictions = []
    for score, label in zip(fusion_prediction[0], fusion_classes):
        fusion_predictions.append({'class': label, 'score': score})
    
    # Combine YOLOv3 and fusion model results
    combined_predictions = []
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        combined_predictions.append({
            'class': classes[class_id],
            'bbox': box,
            'confidence': confidence
        })
    
    combined_predictions.append({'fusion_predictions': fusion_predictions})
    
    # Display the results
    cv2.imshow('Detection Result', processed_image)
    print(f'Combined Predictions for sample {i}:')
    for pred in combined_predictions:
        if 'fusion_predictions' in pred:
            print('  Fusion Predictions:')
            for fusion_pred in pred['fusion_predictions']:
                print(f'    - Class: {fusion_pred["class"]}, Score: {fusion_pred["score"]:.4f}')
        else:
            print(f'  - Class: {pred["class"]}, BBox: {pred["bbox"]}, Confidence: {pred["confidence"]:.4f}')
    
    if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break
    cv2.destroyAllWindows()
