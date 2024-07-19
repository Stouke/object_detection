import numpy as np
import tensorflow as tf
from pykitti import raw
import os
import cv2
from fusion_model import create_fusion_model
import xml.etree.ElementTree as ET

# Specify the dataset location
basedir = 'C:/Users/n2309064h/Desktop/Multimodal_code/kitti'
date = '2011_09_26'
drive = '0005'
label_file = 'C:/Users/n2309064h/Desktop/Multimodal_code/kitti/2011_09_26/2011_09_26_drive_0005_sync/tracklet_labels.xml'

# Verify that the calibration files exist
calib_files = [
    os.path.join(basedir, date, 'calib_cam_to_cam.txt'),
    os.path.join(basedir, date, 'calib_imu_to_velo.txt'),
    os.path.join(basedir, date, 'calib_velo_to_cam.txt')
]

for filepath in calib_files:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

# Load the dataset
dataset = raw(basedir, date, drive)

# Parse the tracklet labels
def parse_tracklet_labels(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()
    labels = []

    for tracklet in root.findall('tracklet'):
        object_type = tracklet.find('objectType').text
        h = float(tracklet.find('h').text)
        w = float(tracklet.find('w').text)
        l = float(tracklet.find('l').text)

        for pose in tracklet.findall('poses/item'):
            frame = int(pose.find('frame').text)
            tx = float(pose.find('tx').text)
            ty = float(pose.find('ty').text)
            tz = float(pose.find('tz').text)

            labels.append({
                'frame': frame,
                'type': object_type,
                'h': h,
                'w': w,
                'l': l,
                'tx': tx,
                'ty': ty,
                'tz': tz
            })
    return labels

labels = parse_tracklet_labels(label_file)

# Function to convert point clouds to depth and reflectance maps
def point_cloud_to_maps(point_cloud, image_shape):
    depth_map = np.zeros(image_shape)
    reflectance_map = np.zeros(image_shape)
    
    for point in point_cloud:
        x, y, z, reflectance = point
        if z > 0:
            u = int((x * 0.54) / z + image_shape[1] / 2)
            v = int((y * 0.54) / z + image_shape[0] / 2)
            if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
                depth_map[v, u] = z
                reflectance_map[v, u] = reflectance
    return depth_map, reflectance_map

# Load the YOLOv3 model
net = cv2.dnn.readNet(os.path.join('C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/yolov3/yolov3.weights'), 
                      os.path.join('C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/yolov3/yolov3.cfg'))

# Load the classes
classes = []
with open(os.path.join('C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/yolov3/coco.names'), 'r') as f:
    classes = f.read().splitlines()

# Function to preprocess the input images for YOLOv3
def preprocess_image(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(image.shape) < 3 or image.shape[2] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

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
    
    return boxes, class_ids, confidences

# Load the fusion model
num_classes = 2  # Number of classes: pedestrian and car
fusion_model = create_fusion_model((416, 416, 1), num_classes)

# Assign labels based on detected objects and fusion model predictions
def assign_label(class_ids, fusion_prediction):
    for class_id in class_ids:
        if classes[class_id] == 'car':
            return 1
        elif classes[class_id] == 'pedestrian':
            return 0
    if fusion_prediction[0][1] > 0.5:  # Assuming index 1 is 'car'
        return 1
    elif fusion_prediction[0][0] > 0.5:  # Assuming index 0 is 'pedestrian'
        return 0
    return -1

# Preprocess a single frame
def preprocess_frame(frame_idx):
    image_shape = (416, 416)
    image = dataset.get_cam2(frame_idx)
    image = np.array(image)  # Ensure image is a NumPy array
    image = cv2.resize(image, image_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    point_cloud = dataset.get_velo(frame_idx)
    depth_map, reflectance_map = point_cloud_to_maps(point_cloud, image_shape)

    # Detect objects in the image
    _, class_ids, _ = detect_objects(image)

    # Resize arrays to (416, 416)
    resized_depth_map = cv2.resize(depth_map, image_shape)
    resized_reflectance_map = cv2.resize(reflectance_map, image_shape)
    resized_camera_image = cv2.resize(image, image_shape)

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

    # Assign a label based on detected objects and fusion model prediction
    label = assign_label(class_ids, fusion_prediction)

    return image, depth_map, reflectance_map, label

# Load a subset of data for training/testing
num_frames = len(dataset.timestamps)
images, depth_maps, reflectance_maps, labels = [], [], [], []

for frame_idx in range(num_frames):
    image, depth_map, reflectance_map, label = preprocess_frame(frame_idx)
    images.append(image)
    depth_maps.append(depth_map)
    reflectance_maps.append(reflectance_map)
    labels.append(label)

images = np.array(images)
depth_maps = np.array(depth_maps)
reflectance_maps = np.array(reflectance_maps)
labels = np.array(labels)

# Ensure all arrays have the same length
assert len(images) == len(depth_maps) == len(reflectance_maps) == len(labels), "Inconsistent number of samples in input data"

# Save preprocessed data
base_dir = 'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data'
np.save(os.path.join(base_dir, 'images.npy'), images)
np.save(os.path.join(base_dir, 'depth_maps.npy'), depth_maps)
np.save(os.path.join(base_dir, 'reflectance_maps.npy'), reflectance_maps)
np.save(os.path.join(base_dir, 'labels.npy'), labels)

np.savez(os.path.join(base_dir, 'kitti_data.npz'), images=images, depth_maps=depth_maps, reflectance_maps=reflectance_maps, labels=labels)
