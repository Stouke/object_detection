import numpy as np
import tensorflow as tf
from pykitti import raw
import os
import cv2

# Specify the dataset location
basedir = 'C:/Users/n2309064h/Desktop/Multimodal_code/kitti'
date = '2011_09_26'
drive = '0005'

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

# Preprocess a single frame
def preprocess_frame(frame_idx):
    image_shape = (416, 416)
    image = dataset.get_cam2(frame_idx)
    image = np.array(image)  # Ensure image is a NumPy array
    image = cv2.resize(image, image_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    point_cloud = dataset.get_velo(frame_idx)
    depth_map, reflectance_map = point_cloud_to_maps(point_cloud, image_shape)

    return image, depth_map, reflectance_map

# Load a subset of data for training/testing
num_frames = len(dataset.timestamps)
images, depth_maps, reflectance_maps, labels = [], [], [], []

for frame_idx in range(num_frames):
    image, depth_map, reflectance_map = preprocess_frame(frame_idx)
    images.append(image)
    depth_maps.append(depth_map)
    reflectance_maps.append(reflectance_map)
    labels.append(1)  # Placeholder for labels, replace with actual label generation

images = np.array(images)
depth_maps = np.array(depth_maps)
reflectance_maps = np.array(reflectance_maps)
labels = np.array(labels)

# Ensure all arrays have the same length
assert len(images) == len(depth_maps) == len(reflectance_maps) == len(labels), "Inconsistent number of samples in input data"

# Save preprocessed data

base_dir = 'C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data'

np.save(os.path.join(base_dir, 'val_images.npy'), images)
np.save(os.path.join(base_dir, 'val_depth_maps.npy'), depth_maps)
np.save(os.path.join(base_dir, 'val_reflectance_maps.npy'), reflectance_maps)
np.save(os.path.join(base_dir, 'val_labels.npy'), labels)

np.savez(os.path.join(base_dir, 'val_kitti_data.npz'), images=images, depth_maps=depth_maps, reflectance_maps=reflectance_maps, labels=labels)