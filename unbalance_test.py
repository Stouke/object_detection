import numpy as np

# Load preprocessed data
data = np.load('C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/kitti_data.npz')
labels = data['labels']

# Filter the data to include only pedestrian (label 1) and car (label 3)
valid_indices = np.where((labels == 1) | (labels == 3))[0]
filtered_labels = labels[valid_indices]

# Remap the labels: pedestrian (1) -> 0, car (3) -> 1
remapped_labels = np.where(filtered_labels == 1, 0, 1)

# Count the occurrences of each class
unique_labels, counts = np.unique(remapped_labels, return_counts=True)
print(f"Unique labels: {unique_labels}")
print(f"Counts: {counts}")
