import numpy as np

# Load preprocessed data
data = np.load('C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/kitti_data.npz')
labels = data['labels']

# Count the number of samples for each label
unique, counts = np.unique(labels, return_counts=True)
label_distribution = dict(zip(unique, counts))

print("Label Distribution:", label_distribution)
