import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import xml.etree.ElementTree as ET

# Function to parse tracklet labels
def parse_tracklet_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []

    for item in root.findall('.//item'):
        object_type = item.find('objectType').text
        for pose in item.find('poses'):
            tx = float(pose.find('tx').text)
            ty = float(pose.find('ty').text)
            tz = float(pose.find('tz').text)
            labels.append({
                'type': object_type,
                'tx': tx,
                'ty': ty,
                'tz': tz
            })

    return labels

# Load parsed labels
labels = parse_tracklet_labels(r'C:\Users\n2309064h\Desktop\Multimodal_code\kitti\2011_09_26\2011_09_26_drive_0005_sync\tracklet_labels.xml')

# Map object types to numerical labels
label_map = {'Car': 1, 'Pedestrian': 0}
numerical_labels = [label_map[label['type']] for label in labels]

# Convert to numpy array
labels = np.array(numerical_labels)

# Load preprocessed data
data = np.load('C:/Users/n2309064h/Desktop/Multimodal_code/object_detector_(yolov3_w_fusion)/preprocessed_data/kitti_data.npz')
images = data['images']
depth_maps = data['depth_maps']
reflectance_maps = data['reflectance_maps']

# Ensure all arrays have the same length
assert len(images) == len(depth_maps) == len(reflectance_maps) == len(labels), "Inconsistent number of samples in input data"

# Adjust shapes if necessary
if len(depth_maps.shape) == 5:
    depth_maps = depth_maps.reshape((depth_maps.shape[0], depth_maps.shape[2], depth_maps.shape[3]))
if len(reflectance_maps.shape) == 5:
    reflectance_maps = reflectance_maps.reshape((reflectance_maps.shape[0], reflectance_maps.shape[2], reflectance_maps.shape[3]))

# Ensure that all arrays are of the same shape
assert images.shape == depth_maps.shape == reflectance_maps.shape, "Shapes of input arrays are not the same"

# Filter the data to include only pedestrian (label 0) and car (label 1)
valid_indices = np.where((labels == 0) | (labels == 1))[0]
images = images[valid_indices]
depth_maps = depth_maps[valid_indices]
reflectance_maps = reflectance_maps[valid_indices]
labels = labels[valid_indices]

# Stack the arrays along a new dimension to prepare for splitting
stacked_data = np.stack([images, depth_maps, reflectance_maps], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(stacked_data, labels, test_size=0.2, random_state=42)

# Separate the combined arrays back into individual components
X_train_images, X_train_depth, X_train_reflectance = np.split(X_train, 3, axis=1)
X_test_images, X_test_depth, X_test_reflectance = np.split(X_test, 3, axis=1)

# Flatten the data for comparison
X_train_images_flat = X_train_images.reshape(len(X_train_images), -1)
X_test_images_flat = X_test_images.reshape(len(X_test_images), -1)

# Check for overlap
overlap = np.intersect1d(X_train_images_flat.view([('', X_train_images_flat.dtype)]*X_train_images_flat.shape[1]), 
                         X_test_images_flat.view([('', X_test_images_flat.dtype)]*X_test_images_flat.shape[1]))

print(f"Number of overlapping samples: {len(overlap)}")

# Remove the extra dimension added by np.split
X_train_images = np.squeeze(X_train_images, axis=1)
X_train_depth = np.squeeze(X_train_depth, axis=1)
X_train_reflectance = np.squeeze(X_train_reflectance, axis=1)

X_test_images = np.squeeze(X_test_images, axis=1)
X_test_depth = np.squeeze(X_test_depth, axis=1)
X_test_reflectance = np.squeeze(X_test_reflectance, axis=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_data(images, depth_maps, reflectance_maps, labels, datagen):
    augmented_images, augmented_depth_maps, augmented_reflectance_maps, augmented_labels = [], [], [], []
    for i in range(len(images)):
        img = np.stack([images[i]]*3, axis=-1)  # Convert to 3-channel
        depth_map = np.stack([depth_maps[i]]*3, axis=-1)  # Convert to 3-channel
        reflectance_map = np.stack([reflectance_maps[i]]*3, axis=-1)  # Convert to 3-channel
        label = labels[i]
        for _ in range(5):  # Augment each sample 5 times
            augmented_img = datagen.random_transform(img)
            augmented_depth_map = datagen.random_transform(depth_map)
            augmented_reflectance_map = datagen.random_transform(reflectance_map)
            augmented_images.append(augmented_img[:, :, 0])  # Convert back to single-channel
            augmented_depth_maps.append(augmented_depth_map[:, :, 0])  # Convert back to single-channel
            augmented_reflectance_maps.append(augmented_reflectance_map[:, :, 0])  # Convert back to single-channel
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_depth_maps), np.array(augmented_reflectance_maps), np.array(augmented_labels)

# Augment training data
X_train_images, X_train_depth, X_train_reflectance, y_train = augment_data(X_train_images, X_train_depth, X_train_reflectance, y_train, datagen)

# Ensure all arrays have the same length after augmentation
assert len(X_train_images) == len(X_train_depth) == len(X_train_reflectance) == len(y_train), "Inconsistent number of samples after augmentation"

# Create the model
num_classes = 2  # Since we are predicting two classes: pedestrian and car
model = create_fusion_model((416, 416, 1), num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation loop
fold_no = 1
for train, val in kfold.split(X_train_images, y_train):
    print(f'Training fold {fold_no} ...')
    
    # Train the model with early stopping
    history = model.fit(
        [X_train_images[train], X_train_depth[train], X_train_reflectance[train]],
        tf.keras.utils.to_categorical(y_train[train], num_classes=num_classes),
        validation_data=(
            [X_train_images[val], X_train_depth[val], X_train_reflectance[val]],
            tf.keras.utils.to_categorical(y_train[val], num_classes=num_classes)
        ),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    fold_no += 1

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([X_test_images, X_test_depth, X_test_reflectance], tf.keras.utils.to_categorical(y_test, num_classes=num_classes))
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Save the trained model
model.save('trained_fusion_model.h5')
