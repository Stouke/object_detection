from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def create_lidar_feature_extractor(input_shape):
    input_depth = Input(shape=input_shape)
    input_reflectance = Input(shape=input_shape)
    
    def cnn_branch(input):
        x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        return x
    
    depth_features = cnn_branch(input_depth)
    reflectance_features = cnn_branch(input_reflectance)
    
    combined_features = concatenate([depth_features, reflectance_features])
    
    model = Model(inputs=[input_depth, input_reflectance], outputs=combined_features)
    return model

def create_fusion_model(input_shape, num_classes):
    input_camera = Input(shape=input_shape, name='camera_input')
    cnn_camera = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input_camera)
    cnn_camera = MaxPooling2D((2, 2))(cnn_camera)
    cnn_camera = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(cnn_camera)
    cnn_camera = MaxPooling2D((2, 2))(cnn_camera)
    cnn_camera = Flatten()(cnn_camera)
    cnn_camera = BatchNormalization()(cnn_camera)
    cnn_camera = Dropout(0.6)(cnn_camera)

    lidar_model = create_lidar_feature_extractor((416, 416, 1))

    combined_features = concatenate([cnn_camera, lidar_model.output])
    combined_features = BatchNormalization()(combined_features)
    combined_features = Dropout(0.6)(combined_features)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined_features)
    output = Dense(num_classes, activation='softmax')(x)  # Using softmax for multi-class classification

    model = Model(inputs=[input_camera, lidar_model.input[0], lidar_model.input[1]], outputs=output)
    return model
