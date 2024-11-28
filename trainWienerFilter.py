import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import cv2

# Paths
train_dir = 'Dataset2/train'
test_dir = 'Dataset2/test'

# Image Parameters
img_height, img_width = 224, 224
batch_size = 32


# High-Pass Filter Function
def apply_high_pass_filter(image):
    # Apply the filter to each channel separately if the image is RGB
    if len(image.shape) == 3 and image.shape[-1] == 3:  # Check if the image has 3 channels (RGB)
        channels = cv2.split(image)
        filtered_channels = [apply_high_pass_filter_to_channel(channel) for channel in channels]
        filtered_image = cv2.merge(filtered_channels)
    else:
        filtered_image = apply_high_pass_filter_to_channel(image)
    return filtered_image


def apply_high_pass_filter_to_channel(channel):
    # Convert to float32
    channel = np.float32(channel)

    # Perform DFT
    dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # Radius of the low-frequency region
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize to range [0, 255]
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)


# Data Generator with High-Pass Filter
def data_generator_with_high_pass_filter(directory, batch_size, img_height, img_width, class_mode='categorical'):
    def preprocessing_function(image):
        return apply_high_pass_filter(image)

    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size,
                                            class_mode=class_mode,
                                            color_mode='rgb')
    return generator


train_generator = data_generator_with_high_pass_filter(train_dir, batch_size, img_height, img_width)
test_generator = data_generator_with_high_pass_filter(test_dir, batch_size, img_height, img_width)

# Model Creation
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_generator, epochs=10, validation_data=test_generator)
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Calculate additional metrics
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))

print('Classification Report')
target_names = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))