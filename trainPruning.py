import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt


def apply_image_pruning(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Ensure the grayscale image is in 8-bit single-channel format
    if gray_image.dtype != np.uint8:
        gray_image = cv2.convertScaleAbs(gray_image)

    # Convert the grayscale image to binary
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Apply image pruning (thinning)
    pruned_image = cv2.ximgproc.thinning(binary_image)

    # Convert back to RGB
    pruned_image_rgb = cv2.cvtColor(pruned_image, cv2.COLOR_GRAY2RGB)

    return pruned_image_rgb


def preprocess_image(image):
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Apply image pruning for enhancement
    pruned_image = apply_image_pruning(image_np)

    # Convert back to tensor for compatibility with ImageDataGenerator
    image_tf = tf.convert_to_tensor(pruned_image, dtype=tf.float32)

    return image_tf.numpy()


# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    # rescale=1. / 255
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    # rescale=1. / 255
)

# Load training and validation datasets
train_generator = train_datagen.flow_from_directory(
    'Dataset2/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' for one-hot encoded labels
)

validation_generator = validation_datagen.flow_from_directory(
    'Dataset2/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=False  # Ensure no shuffling during evaluation
)

# Define the number of classes
num_classes = len(train_generator.class_indices)

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for one-hot encoded labels
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=40,
    validation_data=validation_generator
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy}')

# Get predictions and true labels
validation_generator.reset()
y_pred = model.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=-1)
y_true = validation_generator.classes

# Calculate and print manual accuracy
manual_accuracy = accuracy_score(y_true, y_pred)
print(f'Manual accuracy: {manual_accuracy}')

# Print classification report
print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


# Calculate and plot AUC for each class
def plot_auc_curve(y_true, y_pred, num_classes):
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc='lower right')
    plt.show()


# Plot AUC curve
plot_auc_curve(y_true, y_pred, num_classes)

# Save the model
model.save('efficientnet_model_pruned.h5')
