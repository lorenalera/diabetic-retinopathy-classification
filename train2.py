import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt

# Load training and validation datasets
train_dataset = image_dataset_from_directory(
    'Dataset2/train',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'  # Use 'categorical' for one-hot encoded labels
)

validation_dataset = image_dataset_from_directory(
    'Dataset2/test',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=False  # Ensure no shuffling during evaluation
)

# Define the number of classes
num_classes = 3

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
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)
print(f'Test accuracy: {accuracy}')

# Get predictions and true labels
y_pred = []
y_true = []

for images, labels in validation_dataset:
    y_pred_batch = model.predict(images)
    y_true_batch = labels.numpy()

    y_pred.extend(y_pred_batch)
    y_true.extend(y_true_batch)

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Binarize the true labels
y_true_binarized = label_binarize(y_true.argmax(axis=1), classes=range(num_classes))

# Calculate and print manual accuracy
manual_accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
print(f'Manual accuracy: {manual_accuracy}')

# Print classification report
print(classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1), target_names=train_dataset.class_names))

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
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
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
plot_auc_curve(y_true_binarized, y_pred, num_classes)

# Save the model
model.save('efficientnet_model.h5')
