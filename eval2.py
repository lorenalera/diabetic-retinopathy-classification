import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image):
    # Apply bilateral filter to preserve edges
    bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # Convert to tensor
    bilateral_filtered_image = tf.convert_to_tensor(bilateral_filtered_image, dtype=tf.float32)
    # Maximize contrast
    contrast_image = tf.image.adjust_contrast(bilateral_filtered_image, contrast_factor=2.0)
    # Convert to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(contrast_image)
    # Apply gamma correction
    gamma_corrected_image = tf.image.adjust_gamma(grayscale_image, gamma=0.7)
    return gamma_corrected_image

# Set parameters
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 3
TEST_DATASET_DIR = "Dataset2/test"
REPORTS_DIR = "reports"  # Directory to save reports

# Class labels
class_labels = ['Mild_Moderate', 'No_DR', 'Proliferate_DR_Severe']

# Load the trained model
model = tf.keras.models.load_model('efficientnet_dr_classification.h5')

# Create a test data generator
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')

# Get predictions and true labels
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

# Calculate overall accuracy
overall_accuracy = np.mean(y_true == y_pred)
print(f'Overall Accuracy: {overall_accuracy:.4f}')

# Classification Report
classification_rep = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write("============================================\n\n")
    f.write(f'Overall Accuracy: {overall_accuracy:.4f}\n\n')
    for label, metrics in classification_rep.items():
        if label != 'accuracy':
            f.write(f'Class: {label}\n')
            for metric, value in metrics.items():
                f.write(f'{metric.capitalize()}: {value:.4f}\n')
            f.write('\n')
    f.write("============================================\n")
print("Classification Report saved.")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
plt.close()
print("Confusion Matrix saved.")

# AUC and ROC Curve for Each Class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
    roc_auc[i] = roc_auc_score(y_true == i, y_pred_prob[:, i])
    print(f'AUC for {class_labels[i]}: {roc_auc[i]:.4f}')

# Plotting all ROC curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange', 'red', 'purple']
for i, color in zip(range(NUM_CLASSES), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.savefig(os.path.join(REPORTS_DIR, "roc_curves.png"))
plt.close()
print("ROC Curves saved.")
