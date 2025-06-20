🧠 Brain Tumor Detection Using MRI Scans – Deep Learning Project

📄 Overview

This project implements a deep learning-based system to classify brain MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor. It leverages transfer learning with the Xception architecture, augmented image preprocessing, and performance evaluation using key classification metrics. The solution is designed for clinical relevance, achieving high test accuracy and robust predictions.

📂 Project Structure
training/: Directory containing labeled training MRI images
testing/: Directory for testing data
model_checkpoints/: Stores model weights for each epoch
my_model.keras: Final trained model for deployment

results/: Evaluation plots and confusion matrices

🧪 Dataset

Format: MRI images organized in folders by class
Classes: glioma, meningioma, pituitary, notumor
Input size: 299×299 pixels (scaled and normalized)
Source: Assumed from Kaggle or medical imaging archives

📈 Model Architecture

Base model: Xception (pretrained on ImageNet, with top layer removed)

Added Layers:

Flatten
Dropout(0.3)
Dense(128, ReLU)
Dropout(0.25)
Dense(4, Softmax)

📊 Training Configuration

Loss Function: Categorical Crossentropy
Optimizer: Adamax (lr=0.001)
Metrics: Accuracy, Precision, Recall
Data Augmentation: Image brightness, resizing, shuffling
Callbacks: ModelCheckpoint (saves model weights after each epoch)
Epochs: 10
Batch Size: 32

🖼️ Visualization and Evaluation

Plots:
Accuracy, Loss, Precision, Recall over epochs

Tools: Matplotlib, Seaborn

Reports:

Classification report using sklearn

Confusion matrix heatmap

📌 Key Results (from last few epochs)

Epoch	Accuracy	Precision	Recall	Loss
7	99.40%	99.18%	99.18%	0.0123
10	98.95%	98.54%	98.54%	0.0474

✅ Model Prediction Example

The model includes a predict() function that:

Loads and resizes an image
Classifies it using the trained model
Displays the image with predicted label and confidence score
Usage Example:
python
image_path = '/path/to/image.jpg'
predict(image_path)

🧠 Sample Output:

Tumor: Glioma (Confidence: 98.56%)

✅ Evaluation Metrics

Precision: High value indicates low false positives
Recall: Ensures low false negatives
F1-Score: Balanced performance
Confusion Matrix: Visual indicator of class-wise accuracy

💾 Model Saving and Deployment

Final model saved as: my_model.keras
Can be loaded via:
from tensorflow.keras.models import load_model
model = load_model('my_model.keras')

🔮 Future Enhancements

Incorporate 3D MRI slices using 3D CNN
Add tumor segmentation using U-Net
Include patient metadata (age, history) for contextual diagnosis
Build a front-end tool for radiologists to interact with predictions

⚙️ Requirements

Python 3.7+
TensorFlow 2.x
Keras
Matplotlib, Seaborn, Pillow, Pandas, Scikit-learn
Google Colab
