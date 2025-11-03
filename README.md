# Human-activity-recognition-by-using-phone-data
1. Overview

This project performs Human Activity Recognition (HAR) using the UCI HAR Dataset, which contains sensor readings from smartphones (accelerometer and gyroscope).
A 1D Convolutional Neural Network (CNN) model is trained to classify activities such as walking, sitting, standing, and others based on extracted statistical features.

The project includes:
	•	Data loading and preprocessing
	•	Feature scaling and reshaping for CNN input
	•	Deep learning model training (1D CNN)
	•	Evaluation and visualization
	•	Model saving and single-sample inference

⸻

2. Dataset Information

Dataset: UCI HAR Dataset

Description
	•	Data collected from smartphones worn by 30 volunteers performing six daily activities.
	•	Sensors: Accelerometer and Gyroscope.
	•	Preprocessed into 561 feature vectors for each window of 2.56 seconds.

Activities (Labels):
	1.	Walking
	2.	Walking Upstairs
	3.	Walking Downstairs
	4.	Sitting
	5.	Standing
	6.	Laying

⸻

3. Project Structure

├── UCI_HAR_Dataset/
│   ├── train/
│   │   ├── X_train.txt
│   │   ├── y_train.txt
│   │   └── subject_train.txt
│   ├── test/
│   │   ├── X_test.txt
│   │   ├── y_test.txt
│   │   └── subject_test.txt
│   ├── activity_labels.txt
│   └── features.txt
│
├── har_train_cnn.py
├── uicihar_scaler.joblib
├── har_activity_labels.joblib
├── har_cnn_model.h5
└── README.md


⸻

4. Installation & Requirements

Step 1: Clone the project

git clone https://github.com/yourusername/HAR-CNN.git
cd HAR-CNN

Step 2: Install dependencies

pip install numpy pandas scikit-learn tensorflow matplotlib seaborn joblib

Step 3: Download dataset

Download from the UCI repository
and extract into a folder named UCI_HAR_Dataset in the project root.

⸻

5. How to Run

Train and Evaluate

python har_train_cnn.py

This will:
	•	Load and preprocess the dataset
	•	Train a CNN model
	•	Display loss and accuracy plots
	•	Print a classification report
	•	Save the trained model and preprocessing objects

Saved files:
	•	har_cnn_model.h5 — Trained CNN model
	•	uicihar_scaler.joblib — Feature scaler
	•	har_activity_labels.joblib — Activity label mapping

⸻

6. Model Architecture

Layer Type	Output Shape	Parameters
Conv1D (64 filters, kernel=7)	(None, 561, 64)	512
BatchNormalization	(None, 561, 64)	256
Conv1D (64 filters, kernel=5)	(None, 561, 64)	20480
BatchNormalization	(None, 561, 64)	256
MaxPooling1D	(None, 280, 64)	0
Dropout(0.3)	(None, 280, 64)	0
Conv1D (128 filters, kernel=3)	(None, 280, 128)	24704
BatchNormalization	(None, 280, 128)	512
MaxPooling1D	(None, 140, 128)	0
Dropout(0.4)	(None, 140, 128)	0
GlobalAveragePooling1D	(None, 128)	0
Dense(128, relu)	(None, 128)	16512
Dropout(0.4)	(None, 128)	0
Dense(6, softmax)	(None, 6)	774

Total Parameters: ~70K
Optimizer: Adam
Loss: Categorical Crossentropy

⸻

7. Results
	•	Model Accuracy: ~93–96% (varies slightly by seed)
	•	Loss and accuracy curves are plotted automatically.
	•	A confusion matrix visualizes performance across activities.

⸻

8. How to Predict New Data

Example for inference after training:

import joblib
import numpy as np
import tensorflow as tf

# Load saved model and objects
model = tf.keras.models.load_model("har_cnn_model.h5")
scaler = joblib.load("uicihar_scaler.joblib")
labels = joblib.load("har_activity_labels.joblib")

# Example: new sample (561-feature vector)
sample = np.random.rand(1, 561)
sample_scaled = scaler.transform(sample)
sample_reshaped = sample_scaled.reshape((1, 561, 1))

pred = model.predict(sample_reshaped)
pred_label = np.argmax(pred)
print("Predicted activity:", labels[pred_label + 1])


⸻

9. Future Improvements
	•	Use raw sensor signals from Inertial Signals/ for time-series CNN-LSTM modeling
	•	Apply data augmentation on motion windows
	•	Use transformer-based models for temporal attention
	•	Optimize for mobile deployment using TensorFlow Lite

⸻

Would you like me to format this as a downloadable README.md file (for GitHub)?
I can generate and give you the file directly.
