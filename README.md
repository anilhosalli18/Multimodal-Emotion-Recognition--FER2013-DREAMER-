# Multimodal-Emotion-Recognition--FER2013-DREAMER-
This project implements a multimodal emotion recognition system by combining facial expression data from the FER2013 dataset and physiological signals from the DREAMER dataset. Deep learning models extract visual and biosignal features, and fusion techniques improve classification accuracy and robustness for reliable human emotion analysis.

I developped a multimodal emotion recognition platform to analyze the emotions of job candidates and other.

I analye facial, vocal and textual emotions, using mostly deep learning based approaches. I deployed a web app using Flask :

<img width="2879" height="1303" alt="Screenshot 2026-01-24 192710" src="https://github.com/user-attachments/assets/e6a824ab-b65b-4264-a410-f9acb5a9feed" />

The tool can be accessed from the WebApp repository, by installing the requirements and launching main.py .

# Technologies




<img width="845" height="125" alt="techno" src="https://github.com/user-attachments/assets/1a00d193-eba4-446d-bcd7-4e0b26a53577" />


# DATASETS Information

| Dataset Name | Modality              | Data Type                | Description                                          | Usage                               |
| ------------ | --------------------- | ------------------------ | ---------------------------------------------------- | ----------------------------------- |
| FER2013      | Facial Expressions    | Grayscale Images (48×48) | Facial emotion images with 7 emotion classes         | Visual feature extraction using CNN |
| DREAMER      | Physiological Signals | EEG, ECG Signals         | Multimodal biosignal dataset for emotion recognition | Physiological feature extraction    |




| Model Name                         | Modality            | Purpose                               | Framework              |
| ---------------------------------- | ------------------- | ------------------------------------- | ---------------------- |
| CNN (Convolutional Neural Network) | Image (FER2013)     | Facial emotion classification         | TensorFlow / PyTorch   |
| 1D-CNN / Feature-based ML          | EEG & ECG (DREAMER) | Physiological signal classification   | Scikit-learn / PyTorch |
| Multimodal Fusion Network          | Combined            | Feature-level / Decision-level fusion | Custom                 |

DATASET LINKS:
FER2013 on Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

DREAMER on Zenodo: https://zenodo.org/records/546113 (Get access)

# Methodology
VIdeo Analysis
<img width="1655" height="857" alt="image" src="https://github.com/user-attachments/assets/57d7c9ab-d1d4-41d6-8b99-4014752f45fb" />
# 🔄 Pipeline & Model Description

The system implements a real-time and uploaded video–based emotion recognition pipeline that captures facial data, performs emotion classification, and presents analytical results through an interactive web interface.

First, the application launches either a live webcam stream or processes an uploaded video. Each frame is analyzed to detect faces using the Histogram of Oriented Gradients (HOG) method. The detected face region is automatically cropped, zoomed, and resized to 48 × 48 pixels before being passed to a pre-trained deep learning model for inference.

For every detected face, the model predicts emotion probabilities across multiple classes including Anger, Happiness, Fear, Sadness, Surprise, Disgust, and Neutral. The system computes the dominant emotion, probability distribution, total probability sum, and overall prediction accuracy. Each session is stored in a history panel displaying timestamp, input source (Live/Upload), number of detected faces, emotion scores, preview video, and action controls.

In addition to emotion classification, the system generates a VAD (Valence–Arousal–Dominance) interpretation for the detected emotion. This provides psychological insight such as emotional positivity/negativity, intensity level, and confidence state, which enhances human–emotion understanding beyond basic classification.

The emotion recognition model is based on the Xception convolutional neural network architecture, selected for its superior accuracy and robustness during experimental evaluation. The model was optimized using:

i.   Data augmentation for improved generalization

ii.  Early stopping to prevent overfitting

iii. Learning rate reduction on plateau for stable convergence

iv.  L2 regularization for model stability

v.   Class weight balancing to handle dataset imbalance

The best-performing trained model was saved and deployed for real-time inference in the web application.


# DEMO'S
<img width="2877" height="1443" alt="Screenshot 2026-01-24 192843" src="https://github.com/user-attachments/assets/2191fcfc-e9b6-434b-91f5-6b7e63047eff" />


Output Screen Explanation (Your Recorded Sessions)


**✅ 1. Session History Dashboard**

Displays all previously recorded emotion detection sessions.

Each row represents one analyzed video session (Live or Uploaded).

Shows the date and time of processing.


**✅ 2. Input Source Information**

Indicates whether the input came from:

Live Webcam

Uploaded Video

Helps track testing scenarios and data sources.


**✅ 3. Dominant Emotion Detection**

Shows the predicted dominant emotion (e.g., Happy, Neutral).

Determined based on the highest probability score among emotion classes.


**✅ 4. Face Detection Count**

Displays the average number of faces detected in the video.

Useful to validate multi-face or single-face detection accuracy.


**✅ 5. Emotion Probability Scores**

Shows probability values for each emotion class:

Anger

Happiness

Fear

Sadness

Surprise

Disgust

Neutral

Example: Happiness = 0.96 indicates strong confidence.



**✅ 6. Probability Sum Validation**

Confirms that all emotion probabilities sum to 1.00.

Ensures correct softmax normalization of model output.



**✅ 7. Prediction Accuracy**

Displays the overall confidence or accuracy score for the prediction.

Example shown: 94.1% accuracy.



✅ 8. Video Preview Panel

Shows a thumbnail preview of the processed video.

Allows playback for visual verification.



**✅ 9. Action Controls**

Provides a Delete button to remove unwanted session records.



**✅ 10. VAD Interpretation Panel**

Displays Valence–Arousal–Dominance analysis for the detected emotion.

Example for Happy:

Valence: Positive (pleasant emotion)

Arousal: Medium (moderate intensity)

Dominance: High (confident state)

Adds psychological interpretation to the emotion output.



**✅ 11. Multi-Session Comparison**

Allows users to compare multiple sessions side by side.

Helps evaluate consistency of model predictions across different inputs.



**✅ 12. User-Friendly Interface**

Clean dark theme UI with structured columns.

Real-time analytics visualization for easy understanding.


## How to Use it?

To use the web app :

Clone the project locally


Go in the WebApp folder


Run `$ pip install -r requirements.txt``


Launch python main.py
