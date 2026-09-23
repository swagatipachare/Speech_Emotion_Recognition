# Speech Emotion Recognition

A Speech Emotion Recognition (SER) project that uses **Machine Learning and Deep Learning techniques** to classify human speech into different emotional categories. The project includes model development, performance evaluation, visualization, and a Streamlit-based application for emotion prediction.

## 📌 Project Overview

Speech Emotion Recognition is a task in **audio/speech processing** and **machine learning** that aims to identify the emotional state expressed in a person's voice.

This project explores both traditional Machine Learning and Deep Learning approaches for emotion classification. Different models are trained and evaluated, and the results are visualized using confusion matrices and F1-score plots.

## 🎯 Objectives

* Analyze speech/audio data for emotion classification.
* Apply Machine Learning techniques for speech emotion recognition.
* Develop Deep Learning models for emotion classification.
* Compare model performance using evaluation metrics.
* Visualize classification results using confusion matrices and F1-score graphs.
* Provide a Streamlit interface for emotion prediction.

## 🧠 Models Used

### Machine Learning

* Random Forest
* Gradient Boosting
* MLP

### Deep Learning

* RNN
* LSTM
* Bi-LSTM
* CNN-LSTM

The project contains separate notebooks for Machine Learning and Deep Learning experimentation.

## 📊 Model Evaluation

The models are evaluated using classification metrics such as:

* Precision
* Recall
* F1-Score
* Confusion Matrix

The repository contains the generated evaluation results and visualizations in the `Results` directory.

### Evaluation Results

The project includes:

* Confusion Matrix for the deep learning model
* Confusion Matrix for the Random Forest model
* F1-Score visualization
* Classification metrics in CSV format

## 🖥️ Streamlit Application

A Streamlit application is included in `app.py`.

The application allows the user to:

1. Upload an audio file.
2. Upload a trained `.h5` model.
3. Process the uploaded audio/model.
4. Generate an emotion classification result.

The trained model files are not included in the GitHub repository to keep the repository lightweight.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Streamlit
* Jupyter Notebook

## 🔄 Project Workflow

```text
Speech / Audio Data
        ↓
Data Preprocessing
        ↓
Feature Extraction
        ↓
Machine Learning / Deep Learning Models
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Confusion Matrix & F1-Score
        ↓
Streamlit Application
        ↓
Emotion Prediction
```

## 📈 Results

The project provides visual and numerical evaluation results for the developed models.

The `Results` folder contains:

* Model confusion matrices
* F1-score visualization
* High-resolution evaluation images

The CSV files contain classification metrics generated during the Machine Learning experiments.

## 🚀 Future Improvements

* Add real-time speech emotion recognition.
* Improve audio feature extraction.
* Perform hyperparameter optimization.
* Use additional speech emotion datasets.
* Experiment with Transformer-based audio models.
* Deploy the Streamlit application online.
* Add support for multiple audio formats.
* Improve model generalization across different speakers and recording environments.

## 💡 Applications

Speech Emotion Recognition can be used in:

* Human-computer interaction
* Customer service analysis
* Virtual assistants
* Call-center analytics
* Emotion-aware applications
* Healthcare and assistive technologies
* Educational applications

## 👩‍💻 Author

**Swagati Pachare**

GitHub: [Swagati Pachare](https://github.com/swagatipachare)

---

