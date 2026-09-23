import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tempfile
import os

# Streamlit app setup
st.title('Emotion Prediction from Audio')

# File uploader for the model file
model_file = st.file_uploader("Upload the trained model file (.h5)", type=['h5'])
if model_file is not None:
    # Save the model file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_model_file:
        tmp_model_file.write(model_file.getbuffer())
        tmp_model_path = tmp_model_file.name  # Get the file path

    model = load_model(tmp_model_path)  # Load the uploaded model
    st.success("Model loaded successfully!")

    # Define emotion labels (update with your labels)
    emotion_labels = ['anger', 'fear', 'sad', 'happy', 'disgust', 'neutral']
    num_classes = len(emotion_labels)

    # Label Encoder (same as used during training)
    label_encoder = LabelEncoder()
    label_encoder.fit(emotion_labels)

    # Function to extract features from audio
    def extract_features(file_path, max_pad_len=40):  # Changed max_pad_len to 40
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features = np.vstack((mfcc, chroma, mel))

        # Ensure the features are the same length
        if features.shape[1] < max_pad_len:
            pad_width = max_pad_len - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_pad_len]

        return features.T  # Transpose to (time_steps, features)

    # File uploader for audio files
    audio_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3'])
    
    if audio_file is not None:
        # Save the uploaded audio file temporarily
        with open("uploaded_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())
        
        st.audio(audio_file, format='audio/wav')  # Show the uploaded audio

        # Extract features
        features = extract_features("uploaded_audio.wav")
        
        # Print the shape of the extracted features for debugging
        st.write(f"Shape of extracted features: {features.shape}")
        
        try:
            # Reshape features for model input
            features = features.reshape(1, features.shape[0], features.shape[1])
            st.write(f"Shape of features after reshaping: {features.shape}")
        except Exception as e:
            st.write(f"Error reshaping features: {e}")

        # Predict emotion using the model
        try:
            prediction = model.predict(features)
            st.write(f"Prediction: {prediction}")  # Show the prediction for debugging

            # Get the class with the highest probability
            predicted_class = np.argmax(prediction, axis=1)
            
            # Check if predicted_class is within the expected range of labels
            if predicted_class >= 0 and predicted_class < num_classes:
                predicted_label = label_encoder.inverse_transform(predicted_class)
                st.write(f"Predicted Emotion: {predicted_label[0]}")
            else:
                st.write("Predicted class is out of range!")
        except Exception as e:
            st.write(f"Error during prediction: {e}")
