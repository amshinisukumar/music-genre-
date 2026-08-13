# 🎵 AI Music Intelligence System

An AI-powered music analysis application that performs music genre classification and audio intelligence using machine learning and audio signal processing.

## 🚀 Features

- 🎵 Music genre classification
- 🎧 Audio quality analysis
- 🧹 Basic noise reduction
- 🎼 MFCC feature extraction
- 📊 Genre probability analysis
- 🔀 Cross-genre analysis
- 🧬 Audio DNA analysis
- 🔍 Explainable AI
- 🎶 Smart playlist categorization

## 🤖 Machine Learning

The system uses a Random Forest Classifier trained on audio features extracted from the GTZAN music genre dataset.

### Features used

- 20 MFCC features
- Spectral Centroid
- Spectral Bandwidth
- Zero Crossing Rate
- Spectral Rolloff
- Tempo
- RMS Energy

## 🎼 Supported Genres

The model can classify the genres represented in the GTZAN training dataset:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 🛠️ Technologies

- Python
- Streamlit
- NumPy
- Librosa
- Scikit-learn
- SciPy
- Matplotlib
- Joblib

## 📁 Project Structure

```text
music-genre-/
│
├── app.py
├── train_model.py
├── requirements.txt
├── packages.txt
├── README.md
│
└── models/
    └── genre_model.pkl
