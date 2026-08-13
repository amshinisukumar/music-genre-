import os
import numpy as np
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "dataset/genres"
MODEL_PATH = "models/genre_model.pkl"


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(file_path):

    try:

        y, sr = librosa.load(file_path, sr=22050, duration=30)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        )

        mfcc_features = np.mean(mfcc, axis=1)

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )

        centroid_mean = np.mean(spectral_centroid)

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )

        bandwidth_mean = np.mean(spectral_bandwidth)

        # Zero Crossing Rate
        zero_crossing = librosa.feature.zero_crossing_rate(y)

        zcr_mean = np.mean(zero_crossing)

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )

        rolloff_mean = np.mean(rolloff)

        # Tempo
        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        tempo_value = float(np.asarray(tempo).flatten()[0])

        # RMS Energy
        rms = librosa.feature.rms(y=y)

        rms_mean = np.mean(rms)

        features = np.concatenate([
            mfcc_features,
            [centroid_mean],
            [bandwidth_mean],
            [zcr_mean],
            [rolloff_mean],
            [tempo_value],
            [rms_mean]
        ])

        return features

    except Exception as e:

        print("Error:", file_path)
        print(e)

        return None


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

X = []
y = []

genres = sorted(os.listdir(DATASET_PATH))

print("Genres:", genres)

for genre in genres:

    genre_path = os.path.join(
        DATASET_PATH,
        genre
    )

    if not os.path.isdir(genre_path):
        continue

    print("\nProcessing:", genre)

    for file in os.listdir(genre_path):

        if file.endswith(".wav"):

            file_path = os.path.join(
                genre_path,
                file
            )

            features = extract_features(file_path)

            if features is not None:

                X.append(features)
                y.append(genre)


X = np.array(X)
y = np.array(y)

print("\nFeature shape:", X.shape)
print("Number of samples:", len(X))


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("MODEL ACCURACY:", accuracy)
print("================================")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)
