'''
Credit goes to https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/blob/master/lstm_genre_classifier_keras.py
Modifications by Ethan B for 477 course project

'''

import librosa
import logging
import sys
import numpy as np
from keras.models import model_from_json
from keras import backend as K

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction and genre list

# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_model(model_path, weights_path):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return trained_model


def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features


def get_genre(model, music_path):
    "Predict genre of music using a trained model"
    prediction = model.predict(extract_audio_features(music_path))
    predict_genre = GenreFeatureData().genre_list[np.argmax(prediction)]
    return predict_genre


if __name__ == "__main__":
    print("loading model...\n")
    MODEL = load_model("model\lstm_genre_classifier_model.json", "model\lstm_genre_classifier_model.h5")
    # size = get_model_memory_usage(30, MODEL)
    # size = size * 1000000
    print("\nmodel loaded")
    # print('trained model size: ' + str(size) + "bytes")

    
    while(1):
        print("Enter path (or 'exit' to stop):\n>", end="")
        # PATH = sys.argv[1] if len(sys.argv) == 2 else "LSTM\Test_Samples\Khemmis-Hunted-04BeyondTheDoor.mp3" # Actual song i downloaded to test lol  @ethanB
        PATH = input()
        if PATH.lower() == "exit": break
        try:
            GENRE = get_genre(MODEL, PATH)
        except:
            print("could not generate prediction. check fp")
            continue
        print("Model predict: {}".format(GENRE))