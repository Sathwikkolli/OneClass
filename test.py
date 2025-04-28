import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from utils_combined import AudioSVMClassifier
import os
import torch
import pickle
import joblib

######### inputs ###########
feature_model_name = "wavlm_large"
speaker = "Donald_Trump"
test_audio_file = "./donald_trump.mp3"
saved_model_dir = "/data/Speaker_Specific_Models"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the detector (do NOT retrain)
Trump_detector = AudioSVMClassifier(speaker, feature_model_name, device)

# Load the scaler and svm models
scaling_model = os.path.join(saved_model_dir, speaker, 'scaling_models', f"{feature_model_name}" + '.pkl')
svm_model = os.path.join(saved_model_dir, speaker, 'svm_models', f"{feature_model_name}" + '.pkl')

Trump_detector.scaler = joblib.load(scaling_model)
Trump_detector.svm_model = joblib.load(svm_model)

# Use the model to predict
prediction, score = Trump_detector.predict_from_audio(test_audio_file, load_model_dir=None)  # don't reload here again
print("Prediction for the test audio file:", prediction)
print("Score for the test audio file:", score)