import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import torch
from e2e_utils import AudioSVMClassifier
import os


# base_directory = "/data/Deep_Fake_Data/"
# feat_directory = "Features_no_padding"
# feature = ['wavlm_large']
# output_dir = 'one_class_svm'
# speaker_name = 'p229'
# # output_dir = os.path.join(base_directory, 'oc_svm_models')

# output_dir = 'one_class_svm'

# print(output_dir)

# # Trump_model =  AudioSVMClassifier(speaker_name, base_directory, feature, output_dir)
# p229_model =  AudioSVMClassifier(speaker_name, base_directory, feature, output_dir)

# test_featrure = '/data/Deep_Fake_Data/Features_no_padding/ASV19/p229/train/Original/wavlm_large'
# y_orig, y_pred = p229_model.test(test_featrure)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the detector with a chosen S3PRL model
model_name = "wavlm_large"
speaker_name = "Donald_Trump"
Trump_detector = AudioSVMClassifier(model_name, device, speaker=speaker_name)

# Example: Use the predict_from_audio method to predict on a new audio file.
saved_model_dir = '/data/Speaker_Specific_Models/' + speaker_name
test_audio_file = "/data/FF_V2/Famous_Figures/Donald_Trump/spoof/Donald_Trump_00001_FISHSPEECH_616_0.5.wav"  # Update this path to your test file
prediction = Trump_detector.predict_from_audio(test_audio_file, load_model_dir=saved_model_dir)
print("Prediction for the test audio file provided:", prediction)





