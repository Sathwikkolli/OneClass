import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from utils import AudioSVMClassifier
import os


base_directory = "/data/Deep_Fake_Data/"
feat_directory = "Features_no_padding"
feature = ['wavlm_large']
output_dir = 'one_class_svm'
speaker_name = 'Donald_Trump'
# output_dir = os.path.join(base_directory, 'oc_svm_models')

output_dir = 'one_class_svm'

print(output_dir)


Trump_model =  AudioSVMClassifier(speaker_name, base_directory, feature, output_dir)

test_featrure = '/data/Deep_Fake_Data/Features_no_padding/no_laundering/Donald_Trump/train/bona-fide'
y_orig, y_pred = Trump_model.test(test_featrure)

