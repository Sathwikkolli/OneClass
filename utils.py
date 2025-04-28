import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

import librosa

class AudioSVMClassifier:
    def __init__(self, speaker, feature, output_dir):
        """
        Initialize a speaker-specific instance of the classifier for a given feature.
        """
        self.speaker = speaker
        self.feature = feature
        # self.base_directory = base_directory
        self.svm_model = None
        self.scaler = None
        self.output_dir = output_dir
        self.speaker_output_dir = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.original_x_test = None
        self.original_y_test = None
        self.binary_y_test = None
        self.y_pred = None
        


    def single_audio_feature_extraction(self, audio_path, model):
        """
        Extracts features from a single audio file using a given model.
        """
        audio_data, sample_rate = librosa.load(audio_path, sr=None)

        embedding = model.predict(audio_data)

        embedding = embedding.reshape(1, -1)

        return embedding
    
    def Folder_audio_feature_extraction(self, speaker_folder_path, model):
        """
        Extracts features from all audio files in a given speaker's folder.
        """
        embeddings_list = []

        for file_name in os.listdir(speaker_folder_path):
            if file_name.lower().endswith('.wav'):
                audio_path = os.path.join(speaker_folder_path, file_name)
                embedding = single_audio_feature_extraction(audio_path, model)
                embeddings_list.append(embedding)

        if embeddings_list:
            embeddings_array =np.vstack(embeddings_list)

        else:
            embeddins_array = np.array([])

        return embeddings_array

    def load_feature_data(self, file_path):
        """Load and process a pickle file containing audio features."""
        with open(file_path, 'rb') as f:
            feature_data = pickle.load(f)

        if len(feature_data.shape) == 2:
            feature_data = feature_data[0]
        elif len(feature_data.shape) == 3:
            feature_data = feature_data.squeeze(axis=1)

        return feature_data

    def read_pickle_files(self, feature_dir, label):
        """Read pickle files and return feature arrays and labels."""
        features, labels = [], []
        
        if not os.path.exists(feature_dir):
            print(f"Feature {label} doesn’t exist for {feature_dir}")
            return None, None

        for file in os.listdir(feature_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(feature_dir, file)
                feature_data = self.load_feature_data(file_path)
                features.append(feature_data)
                labels.append(label)

        features_array = np.array(features)
        labels_array = np.array(labels)

        if len(features_array.shape) > 2:
            features_array = features_array.reshape(features_array.shape[0], -1)  # Flatten if necessary

        return features_array, labels_array

    def train(self, feature_path, nu, gamma, kernel_input, model=None):
        
        """Train an SVM model using original audio data."""
        x, y = self.read_pickle_files(feature_path, self.feature)
        if x is None or y is None:
            return None, None
        
        self.x_train, self.original_x_test, self.y_train, self.original_y_test = train_test_split(x, y, test_size=0.3)

        self.scaler = StandardScaler().fit(self.x_train)
        
        train_data_scaled = self.scaler.transform(self.x_train)

        if model=='OneClassSVM':
        
            self.svm_model = svm.OneClassSVM(nu=nu, gamma=gamma, kernel=kernel_input)

        if model=='IsolationForest':
            self.svm_model = IsolationForest(random_state=42, n_estimators=100, max_samples='auto', contamination=0.1)

        
        
        self.svm_model.fit(train_data_scaled)

        self.speaker_output_dir = f"{self.output_dir}/{self.speaker}"

        # Save models and test data
        self.save_model()

        # return self.x_test_original, self.y_test_original

    def save_model(self):
        """Save the trained SVM model and scaler."""
        os.makedirs(f"{self.speaker_output_dir}/scaling_models", exist_ok=True)
        os.makedirs(f"{self.speaker_output_dir}/svm_models", exist_ok=True)

        pickle.dump(self.scaler, open(f"{self.speaker_output_dir}/scaling_models/{self.feature}.pkl", 'wb'))
        pickle.dump(self.svm_model, open(f"{self.speaker_output_dir}/svm_models/{self.feature}.pkl", 'wb'))

    def test(self, deepfake_path,include_original=False):
        """Test the trained model on deepfake data."""
        self.x_test, self.y_test = self.read_pickle_files(deepfake_path, self.feature)
        
        if self.x_test is None or self.y_test is None:
            return None

        # Load saved models if not already in memory
        if self.scaler is None or self.svm_model is None:
            print('there are no scaler and svm models')

            with open(f"{self.speaker_output_dir}/scaling_models/{self.feature}.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

            with open(f"{self.speaker_output_dir}/svm_models/{self.feature}.pkl", 'rb') as f:
                self.svm_model = pickle.load(f)
                
        # Scale the deepfake test data
        

    
        scaled_x_test = self.scaler.transform(self.x_test)
        self.binary_y_test = np.full(scaled_x_test.shape[0], -1)  # Label -1 for deepfake


        if include_original:
            self.original_y_test = np.full(self.original_y_test.shape[0], 1)
            scaled_x_test = np.concatenate((scaled_x_test, self.original_x_test), axis=0 )
            self.binary_y_test = np.concatenate((self.binary_y_test, self.original_y_test), axis=0)

            
        self.y_pred = self.svm_model.predict(scaled_x_test)



        return self.binary_y_test, self.y_pred

    def evaluate_model(self, compute_accuracy=False, compute_auc=False, compute_conf_matrix=False):

        results = {}
        """Evaluate model performance using confusion matrix, accuracy, and AUC."""
        y_scores = self.svm_model.decision_function(self.x_test)

        if compute_accuracy:
            accuracy = round(accuracy_score(self.binary_y_test, self.y_pred) * 100, 2)
            results['accuracy'] = accuracy

        if compute_auc:
            fpr, tpr, _ = roc_curve(self.binary_y_test, y_scores, pos_label=1)
            auc_score = round(auc(fpr, tpr), 4)
            results['auc'] = auc_score

        if compute_conf_matrix:
            conf_mat = confusion_matrix(self.binary_y_test, self.y_pred)
            TN, FP, FN, TP = conf_mat.ravel()

            FPR = round((FP / (FP + TN)) * 100, 2)
            FNR = round((FN / (FN + TP)) * 100, 2)

            results["confusion_matrix"] = conf_mat
            results["false_positive_rate"] =  FPR
            results["false_negative_rate"] = FNR

        return results

    def process_single_file(self, file_path):
        """Process and extract features from a single file."""
        return self.load_feature_data(file_path)
