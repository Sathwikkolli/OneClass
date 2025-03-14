import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
# import librosa

class AudioSVMClassifier:
    def __init__(self, speaker, base_directory, feature, gamma=0.001, nu=0.5):
        """
        Initialize a speaker-specific instance of the classifier for a given feature.
        """
        self.speaker = speaker
        self.feature = feature
        self.gamma = gamma
        self.nu = nu
        self.base_directory = base_directory

        self.svm_model = None
        self.scaler = None
        self.x_test_original = None
        self.y_test_original = None


    def single_audio_feature_extraction(self, audio_path, model):
        audio_data, sample_rate = librosa.load(audio_path, sr=None)

        embedding = model.predict(audio_data)

        embedding = embedding.reshape(1, -1)

        return embedding
    
    def speakerFolder_audio_feature_extraction(self, speaker_folder_path, model):
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

    def train(self, feature_path):
        """Train an SVM model using original audio data."""
        x, y = self.read_pickle_files(feature_path, self.feature)
        if x is None or y is None:
            return None, None
        
        x_train, self.x_test_original, y_train, self.y_test_original = train_test_split(x, y, test_size=0.3)

        self.scaler = StandardScaler().fit(x_train)
        # scaler = StandardScaler().fit(x_train)
        
        train_data_scaled = self.scaler.transform(x_train)
        # train_data_scaled = scaler.transform(x_train)

        print(f'the shape of the train data is {x_train.shape}')

        self.svm_model = svm.OneClassSVM(nu=self.nu, gamma=self.gamma, kernel='rbf')
        # svm_model = svm.OneClassSVM(nu=self.nu, gamma=self.gamma, kernel='rbf')
        
        self.svm_model.fit(train_data_scaled)
        # svm_model.fit(train_data_scaled)

        # Save models and test data
        # self.save_model(x_test, y_test)
        self.save_model(self.x_test_original, self.y_test_original)

        return self.x_test_original, self.y_test_original

    def save_model(self, x_test, y_test):
        """Save the trained SVM model and scaler."""
        model_dir = f'new_svm_trained_models/{self.speaker}'
        os.makedirs(f"{model_dir}/scaling_models", exist_ok=True)
        os.makedirs(f"{model_dir}/svm_models", exist_ok=True)
        os.makedirs(f"{model_dir}/x_test", exist_ok=True)
        os.makedirs(f"{model_dir}/y_test", exist_ok=True)

        pickle.dump(self.scaler, open(f"{model_dir}/scaling_models/{self.feature}.pkl", 'wb'))
        pickle.dump(self.svm_model, open(f"{model_dir}/svm_models/{self.feature}.pkl", 'wb'))

        pickle.dump(x_test, open(f"{model_dir}/x_test/{self.feature}_x_test.pkl", 'wb'))
        pickle.dump(y_test, open(f"{model_dir}/y_test/{self.feature}_y_test.pkl", 'wb'))

    def test(self, deepfake_path):
        """Test the trained model on deepfake data."""
        deepfake_x_test, deepfake_y_test = self.read_pickle_files(deepfake_path, self.feature)
        
        if deepfake_x_test is None or deepfake_y_test is None:
            return None

        # Load saved models if not already in memory
        if self.scaler is None or self.svm_model is None:
            print('there are no scaler and svm models')
            model_dir = f'new_svm_trained_models/{self.speaker}'
            with open(f"{model_dir}/scaling_models/{self.feature}.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
                # print(f' the scaler model is{self.scaler}')
            with open(f"{model_dir}/svm_models/{self.feature}.pkl", 'rb') as f:
                self.svm_model = pickle.load(f)
                
        # Scale the deepfake test data
        scaled_deepfake_test = self.scaler.transform(deepfake_x_test)
        deepfake_y_true = np.full(scaled_deepfake_test.shape[0], -1)  # Label -1 for deepfake
        original_y_true = np.full(self.x_test_original.shape[0], 1)
        x_test_combined = np.vstack((self.x_test_original, scaled_deepfake_test))
        y_test_combined = np.concatenate((original_y_true, deepfake_y_true ), axis=0)

        return self.evaluate_model(x_test_combined, y_test_combined)


    def evaluate_model(self, test_data, test_labels):
        """Evaluate model performance using confusion matrix, accuracy, and AUC."""
        y_scores = self.svm_model.decision_function(test_data)
        y_pred = self.svm_model.predict(test_data)

        conf_mat = confusion_matrix(test_labels, y_pred)
        TN, FP, FN, TP = conf_mat.ravel()

        FPR = round((FP / (FP + TN)) * 100, 2)
        FNR = round((FN / (FN + TP)) * 100, 2)
        accuracy = round(accuracy_score(test_labels, y_pred) * 100, 2)

        fpr, tpr, _ = roc_curve(test_labels, y_scores, pos_label=1)
        auc_score = round(auc(fpr, tpr), 4)

        return {
            "True Negative": TN, "False Positive": FP, 
            "False Negative": FN, "True Positive": TP, 
            "FPR": FPR, "FNR": FNR, "Accuracy": accuracy, "AUC": auc_score
        }

    def process_single_file(self, file_path):
        """Process and extract features from a single file."""
        return self.load_feature_data(file_path)

    # def process_multiple_directories(self, directories):
    #     """Process multiple directories by calling process_single_file."""
    #     results = {}
    #     for directory in directories:
    #         for file in os.listdir(directory):
    #             if file.endswith(".pkl"):
    #                 file_path = os.path.join(directory, file)
    #                 results[file] = self.process_single_file(file_path)
    #     return results
