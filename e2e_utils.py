import os
import pickle
import numpy as np
import torch
import torchaudio
from s3prl.nn import S3PRLUpstream, Featurizer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

class AudioSVMClassifier:
    def __init__(self, model_name, device, speaker=None):
        """
        Initialize the detector with a S3PRL model on the specified device.
        
        Args:
            model_name (str): Name of the S3PRL model to load.
            device (torch.device): Device (CPU or GPU) on which to run the model.
            speaker (str, optional): Identifier for the speaker (if applicable).
        """
        self.model_name = model_name
        self.device = device
        self.speaker = speaker
        print(f"Loading model: {model_name}")
        self.model = S3PRLUpstream(model_name).to(device).eval()
        self.featurizer = Featurizer(self.model).to(device)
        print(f"Model {model_name} loaded on {device}")
        
        # SVM classifier attributes
        self.svm_model = None
        self.scaler = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.binary_y_test = None
        self.y_pred = None

    def extract_features(self, file_path, aggregate_emb=False, layer_number=0):
        """
        Extract features from an audio file using the loaded S3PRL model.
        
        Args:
            file_path (str): Path to the audio file.
            aggregate_emb (bool): 
                - If True, returns a single aggregated embedding across layers.
                - If False, returns the embedding(s) from the model's hidden layers.
            layer_number (int, optional): When aggregate_emb is False, if a specific 
                layer is desired, specify its index (0-indexed). If not provided, all 
                layer embeddings are returned.
        
        Returns:
            numpy.ndarray: 
                - Aggregated embedding if aggregate_emb=True.
                - Otherwise, either the specified layer’s embedding or a stack of all 
                  layer embeddings.
        """
        # Load the audio file and obtain metadata.
        waveform, sample_rate = torchaudio.load(file_path)
        metadata = torchaudio.info(file_path)
        
        # Ensure waveform has shape [batch, samples]
        if waveform.ndimension() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute maximum length and pad/truncate accordingly.
        max_length = int(sample_rate * (metadata.num_frames / metadata.sample_rate))
        wavs = torch.zeros(waveform.size(0), max_length)
        for i, wav in enumerate(waveform):
            wavs[i, :min(max_length, wav.size(0))] = wav[:max_length]
        wavs_len = torch.LongTensor([min(max_length, waveform.size(1)) for _ in range(waveform.size(0))])
        
        # Forward pass through the model.
        with torch.no_grad():
            all_hs, all_hs_len = self.model(wavs.to(self.device), wavs_len.to(self.device))
        
        if aggregate_emb:
            # Compute aggregated embedding: average the mean-pooled embeddings of all layers.
            embedding = self.aggregate_embeddings(all_hs)
            return embedding.cpu().numpy()
        else:
            # Get embeddings for each layer: mean pooling over the time dimension.
            layer_embeddings = [layer.mean(dim=1).cpu().numpy() for layer in all_hs]
            if layer_number is not None:
                if layer_number < 0 or layer_number >= len(layer_embeddings):
                    raise ValueError(f"Invalid layer_number {layer_number}. Must be between 0 and {len(layer_embeddings)-1}.")
                return layer_embeddings[layer_number]
            else:
                return np.stack(layer_embeddings, axis=0)

    def aggregate_embeddings(self, all_hs):
        """
        Aggregates embeddings from all hidden layers by computing the mean over time for 
        each layer and then averaging across layers.
        
        Args:
            all_hs (list of torch.Tensor): List of hidden state tensors.
        
        Returns:
            torch.Tensor: Aggregated embedding of shape [1, n_features].
        """
        embeddings_list = [layer.mean(dim=1) for layer in all_hs]
        final_embedding = sum(embeddings_list) / len(embeddings_list)
        return final_embedding

    def train_svm(self, X, nu, gamma):
        """
        Train a one-class SVM on the provided feature matrix.
        
        Args:
            X (numpy.ndarray): Feature array for training (shape: n_samples x n_features).
            nu (float): Nu parameter for OneClassSVM.
            gamma (float): Gamma parameter for OneClassSVM.
        """
        # Label all training samples as 1 (target/real class)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, np.ones(X.shape[0]), test_size=0.3, random_state=42
        )
        self.scaler = StandardScaler().fit(self.x_train)
        train_data_scaled = self.scaler.transform(self.x_train)
        self.svm_model = svm.OneClassSVM(nu=nu, gamma=gamma, kernel='rbf')
        self.svm_model.fit(train_data_scaled)
        print("SVM training completed.")

    def train_svm_from_folder(self, folder_path, nu, gamma, aggregate_emb=False, layer_number=0):
        """
        Extract features from all audio files in a given folder and train the SVM.
        
        Args:
            folder_path (str): Path to the folder containing audio files.
            nu (float): Nu parameter for OneClassSVM.
            gamma (float): Gamma parameter for OneClassSVM.
            aggregate_emb (bool): Whether to aggregate embeddings across layers.
            layer_number (int, optional): If not aggregating, specify the layer index to use.
        """
        valid_extensions = ('.wav', '.flac', '.mp3')
        feature_list = []
        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(folder_path, filename)
                try:
                    emb = self.extract_features(file_path)
                    # Ensure emb is a 2D array.
                    if emb.ndim == 1:
                        emb = emb.reshape(1, -1)
                    feature_list.append(emb)
                    print(f"Extracted features from {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        if not feature_list:
            raise ValueError("No valid audio files found in the folder.")
        # Stack all embeddings into a single feature matrix
        X = np.vstack(feature_list)
        print(f"Extracted features for {X.shape[0]} audio files. Training SVM...")
        self.train_svm(X, nu, gamma)

    def test_svm(self, X):
        """
        Test the trained SVM on provided features.
        
        Args:
            X (numpy.ndarray): Feature array for testing (shape: n_samples x n_features).
            
        Returns:
            tuple: (binary_y_test, y_pred) where binary_y_test are the true labels 
                   (set to -1 to denote anomalies) and y_pred are the SVM predictions.
        """
        self.x_test = X
        scaled_x_test = self.scaler.transform(self.x_test)
        # For testing, we label all samples as -1 (anomaly), as typically expected in one-class SVM tests.
        self.binary_y_test = np.full(scaled_x_test.shape[0], -1)
        self.y_pred = self.svm_model.predict(scaled_x_test)
        return self.binary_y_test, self.y_pred

    def evaluate_model(self, compute_accuracy=False, compute_auc=False, compute_conf_matrix=False):
        """
        Evaluate the SVM model using accuracy, AUC, and/or confusion matrix.
        
        Args:
            compute_accuracy (bool): Whether to compute accuracy.
            compute_auc (bool): Whether to compute the AUC.
            compute_conf_matrix (bool): Whether to compute the confusion matrix.
            
        Returns:
            dict: A dictionary containing the computed metrics.
        """
        results = {}
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
            if conf_mat.size == 4:
                TN, FP, FN, TP = conf_mat.ravel()
                results["confusion_matrix"] = conf_mat
                results["false_positive_rate"] = round((FP / (FP + TN)) * 100, 2)
                results["false_negative_rate"] = round((FN / (FN + TP)) * 100, 2)
            else:
                results["confusion_matrix"] = conf_mat
        return results

    def predict_from_audio(self, file_path, aggregate_emb=True, layer_number=None, load_model_dir=None):
        """
        Takes an audio file, extracts its embedding, and sends it through the saved SVM model.
        
        If the SVM model or scaler are not currently loaded in memory, they will be loaded from
        the provided load_model_dir. The directory should contain 'scaling_model.pkl' and 'svm_model.pkl'.
        
        Args:
            file_path (str): Path to the audio file.
            aggregate_emb (bool): Whether to aggregate embeddings across layers.
            layer_number (int, optional): If not aggregating, specify which layer's embedding to use.
            load_model_dir (str, optional): Directory from which to load the saved models if needed.
            
        Returns:
            numpy.ndarray: The prediction result from the SVM model.
        """
        # Load saved SVM model and scaler if not already loaded.
        if self.svm_model is None or self.scaler is None:
            if load_model_dir is None:
                raise ValueError("SVM model and scaler not in memory and load_model_dir not provided.")
            scaler_path = os.path.join(load_model_dir, "scaling_model.pkl")
            svm_path = os.path.join(load_model_dir, "svm_model.pkl")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(svm_path, "rb") as f:
                self.svm_model = pickle.load(f)
            print("Loaded saved SVM model and scaler from disk.")
        
        # Extract embedding from the provided audio file.
        emb = self.extract_features(file_path, aggregate_emb=aggregate_emb, layer_number=layer_number)
        
        # Ensure embedding is two-dimensional.
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim == 2 and emb.shape[0] != 1:
            emb = emb[0].reshape(1, -1)
        
        # Scale the embedding and predict using the SVM model.
        scaled_emb = self.scaler.transform(emb)
        prediction = self.svm_model.predict(scaled_emb)
        return prediction

# --------------------------
# Example usage:
# --------------------------
if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the detector with a chosen S3PRL model
    model_name = "wavlm_large"
    detector = AudioSVMClassifier(model_name, device, speaker="Donald_Trump")
    
    # Suppose we have a folder of audio files for training.
    training_folder = "/data/FF_V2/Famous_Figures/Donald_Trump/Original"  # Update this path to your folder
    detector.train_svm_from_folder(training_folder, nu=0.1, gamma=0.1,)
    
    # Optionally, save the trained scaler and SVM model for later use.
    saved_model_dir = "saved_model"
    os.makedirs(saved_model_dir, exist_ok=True)
    with open(os.path.join(saved_model_dir, "scaling_model.pkl"), "wb") as f:
        pickle.dump(detector.scaler, f)
    with open(os.path.join(saved_model_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(detector.svm_model, f)
    
    # Example: Use the predict_from_audio method to predict on a new audio file.
    test_audio_file = "/data/FF_V2/Famous_Figures/Donald_Trump/spoof/Donald_Trump_00001_FISHSPEECH_616_0.5.wav"  # Update this path to your test file
    prediction = detector.predict_from_audio(test_audio_file, load_model_dir=saved_model_dir)
    print("Prediction for the test audio file provided:", prediction)
