# Speaker-Specific-ASD

This repository contains code for protecting famous figures from audio deepfake attacks.

## Installation

Clone this repository to your workspace using the following command:
```ruby
git clone https://github.com/issflab/speaker-specific-ASD.git
```

Create the conda environment from the provided YAML file:
```ruby
conda env create -f environment.yaml
```

Activate the environment:
```ruby
conda activate speaker-specific-asd
```

### Testing Example:
To test an audio file whether it is real or fake, run test.py. 

First set the following parameters.
```ruby
feature_model_name = "wavlm_large"
speaker = "Donald_Trump"
test_audio_file = "./donald_trump.mp3"
saved_model_dir = "/data/Speaker_Specific_Models"
```

Then run,
```
python3 test.py
```

### Training Example:
You can train the SVM using extracted features from your audio dataset. The following example uses the train_svm_from_folder method:

```ruby
from sklearn import svm

# Initialize the detector with S3PRL model
detector = AudioSVMClassifier(model_name, device, speaker="Donald_Trump")

# Set up training data path
training_folder = "/data/FF_V2/Famous_Figures/Donald_Trump/Original"

# Train the SVM classifier
detector.train_svm_from_folder(training_folder, nu=0.1, gamma=0.1, aggregate_emb=False, layer_number=0)
print("SVM Training Complete.")
```


### Feature Extraction Example:
The code uses the S3PRLUpstream model to extract speech features from audio files. In this example, wavlm_large is used for feature extraction.


```ruby
from s3prl.nn import S3PRLUpstream

# Initialize the S3PRL model
model_name = "wavlm_large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
upstream_model = S3PRLUpstream(model_name).to(device)

# Load an example audio file
audio_path = "path_to_your_audio.wav"
waveform, sample_rate = torchaudio.load(audio_path)
```



### Testing Example:
Once the model is trained, you can predict the label for a given audio sample using the predict_from_audio method.

```ruby
# predict on a test audio fie
test_audio_file = "/data/FF_V2/Famous_Figures/Donald_Trump/spoof/Donald_Trump_00001_FISHSPEECH_616_0.5.wav"
prediction = detector.predict_from_audio(test_audio_file, aggregate_emb=True, load_model_dir=saved_model_dir)
print("Prediction for the test audio file:", prediction)
```







