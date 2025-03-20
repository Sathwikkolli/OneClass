# Speaker-Specific-ASD

This repositry contains code for protecting famous figures from audio deepfake attacks.

environment install
```ruby
conda env create -f environment.yml
```
## Installation

Clone this repository to your workspace using the following command:

git clone https://github.com/issflab/speaker-specific-ASD.git
conda env create -f environment.yml
conda activate oneclasssvm

### Tutorial

- **`base_directory`**: Path to the root data directory. Example: `project_root/data/feature_type`
- **`feat_directory`**: Path to the speech-related feature representations. Example: `speech_feature_representation`
- **`data_names`**: List of feature representation types. Example: `["ASV19", "CODEC2"]`
- **`data_types`**: Type of dataset (train or test). Example: `["train", "test"]`
- **`features`**: List of feature names used for training or testing.
- **`speakers`**: List of speakers. Example: `["Donald_Trump", "Joe_Biden"]`
- **`deepfake`**: List of deepfake versions available for analysis.



## Folder Structure

project_root/
│── data/
│   ├── features_type/             
│   │   ├── speech_feature_representation/ 
│   │   │   ├── feature_representation_1/
│   │   │   │   ├── train/            
│   │   │   │   │   ├── speaker1/              
│   │   │   │   │   │   ├── deepfake_1/       
│   │   │   │   │   │   │   ├── feature_1/     
│   │   │   │   │   │   ├── original/         
│   │   │   │   │   │   │   ├── feature_1/
│   │   ├── ...        
│   │   │   │   │   ├── speaker2/             

