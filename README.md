# Dog Bark Detection System

This project aims to detect dog barks in audio using machine learning techniques. It provides two approaches:

1. **Using CLAP (Contrastive Language-Audio Pretraining)**: Leveraging pre-trained embeddings from the CLAP model and training a classifier.
2. **Without CLAP**: Building and training a Convolutional Neural Network (CNN) from scratch using extracted MFCC features.

---

## **Table of Contents**

- [Dog Bark Detection System](#dog-bark-detection-system)
  - [**Table of Contents**](#table-of-contents)
  - [**Project Structure**](#project-structure)
  - [**Prerequisites**](#prerequisites)
  - [**Setup Instructions**](#setup-instructions)
    - [**1. Clone the Repository**](#1-clone-the-repository)
    - [**2. Set Up the Virtual Environment**](#2-set-up-the-virtual-environment)
    - [**3. Install Dependencies**](#3-install-dependencies)
  - [**Approach 1: Using CLAP**](#approach-1-using-clap)
    - [**1. Organize the Dataset**](#1-organize-the-dataset)
    - [**2. Extract Embeddings**](#2-extract-embeddings)
    - [**3. Train the Classifier**](#3-train-the-classifier)
    - [**4. Run Real-Time Detection**](#4-run-real-time-detection)
  - [**Approach 2: Without CLAP**](#approach-2-without-clap)
    - [**1. Organize the Dataset**](#1-organize-the-dataset-1)
    - [**2. Preprocess Audio Data**](#2-preprocess-audio-data)
    - [**3. Extract Features**](#3-extract-features)
    - [**4. Prepare Data for Training**](#4-prepare-data-for-training)
    - [**5. Train the CNN Model**](#5-train-the-cnn-model)
    - [**6. Run Real-Time Detection**](#6-run-real-time-detection)
  - [**Citations**](#citations)
    - [**Using CLAP**](#using-clap)
    - [**Without CLAP**](#without-clap)
  - [**License**](#license)

---

## **Project Structure**

```
dog_bark_detection/
├── scripts/
│   ├── organize_data.py
│   ├── preprocess_audio.py
│   ├── extract_features.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── extract_embeddings.py
│   ├── train_classifier.py
│   ├── real_time_detection_clap.py
│   ├── real_time_detection_classifier.py
│   ├── real_time_detection.py
│   ├── evaluate_model.py
│   └── plot_history.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## **Prerequisites**

- **Operating System:** macOS
- **Python:** 3.8 or higher (Python 3.11 recommended)
- **Virtual Environment Manager:** `venv` or any other of your choice
- **Audio Libraries:** PortAudio (for PyAudio)

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/your_username/dog_bark_detection.git
cd dog_bark_detection
```

### **2. Set Up the Virtual Environment**

```bash
python3 -m venv bark_env
source bark_env/bin/activate
```

### **3. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Ensure you have `portaudio` installed on your system for PyAudio:

```bash
brew install portaudio
```

---

## **Approach 1: Using CLAP**

### **1. Organize the Dataset**

- Create directories for dog barks and non-dog sounds:

  ```bash
  mkdir -p dataset/dog
  mkdir -p dataset/no_dog
  ```

- Place your audio files in the corresponding directories:

  - `dataset/dog/` for dog bark audio files
  - `dataset/no_dog/` for non-dog audio files

### **2. Extract Embeddings**

Run the script to extract embeddings using CLAP:

```bash
python scripts/extract_embeddings.py
```

This will generate `embeddings_labels.pkl`.

### **3. Train the Classifier**

Train a classifier using the extracted embeddings:

```bash
python scripts/train_classifier.py
```

This will generate `dog_bark_classifier.pkl`.

### **4. Run Real-Time Detection**

Start real-time dog bark detection using the trained classifier:

```bash
python scripts/real_time_detection_classifier.py
```

---

## **Approach 2: Without CLAP**

### **1. Organize the Dataset**

- Use the same dataset structure as in Approach 1.

### **2. Preprocess Audio Data**

Preprocess audio files to ensure consistent sampling rate and format:

```bash
python scripts/preprocess_audio.py
```

### **3. Extract Features**

Extract MFCC features from the audio files:

```bash
python scripts/extract_features.py
```

This will generate `features_labels.pkl`.

### **4. Prepare Data for Training**

Prepare the data for model training:

```bash
python scripts/prepare_data.py
```

This will generate `prepared_data.pkl`.

### **5. Train the CNN Model**

Train the Convolutional Neural Network (CNN):

```bash
python scripts/train_model.py
```

This will generate `dog_bark_detector.h5`.

### **6. Run Real-Time Detection**

Start real-time dog bark detection using the trained CNN model:

```bash
python scripts/real_time_detection.py
```

---

## **Citations**

### **Using CLAP**

If you use the CLAP model in your project, please cite the following:

**CLAP: Learning Audio Concepts from Natural Language Supervision**

```plaintext
@inproceedings{CLAP2022,
  title={CLAP: Learning Audio Concepts from Natural Language Supervision},
  author={Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

**Natural Language Supervision for General-Purpose Audio Representations**

```plaintext
@misc{CLAP2023,
      title={Natural Language Supervision for General-Purpose Audio Representations}, 
      author={Benjamin Elizalde and Soham Deshmukh and Huaming Wang},
      year={2023},
      eprint={2309.05767},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2309.05767}
}
```

### **Without CLAP**

If you use the ESC-50 dataset in your project, please cite the following:

**ESC: Dataset for Environmental Sound Classification**

```plaintext
@inproceedings{ESC50,
  title={ESC: Dataset for Environmental Sound Classification},
  author={Karol J. Piczak},
  booktitle={Proceedings of the 23rd Annual ACM Conference on Multimedia},
  year={2015},
  location={Brisbane, Australia},
  doi={http://dx.doi.org/10.1145/2733373.2806390}
}
```

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note:** Replace `https://github.com/your_username/dog_bark_detection.git` with the actual URL of your GitHub repository.
