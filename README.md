# Dog Silencer System

The **Dog Silencer System** is an end-to-end solution that detects dog barks in real time and emits ultrasonic pulses as a deterrent. The system uses modern machine learning techniques with pre-trained audio embeddings (via the CLAP model) and a custom-trained classifier. It also integrates with Firebase to remotely control settings and log events.

> **Important:** Before running the integrated system, you must train the model to generate the necessary files (e.g., `dog_bark_classifier.pkl`). Follow the instructions below to set up your dataset and train the classifier.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up the Virtual Environment](#2-set-up-the-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [Training the Model](#training-the-model)
  - [1. Organize the Dataset](#1-organize-the-dataset)
  - [2. (Optional) Preprocess Audio Files](#2-optional-preprocess-audio-files)
  - [3. Extract Audio Embeddings](#3-extract-audio-embeddings)
  - [4. Train the Classifier](#4-train-the-classifier)
- [Firebase Configuration](#firebase-configuration)
- [Usage](#usage)
- [Additional Scripts](#additional-scripts)
- [Citations](#citations)
- [License](#license)

---

## Overview

The Dog Silencer System continuously listens for dog barks via a microphone and processes the audio using the CLAP model along with a custom-trained classifier. When a bark is detected, the system:
- Logs the event (timestamp and distance) to Firebase.
- Immediately emits ultrasonic pulses (using Raspberry Pi GPIO pins) to deter the dog.

---

## Key Features

- **Real-Time Audio Detection:** Continuously captures and processes audio.
- **Integrated Deterrent Activation:** Immediately emits ultrasonic pulses upon detecting a dog bark.
- **Remote Control & Logging:** Uses Firebase Realtime Database for remote system management and event logging.
- **Customizable Training Pipeline:** Includes scripts to organize data, extract embeddings, and train a classifier.

---

## Hardware & Software Requirements

- **Hardware:**
  - Raspberry Pi (or similar device with GPIO support)
  - Microphone for capturing audio
  - Ultrasonic emitter and sensor (e.g., HC-SR04) connected via GPIO
- **Software:**
  - **Python:** 3.8 or higher (Python 3.11 recommended)
  - **Libraries:** `pyaudio`, `numpy`, `tensorflow`, `RPi.GPIO`, `firebase-admin`, `soundfile`, `msclap`, etc.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/dog_silencer.git
cd dog_silencer/dog_bark_detection
```

### 2. Set Up the Virtual Environment

```bash
python3 -m venv silencer_env
source silencer_env/bin/activate
```

### 3. Install Dependencies

Upgrade `pip` and install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** For PyAudio, make sure you have PortAudio installed on your system.  
> - **macOS:** `brew install portaudio`  
> - **Raspberry Pi (Debian/Ubuntu):**  
>   ```bash
>   sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
>   ```

---

## Training the Model

Before running the integrated system, you need to train the classifier that detects dog barks. This involves organizing your dataset, extracting audio embeddings with the CLAP model, and training the classifier.

### 1. Organize the Dataset

Create directories for dog barks and non-dog sounds:

```bash
mkdir -p dataset/dog
mkdir -p dataset/no_dog
```

Place your audio files in these directories:
- **Dog barks:** `dataset/dog/`
- **Non-dog sounds:** `dataset/no_dog/`

If you need help organizing files from a larger dataset (e.g., ESC-50), you can use the provided script:

```bash
python organize_data.py
```

### 2. (Optional) Preprocess Audio Files

If your audio files need to be resampled or converted to mono, run:

```bash
python preprocess_audio.py
```

This script standardizes your audio files to a consistent sample rate and format.

### 3. Extract Audio Embeddings

Use the CLAP model to extract embeddings from your audio files. This step creates an `embeddings_labels.pkl` file.

```bash
python extract_embeddings.py
```

### 4. Train the Classifier

Train the classifier using the extracted embeddings. This will generate the file `dog_bark_classifier.pkl` which is used by the integrated system.

```bash
python train_classifier.py
```

After running these steps, verify that you have the following key files in your project directory:
- `embeddings_labels.pkl`
- `dog_bark_classifier.pkl`

---

## Firebase Configuration

The system uses Firebase to manage settings and log events. Before running the system:

1. Obtain your Firebase service account key:
   - Visit the [Firebase Console](https://console.firebase.google.com/).
   - Create (or select) your project.
   - Generate a service account key and download the JSON file.
2. Place the JSON file in the `firebase_config/` directory (e.g., `firebase_config/firebaseconfig.json`).

---

## Usage

The integrated real-time system is implemented in **silencerV1.py**. This script:
- Continuously captures audio.
- Uses the CLAP model and the trained classifier (`dog_bark_classifier.pkl`) to detect dog barks.
- Logs detection events to Firebase.
- Activates ultrasonic pulses upon detection.

To run the system, execute:

```bash
python silencerV1.py
```

Press `CTRL+C` to gracefully stop the system.

---

## Additional Scripts

The repository includes several additional scripts for data preparation, training, and testing:

- **Data Organization & Preprocessing:**
  - `organize_data.py`: Organizes the dataset into proper directories.
  - `preprocess_audio.py`: Resamples and converts audio files.
  - `extract_features.py`: Extracts MFCC features (used if training the CNN model).
  
- **Model Training & Evaluation:**
  - `prepare_data.py`: Prepares and pads feature data.
  - `train_model.py`: Trains a CNN for dog bark detection (alternative approach).
  - `plot_history.py`: Plots training history (accuracy and loss).
  - `evaluate_model.py`: Evaluates the trained CNN.
  
- **Testing & Real-Time Detection (Alternative Implementations):**
  - `test_model.py`: Tests the CNN on individual audio files.
  - `real_time_detection.py`: Implements a CNN-based real-time detection.
  - `real_time_audio_detection_filesave.py` & `real_time_detection_algorithm.py`: Additional variants for real-time processing.

Feel free to explore these scripts if you wish to customize or extend the system.

---

## Citations

If you use the CLAP model or the ESC-50 dataset in your project, please cite the following works:

### CLAP

```bibtex
@inproceedings{CLAP2022,
  title={CLAP: Learning Audio Concepts from Natural Language Supervision},
  author={Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

```bibtex
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

### ESC-50 Dataset

```bibtex
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

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*Replace `https://github.com/your_username/dog_silencer.git` with your actual repository URL.*

Enjoy building and customizing your Dog Silencer System! If you have any questions or suggestions, please open an issue or submit a pull request.