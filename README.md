# 🎵 TIF360ProjectD: Clustering and Music Generation with CVAEs

This project was developed as part of the **TIF360** course and explores the use of **Conditional Variational Autoencoders (CVAEs)** to cluster and generate music from latent representations. The aim is to better understand how structured latent spaces can be used for creative audio synthesis.

---

## 🚀 Overview

We implement a CVAE architecture to learn latent embeddings of musical audio and explore how different regions of the latent space can be used to generate distinct audio samples. This includes:

- Preprocessing musical clips to spectrograms
- Training a CVAE to map these features to a structured latent space.
- Generating new music samples by decoding from various latent points.

---

## 🎧 Audio Examples

Here are some audio samples generated from different regions of the latent space:
#### Rock
(https://github.com/user-attachments/assets/8287aa19-7307-4e9d-b9c6-1cd9f8a79c65)
#### Pop

#### Classical

#### Jazz

---

## 📂 Project Structure
TIF360ProjectD/
├── models/                    # Model definitions and training scripts
│   └── cvae.py                # CVAE model in Pytorch Lightning
│
├── utils/                     # Utility data to handle data conversion
│   ├── labelconversion.csv    # Conversion mapping from file name to genre
│   └── labelencoding.csv      # Conversion mapping from label to one-hot encoding
│
├── audio_samples/             # Generated audio clips for listening
│
├── notebooks/                 # Jupyter notebook for running the model
│   └── CVAE_Music_Generation.ipynb
│
├── outputs/                   # Model checkpoints and path to final model
│   ├── checkpoints/
│   └── model_path/
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── LICENSE                    # Open source license

---

## 🧠 Key Technologies

- Python
- PyTorch
- Librosa
- Scikit-learn
- Matplotlib 

---

## How to run

You can run this directly at Kaggle:

[Insert link]

## Datasets

Can be found on Kaggle:

Spectrograms -> [Insert link]

mp3 -> [Insert link]


