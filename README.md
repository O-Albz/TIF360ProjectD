# TIF360ProjectD: Clustering and Music Generation with CVAEs

## Overview

This project explores AI-driven music generation using a **β-Conditional Variational Autoencoder (β-CVAE)** trained on spectrograms. The model learns a structured latent space conditioned on music genre, enabling controlled generation of genre-specific audio clips.

## Objective

- Learn meaningful latent representations of music.
- Enable **genre-conditioned generation** of spectrograms.
- Evaluate β-CVAE's effectiveness for clustering and synthesis compared to other autoencoder variants.

---

## Audio Example

Here is an example of what the model produces, which is the same for all points in the latent space, for all genres. Posterior collapse...


https://github.com/user-attachments/assets/b60f8e11-0635-4334-99db-e0878f605a90


---

## Project Structure

```
TIF360ProjectD/
├── models/                     # Model definitions and training scripts
│   └── cvae.py                # CVAE model in PyTorch Lightning
├── utils/                     # Utility data to handle data conversion
│   ├── labelconversion.csv    # Conversion mapping from file name to genre
│   └── labelencoding.csv      # Conversion mapping from label to one-hot encoding
├── audio_samples/             # Generated audio clips for listening
├── notebooks/                 # Jupyter notebook for running the model
│   └── CVAE_Music_Generation.ipynb
├── outputs/                   # Model checkpoints and path to final model
│   ├──checkpoints/
│   └── model_path/
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── LICENSE                    # Open source license
```

---

## Key Technologies

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

Spectrograms -> [Kaggle](https://www.kaggle.com/datasets/oskaralbers/fma-stft-spectrograms)

mp3 -> [Kaggle](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium)

## Conclusions

This model is not good enough to synthesize music, would recommend adding a discriminator to the model. Feel free to use this as a start and build upon it.


