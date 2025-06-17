#  Gender Classification from audio using DSAP (Digital Signal Analysis & Processing)

This project is developed as a final project for the subject **DSAP (Digital Signal Analysis and Processing)** in the 7th semester of the Computer Engineering curriculum at IOE, Tribhuvan University.

The main aim is to implement DSAP techniques practically using **Python** and popular libraries like `scipy`, `librosa`, and `sklearn`, and apply them in a **real-world use case**: identifying speaker gender from **spoken Nepali audio**.

---

## Project Objective

* To gain hands-on experience with DSAP techniques.
* To understand audio signal processing pipelines.
* To extract meaningful features from audio signals.
* To classify gender (male/female) using machine learning models.

---

## Dataset

* A subset of the **Google FLEURS dataset** for the Nepali language.
* Contains a total of **3,058 audio clips** (approximately equal male and female).
* Each sample is labeled with speaker gender in a CSV file: `metadata.csv`.

---

## Project Structure

```
├── eda.py                 # Exploratory data analysis and signal processing
├── feature_creation.py   # Feature extraction (MFCC, ZCR, etc.)
├── model_training.py     # Model training, evaluation, and predictions
├── metadata.csv          # Contains file names and gender labels
├── train_data.csv        # Final feature set for training
├── README.md             # Project documentation
```

---

## Step-by-Step Implementation

### 1. **Audio Preprocessing (eda.py)**

* **Load audio files** using `librosa` at 16kHz sampling rate.
* Apply a **Butterworth bandpass filter** (80 Hz to 3000 Hz) to reduce noise and preserve speech clarity.

  * This filter removes mic hums and high-frequency hiss.
* Visualize frequency response using `matplotlib`.

### 2. **Framing & Windowing**

* Audio is split into **25 ms frames** with **10 ms hop** (overlap).
* Apply **Hamming window** to smooth edges of frames.
* Plot time-domain signals before and after windowing.
* Compute **FFT** of frames to analyze frequency content.
* Generate a **spectrogram** (Frequency vs Time) visualization.

### 3. **Spectral Features**

* Extract **Spectral Centroid**: indicates the "center of mass" of the frequency spectrum.
* Helps differentiate between male and female voices (males typically have lower centroids).

---

## Feature Extraction (feature\_creation.py)

For each audio sample, we compute:

* **F0 Mean & Std** using YIN pitch detection.
* **Spectral Centroid**
* **Zero Crossing Rate (ZCR)**
* **MFCCs (Mel Frequency Cepstral Coefficients)**

### Why MFCCs?

* Capture the shape of the vocal tract.
* Mimic how human ears perceive sound.
* Robust for speech-related tasks.

### How are MFCCs calculated?

1. Pre-emphasis
2. Framing
3. Windowing
4. FFT
5. Mel-filterbank
6. Logarithm
7. DCT (Discrete Cosine Transform)

> Final features are stored in `train_data.csv`

---

## Model Training and Evaluation (model\_training\_and\_testing.py)

### Algorithms Used:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting Classifier
* Multi-layer Perceptron (MLP / Neural Network)

### Evaluation:

* Used **10-fold cross-validation** with F1-Score.
* Results compared using `classification_report` and `ConfusionMatrixDisplay`.

---

## Sample Results

```
Logistic Regression | Mean F1 Score: 0.88
KNN                 | Mean F1 Score: 0.84
Decision Tree       | Mean F1 Score: 0.86
Random Forest       | Mean F1 Score: 0.89
SVM                 | Mean F1 Score: 0.87
Gradient Boosting   | Mean F1 Score: 0.90
MLP Classifier      | Mean F1 Score: 0.91 ✅
```

## 🔍 Inference Demo

To classify a new audio sample:

```python
audio_path = "path_to_new_audio.wav"
features = feature_extraction(audio_path)
scaled = scaler.transform(features.reshape(1, -1))
prediction = mlp.predict(scaled)
```

Output will be `male` or `female`.

---

## Libraries Used

* `librosa` – for audio loading and feature extraction
* `scipy` – for signal filtering and DSP operations
* `sklearn` – for ML model training, scaling, evaluation
* `matplotlib`, `seaborn` – for visualizations


## Future Improvements

* Improve speaker diversity in data
* Use deep learning models (CNN, RNN)
* Deploy as a real-time inference web app or API
* Try multilingual models or domain adaptation

