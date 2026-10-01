<div align="center">

# 🤟 IndiSignLang — Indian Sign Language (ISL) Recognition Using CNN

**PRAICP-1000 · Artificial Intelligence Capstone Project**

*Classifying static Indian Sign Language hand signs (24 letter classes) with a tuned CNN — with a light-blue **Streamlit** app for **photo upload** and **real-time webcam** recognition.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)
![Keras Tuner](https://img.shields.io/badge/Keras%20Tuner-1.4.8-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Notebook](https://img.shields.io/badge/Jupyter-Colab%20ready-F37626?logo=jupyter&logoColor=white)
![Classes](https://img.shields.io/badge/Classes-24-blue)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)

</div>

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Highlights](#-key-highlights)
3. [Project Details](#-project-details)
4. [Streamlit App (VS Code)](#-streamlit-app-vs-code)
5. [Dataset](#-dataset)
6. [Pipeline Architecture](#-pipeline-architecture)
7. [Image Preprocessing](#-image-preprocessing)
8. [Data Augmentation](#-data-augmentation)
9. [Model Architecture](#-model-architecture)
10. [Hyperparameter Tuning](#-hyperparameter-tuning)
11. [Results](#-results)
12. [Critical Note on the Results](#-critical-note-on-the-results)
13. [Installation & Usage](#-installation--usage)
14. [Predicting a New Image](#-predicting-a-new-image)
15. [Repository Structure](#-repository-structure)
16. [Challenges & How They Were Solved](#-challenges--how-they-were-solved)
17. [Business Insights](#-business-insights)
18. [Limitations](#-limitations)
19. [Future Work](#-future-work)
20. [Project Links](#-project-links)
21. [Author](#-author)

---

## 🎯 Project Overview

Sign language is the primary mode of communication for many people who are deaf or hard of hearing, yet most people do not understand it. This creates a communication barrier in education, healthcare, public services and daily life.

This project builds a **CNN-based image classifier** that looks at a photograph of a hand sign from the Indian Sign Language dataset and predicts which sign it represents. It covers both goals set by the DataMites project brief:

| # | Goal | How it is addressed |
|---|------|---------------------|
| 1 | **Image preprocessing** | Resize every image to 128×128 RGB, normalise pixel values to [0, 1], augment during training |
| 2 | **ML model to predict the sign** | Baseline CNN → Keras Tuner random search → tuned final CNN, evaluated with a full metric suite |

---

## ✨ Key Highlights

- 🧹 **Full data audit** – corrupt-file check, resolution/orientation analysis, class-balance analysis
- ⚖️ **Imbalance handling** – stratified 70/15/15 split + balanced class weights + macro-averaged metrics
- 🔄 **Augmentation inside the model** – rotation, translation and zoom (flip deliberately excluded)
- 🧪 **Baseline vs tuned comparison** – fair, validation-based model selection (test set never used to choose)
- 🔍 **Deep evaluation** – classification report, confusion matrix, per-sign F1, misclassified-image gallery
- 🔮 **End-to-end inference** – `predict_sign()` returns the top-3 signs with confidence
- 💾 **Deployable artefacts** – model saved as `.keras` + class-name JSON, with a reload-and-verify check
- ♻️ **Reproducible** – fixed seeds, configurable run size (`MAX_IMAGES_PER_CLASS`, `TUNER_TRIALS`, `TUNER_EPOCHS`)

---

## 📋 Project Details

| Field | Value |
|-------|-------|
| **Project Title** | PRAICP-1000: IndiSignLang – Indian Sign Language Recognition |
| **Project Code** | PRAICP-1000 |
| **Team Code** | PTID-AIE-JUL-26-11142 |
| **Project Type** | Artificial Intelligence Capstone Project |
| **Dataset Source** | DataMites Capstone Project Dataset |
| **Submitted By** | Sameeksha |

---

## 🖥 Streamlit App (VS Code)

A light-blue web frontend with **two input modes**:

| Mode | What it does |
|------|--------------|
| 📤 **Upload Photo** | Upload a JPG/PNG of a hand sign → predicted letter, confidence and top-3 bars |
| 🎥 **Real-Time Webcam** | Live camera feed with the prediction overlaid on the video; adjustable *min confidence*, *smoothing* and optional *centre-box (ROI)* |

### Run it in VS Code

```bash
# 1. Open this folder in VS Code  (File → Open Folder…)
# 2. Create & activate a virtual environment (Terminal → New Terminal)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate        # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get the trained model into models/  (choose ONE)
#    a) copy IndiSignLang_CNN_Model.keras + IndiSignLang_class_names.json from the notebook run, OR
#    b) train directly:
python train_model.py              # full run
python train_model.py --epochs 3 --max-per-class 30   # quick test

# 5. Launch the app
streamlit run app.py
```

You can also press **F5** in VS Code and pick *"Streamlit: run app.py"* (pre-configured in `.vscode/launch.json`).

### Notes on real-time mode
- Click **START** and allow camera access (works on `localhost` without HTTPS).
- The CNN runs on every 3rd frame and a short **majority vote** smooths flicker.
- The model was trained on plain-background photos: use a plain wall, good light, and enable **centre box (ROI)** if your background is busy.
- The displayed video is mirrored for comfort; the model always receives the un-mirrored frame, matching training.

---

## 🗂 Dataset

**Name:** IndiSignLang — one directory per sign; the directory name is the class label.

**Download:** [PRAICP-1000-IndiSignLang.zip](https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1000-IndiSignLang.zip) *(downloaded and extracted automatically by the notebook)*

### Dataset facts

| Property | Value |
|----------|-------|
| Total images | **4,972** |
| Classes | **24** — letters `A–Y` excluding `J` and `Z` |
| File format / colour mode | JPEG / RGB |
| Corrupt files | None found |
| Smallest class | `H` — 116 images |
| Largest class | `B` — 259 images |
| Imbalance ratio | ≈ **2.2 : 1** (moderate) |

> **Why no J and Z?** These letters are normally signed with a hand *motion*, which cannot be captured in a single still image.

### Original image resolutions

| Resolution | Orientation | Images |
|-----------|-------------|-------:|
| 640 × 480 | Landscape | 3,963 |
| 1920 × 1088 | Landscape | 830 |
| 1088 × 1920 | Portrait | 179 |

### Domain notes

- Each image shows **one hand on a plain, light background** forming one static sign.
- Meaning is carried by **hand shape** (which fingers are extended/curled/crossed, thumb placement, palm orientation) — a natural fit for CNNs.
- Within-class variation exists in hand size, skin tone, camera angle, position in frame and sharpness (some motion blur).
- Some signs (closed-fist / curled-finger shapes) are visually very similar and are expected to be the hardest to separate.

---

## 🏗 Pipeline Architecture

```mermaid
flowchart LR
    A[Download & extract ZIP] --> B[Auto-detect class folders]
    B --> C[Data audit<br/>corrupt check · sizes · balance]
    C --> D[EDA]
    D --> E[Preprocess<br/>RGB · 128×128 · ÷255]
    E --> F[Stratified split<br/>70 / 15 / 15]
    F --> G[Class weights<br/>from train labels]
    G --> H[Baseline CNN]
    G --> I[Keras Tuner<br/>RandomSearch]
    I --> J[Final tuned CNN]
    H --> K[Model comparison<br/>on validation set]
    J --> K
    K --> L[Best model evaluation<br/>on unseen test set]
    L --> M[predict_sign&#40;&#41;]
    L --> N[Save .keras + class names]
```

---

## 🖼 Image Preprocessing

The same `preprocess_image()` function is used for **training and inference**, guaranteeing identical treatment of new images.

| Step | Detail | Why |
|------|--------|-----|
| **Convert to RGB** | Ensures 3 channels | Consistent input depth |
| **Resize to 128 × 128** | Lanczos resampling, `Image.draft()` for fast JPEG decode of large files | Fixed CNN input size; keeps finger detail while keeping training fast |
| **Normalise** | Divide by 255 → range [0, 1] | Faster, more stable gradient-based training |

Resulting tensor shape per image: `(128, 128, 3)`. Full dataset: `X.shape = (4972, 128, 128, 3)`.

> Resizing to a square slightly alters the aspect ratio of non-square photos. This is applied identically to all training and future images, so the CNN learns from the resized versions consistently.

---

## 🔄 Data Augmentation

Implemented with Keras preprocessing layers **at the start of the model**. They are active only during training and automatically disabled for validation, testing and prediction.

| Transformation | Range | Simulates |
|----------------|-------|-----------|
| `RandomRotation` | factor 0.04 (≈ ±14°) | Tilted hand |
| `RandomTranslation` | ±10 % height & width | Hand not centred |
| `RandomZoom` | ±10 % | Hand nearer/farther from camera |

> 🚫 **Horizontal flip is intentionally NOT used** — a mirrored hand turns a left hand into a right hand (or vice-versa) and may not be a valid example of the same sign.

---

## 🧠 Model Architecture

### Baseline CNN (3,307,736 parameters)

```
Input (128×128×3)
 └─ Data Augmentation (rotation · translation · zoom)
 └─ Conv2D(32, 3×3, ReLU)  → MaxPool(2×2)
 └─ Conv2D(64, 3×3, ReLU)  → MaxPool(2×2)
 └─ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
 └─ Flatten
 └─ Dense(128, ReLU) → Dropout(0.5)
 └─ Dense(24, Softmax)
```

### Final tuned CNN (6,522,200 parameters)

```
Input (128×128×3)
 └─ Data Augmentation
 └─ Conv2D(32, 3×3, ReLU)  → MaxPool(2×2)     ← tuned
 └─ Conv2D(64, 3×3, ReLU)  → MaxPool(2×2)     ← tuned
 └─ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
 └─ Flatten
 └─ Dense(256, ReLU) → Dropout(0.3)           ← tuned
 └─ Dense(24, Softmax)
```

### Training configuration

| Setting | Baseline | Final |
|---------|----------|-------|
| Optimiser | Adam (lr = 0.001) | Adam (lr = 0.001) |
| Loss | Sparse categorical cross-entropy | Sparse categorical cross-entropy |
| Max epochs | 30 | 40 |
| Batch size | 32 | 32 |
| Class weights | ✅ Balanced | ✅ Balanced |
| `EarlyStopping` | monitor `val_loss`, patience 5, restore best weights | patience 6, restore best weights |
| `ReduceLROnPlateau` | factor 0.5, patience 2, min lr 1e-6 | same |
| `ModelCheckpoint` | best `val_loss` | best `val_loss` |

---

## 🎛 Hyperparameter Tuning

**Method:** Keras Tuner `RandomSearch` — 8 trials, up to 15 epochs each, objective `val_accuracy`, seed 42.

### Search space (96 possible combinations)

| Hyperparameter | Values |
|----------------|--------|
| Filters — conv layer 1 | 32, 64 |
| Filters — conv layer 2 | 64, 128 |
| Dense units | 128, 256 |
| Dropout | 0.3, 0.4, 0.5 |
| Learning rate | 0.001, 0.0001 |
| Batch size | 16, 32 |

### Trial ranking

| Rank | Batch | Filters 1 | Filters 2 | Dense | Dropout | LR | Val Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | 32 | 32 | 64 | 256 | 0.3 | 0.001 | **0.9960** |
| 2 | 32 | 32 | 128 | 128 | 0.4 | 0.001 | 0.9933 |
| 3 | 16 | 32 | 128 | 256 | 0.5 | 0.001 | 0.9906 |
| 4 | 16 | 32 | 128 | 128 | 0.4 | 0.001 | 0.9893 |
| 5 | 16 | 32 | 64 | 256 | 0.4 | 0.0001 | 0.9879 |
| 6 | 32 | 32 | 64 | 128 | 0.5 | 0.001 | 0.9705 |
| 7 | 16 | 32 | 128 | 128 | 0.4 | 0.0001 | 0.9638 |
| 8 | 32 | 32 | 64 | 128 | 0.5 | 0.0001 | 0.8968 |

⏱ Total search time on CPU (no GPU): **≈ 5 h 28 min**.

---

## 📊 Results

### Data split (stratified)

| Set | Images | Purpose |
|-----|-------:|---------|
| Training (70 %) | 3,480 | Learn weights |
| Validation (15 %) | 746 | Early stopping, tuning, model selection |
| Test (15 %) | 746 | Final unbiased estimate (used once) |

### Model comparison

| Model | Split | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|-------|:-------:|:-------:|:-------:|:-------:|
| Baseline CNN | Validation | 0.9946 | 0.9944 | 0.9932 | 0.9937 |
| Baseline CNN | Test | 0.9933 | 0.9936 | 0.9921 | 0.9926 |
| **Final Tuned CNN** | Validation | **0.9987** | **0.9980** | **0.9978** | **0.9979** |
| **Final Tuned CNN** | Test | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Selected model:** Final Tuned CNN (highest *validation* accuracy).
**Misclassified test images:** **0 / 746**.

Every one of the 24 signs achieved precision = recall = F1 = 1.00 on the test set.

---

## ⚠️ Critical Note on the Results

A perfect 100 % test score is excellent, but it should be read carefully rather than taken at face value:

- **Random image-level split.** Images were split randomly, not by person/session. If the dataset contains many near-identical shots of the same hand (very common in sign-language collections), near-duplicates can land in both train and test, which inflates scores.
- **Small test set.** 746 images means a handful of errors would have moved accuracy by only a few tenths of a percent; 0 errors does not prove the true error rate is 0.
- **Controlled data.** Plain backgrounds and a single hand are easier than real-world webcam conditions (clutter, lighting changes, different signers).

**How to validate more rigorously:** use a *signer-wise* (group) split, test on photos you capture yourself, and try cluttered backgrounds. Expect lower numbers in those settings — that is normal and more informative.

---

## 🚀 Notebook Usage

Open `PRAICP_1000_IndiSignLang.ipynb` in Google Colab (*Runtime → Run all*) or locally with Jupyter. The dataset downloads automatically. For a quick dry-run lower `MAX_IMAGES_PER_CLASS`, `TUNER_TRIALS`, `TUNER_EPOCHS`, `BASELINE_EPOCHS` and `FINAL_EPOCHS`.

> The model always outputs one of the 24 letters, even for images with no valid sign, and confidence is not a guarantee of correctness.

---

## 📁 Repository Structure

```
ISL-Recognition-App/
├── app.py                          # Streamlit frontend (Upload Photo + Real-Time Webcam)
├── train_model.py                  # Train final CNN and save to models/
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                 # Light-blue theme
├── .vscode/
│   ├── launch.json                 # F5 run/debug configs
│   ├── settings.json
│   └── extensions.json
├── utils/
│   ├── predictor.py                # Model loading, preprocessing, prediction
│   └── webcam.py                   # Real-time video processor (streamlit-webrtc)
├── models/
│   ├── IndiSignLang_CNN_Model.keras        # ← put/generate here
│   └── IndiSignLang_class_names.json       # ← put/generate here
└── PRAICP_1000_IndiSignLang.ipynb  # (optional) original notebook
```

---

## 🧗 Challenges & How They Were Solved

| Challenge | Solution |
|-----------|----------|
| **Three different resolutions / orientations** | Unified 128×128 RGB resize; `Image.draft()` for fast decoding of 1920×1088 files |
| **Moderate class imbalance (2.2 : 1)** | Stratified splits, balanced class weights, macro-averaged metrics |
| **Visually similar signs** | Per-sign F1, confusion matrix and misclassified-image gallery rather than accuracy alone |
| **Overfitting on ~3.5k training images** | Augmentation, dropout, early stopping, learning-rate reduction |
| **Information leakage in model selection** | Hyperparameters and best model chosen on validation only; test used once |
| **Memory and compute limits** | 128×128 storage, full-size arrays freed after splitting, configurable trials/epochs/subset size |

---

## 💼 Business Insights

1. **Assistive communication tool** – A phone/web app built on this model could help non-signers understand sign-language users in schools, hospitals, banks and public offices.
2. **Preprocessing enables real-world use** – Standardising inputs lets one model handle photos from different cameras and orientations.
3. **Confusion analysis guides data collection** – Weak or confused signs identify exactly where to collect more varied photos.
4. **Tuning value is measurable** – The tuned model improved validation accuracy (0.9946 → 0.9987) and removed all test errors versus the baseline's 5 (≈ 0.67 % of 746).
5. **Scope matters** – This is a building block for *static letters*; production systems need words, sentences, two-handed signs and motion.

---

## 🚧 Limitations

- Recognises **static single-hand letter signs only** (no J, Z, words or sentences).
- Trained on a **controlled, plain-background** dataset; real-world robustness is unverified.
- **Random split** rather than signer-wise split (see [critical note](#-critical-note-on-the-results)).
- No out-of-distribution / "no sign" rejection class.
- Not validated with deaf or hard-of-hearing users — **should not be used in safety-critical or official communication settings** without such validation.

---

## 🔭 Future Work

- [ ] Signer-wise / group-based splitting and cross-validation
- [ ] **Transfer learning** (MobileNetV2, EfficientNet, ResNet) for stronger generalisation
- [ ] **Real-time webcam** recognition with MediaPipe hand-landmark detection
- [ ] Video / sequence models (CNN-LSTM, 3D-CNN, Transformers) for motion signs such as J and Z
- [ ] Extend to ISL **words and phrases**, and two-handed signs
- [ ] Add a "no sign / unknown" class and confidence thresholding
- [ ] Deploy as a **Streamlit / Gradio** web app or **TensorFlow Lite** mobile app
- [ ] Explainability with **Grad-CAM** to show which hand regions drive predictions
- [ ] Larger Keras Tuner search (more trials, Bayesian optimisation / Hyperband)

---

## 🔗 Project Links

| Resource | Link |
|----------|------|
| 💻 **GitHub Repository** | [Sameekshagithub/PRAICP-1000-Indian-Sign-Language-ISL-Recognition-Using-CNN](https://github.com/Sameekshagithub/PRAICP-1000-Indian-Sign-Language-ISL-Recognition-Using-CNN.git) |
| 📂 **Google Drive (PPT & Report)** | [Open folder](https://drive.google.com/drive/folders/1uiFsyNDpfCWIWidHgfDLZDjr36PixxGG?usp=drive_link) |
| 📦 **Dataset** | [PRAICP-1000-IndiSignLang.zip](https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1000-IndiSignLang.zip) |

---

## 👩‍💻 Author

**Sameeksha**
AI Capstone Project · Team Code `PTID-AIE-JUL-26-11142`

---

<div align="center">

⭐ If you found this project useful, consider giving the repository a star!

*Built to improve communication accessibility with AI.*

</div>
