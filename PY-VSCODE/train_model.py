"""Train the final ISL CNN (best hyper-parameters from the notebook's Keras Tuner run)
and save it to models/ so the Streamlit app can use it.

Usage:
    python train_model.py                      # full training
    python train_model.py --epochs 3 --max-per-class 30   # quick test
"""
import argparse
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                                     RandomRotation, RandomTranslation, RandomZoom)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

SEED = 42
IMG_SIZE = 128
IMG_EXT = (".jpg", ".jpeg", ".png")
URL = "https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1000-IndiSignLang.zip"
ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "PRAICP-1000-IndiSignLang.zip"
EXTRACT_DIR = ROOT / "IndiSignLang_data"
MODEL_DIR = ROOT / "models"

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


def download_and_extract():
    if not ZIP_PATH.exists():
        print("Downloading dataset…")
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    if not EXTRACT_DIR.is_dir():
        print("Extracting…")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)


def find_dataset_root(base: Path) -> Path:
    best, best_n = None, 0
    for dirpath, dirnames, _ in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith(("__", "."))]
        n = sum(1 for d in dirnames
                if any(f.lower().endswith(IMG_EXT) for f in os.listdir(os.path.join(dirpath, d))))
        if n > best_n:
            best, best_n = Path(dirpath), n
    return best


def preprocess(path):
    with Image.open(path) as img:
        img.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))
        return np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.uint8)


def build_model(num_classes):
    aug = Sequential([
        RandomRotation(0.04, fill_mode="nearest"),
        RandomTranslation(0.10, 0.10, fill_mode="nearest"),
        RandomZoom(0.10, 0.10, fill_mode="nearest"),
    ], name="data_augmentation")
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)), aug,
        Conv2D(32, 3, activation="relu"), MaxPooling2D(2),
        Conv2D(64, 3, activation="relu"), MaxPooling2D(2),
        Conv2D(128, 3, activation="relu"), MaxPooling2D(2),
        Flatten(),
        Dense(256, activation="relu"), Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="final_cnn")
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-per-class", type=int, default=None, help="use a subset for a quick test")
    args = ap.parse_args()

    download_and_extract()
    data_root = find_dataset_root(EXTRACT_DIR)
    classes = sorted(d.name for d in data_root.iterdir() if d.is_dir())
    print(f"Dataset: {data_root} | {len(classes)} classes: {classes}")

    paths, labels = [], []
    for i, c in enumerate(classes):
        files = sorted(f for f in (data_root / c).iterdir() if f.suffix.lower() in IMG_EXT)
        if args.max_per_class:
            files = files[: args.max_per_class]
        paths += files; labels += [i] * len(files)

    X = np.array([preprocess(p) for p in tqdm(paths, desc="Preprocessing")], dtype="float32") / 255.0
    y = np.array(labels)

    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    i_val, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    X_tr, y_tr, X_val, y_val, X_te, y_te = X[i_tr], y[i_tr], X[i_val], y[i_val], X[i_te], y[i_te]
    print(f"Train/Val/Test: {len(y_tr)}/{len(y_val)}/{len(y_te)}")

    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    cw = {i: float(w) for i, w in enumerate(cw)}

    MODEL_DIR.mkdir(exist_ok=True)
    ckpt = str(MODEL_DIR / "best_final_cnn.keras")
    model = build_model(len(classes))
    model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs,
        batch_size=args.batch_size, class_weight=cw,
        callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                   ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
                   ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)],
    )

    pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print(f"\nTest accuracy: {accuracy_score(y_te, pred) * 100:.2f}%")
    print(classification_report(y_te, pred, labels=np.arange(len(classes)),
                                target_names=classes, zero_division=0))

    model.save(MODEL_DIR / "IndiSignLang_CNN_Model.keras")
    (MODEL_DIR / "IndiSignLang_class_names.json").write_text(json.dumps(classes))
    print(f"Saved model + class names to {MODEL_DIR}")


if __name__ == "__main__":
    main()
