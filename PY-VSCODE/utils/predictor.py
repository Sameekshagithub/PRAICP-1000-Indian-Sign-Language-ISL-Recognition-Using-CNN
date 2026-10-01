"""Model loading and prediction helpers shared by the upload and real-time modes."""
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 128
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "IndiSignLang_CNN_Model.keras"
CLASSES_PATH = BASE_DIR / "models" / "IndiSignLang_class_names.json"

# Fallback if the JSON file is missing: letters A-Y without J and Z (alphabetical, as in training)
DEFAULT_CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def model_files_exist() -> bool:
    return MODEL_PATH.exists()


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Same steps as training: RGB -> 128x128 (Lanczos) -> scale to [0, 1]."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.asarray(img, dtype="float32") / 255.0


class ISLPredictor:
    def __init__(self):
        import tensorflow as tf  # imported lazily so the UI loads fast

        self.model = tf.keras.models.load_model(str(MODEL_PATH))
        if CLASSES_PATH.exists():
            self.class_names = json.loads(CLASSES_PATH.read_text())
        else:
            self.class_names = DEFAULT_CLASSES
        self._lock = threading.Lock()  # webcam thread + UI thread safety

    def predict_array(self, rgb_uint8: np.ndarray, top_k: int = 3):
        """rgb_uint8: HxWx3 uint8 RGB array. Returns list of (label, probability)."""
        x = preprocess_pil(Image.fromarray(rgb_uint8))[None, ...]
        with self._lock:
            probs = self.model(x, training=False).numpy()[0]
        idx = np.argsort(probs)[::-1][:top_k]
        return [(self.class_names[i], float(probs[i])) for i in idx]

    def predict_pil(self, img: Image.Image, top_k: int = 3):
        return self.predict_array(np.asarray(img.convert("RGB")), top_k)
