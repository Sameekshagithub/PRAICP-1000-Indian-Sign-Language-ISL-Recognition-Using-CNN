"""Real-time webcam processor for streamlit-webrtc."""
import threading
from collections import Counter, deque

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.predictor = None          # set from the main thread once the model is loaded
        self.use_roi = False           # predict only on a centred square box
        self.predict_every = 3         # run the CNN every N frames (speed)
        self.min_conf = 0.60           # ignore predictions below this confidence
        self.smooth_n = 7              # majority vote over last N predictions
        self._frame_i = 0
        self._history = deque(maxlen=15)
        self._lock = threading.Lock()
        self.label = "-"
        self.confidence = 0.0
        self.top3 = []

    @staticmethod
    def _roi(img):
        h, w = img.shape[:2]
        s = int(min(h, w) * 0.7)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return img[y0:y0 + s, x0:x0 + s], (x0, y0, s)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        self._frame_i += 1
        roi_box = None

        if self.use_roi:
            _, roi_box = self._roi(bgr)

        if self.predictor is not None and self._frame_i % max(1, self.predict_every) == 0:
            src = self._roi(bgr)[0] if self.use_roi else bgr
            # the model sees the un-mirrored image, exactly like the training photos
            rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            top3 = self.predictor.predict_array(rgb, top_k=3)
            lbl, conf = top3[0]
            with self._lock:
                self.top3 = top3
                self._history.append((lbl, conf) if conf >= self.min_conf else (None, conf))
                window = list(self._history)[-self.smooth_n:]
                recent = [l for l, _ in window if l]
                if recent:
                    best, _ = Counter(recent).most_common(1)[0]
                    self.label = best
                    self.confidence = float(np.mean([c for l, c in window if l == best]))
                else:
                    self.label, self.confidence = "?", conf

        out = cv2.flip(bgr, 1)  # mirror for a natural webcam view
        if roi_box is not None:
            x0, y0, s = roi_box  # centred box, so mirroring keeps it in the same place
            cv2.rectangle(out, (x0, y0), (x0 + s, y0 + s), (229, 136, 30), 3)

        with self._lock:
            text = f"{self.label}  {self.confidence * 100:.0f}%"
        cv2.rectangle(out, (0, 0), (270, 60), (255, 244, 230), -1)
        cv2.putText(out, text, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 80, 20), 3, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(out, format="bgr24")
