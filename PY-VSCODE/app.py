"""IndiSignLang - Indian Sign Language Recognition (Streamlit frontend).

Two modes:
  1. Upload Photo   - classify a hand-sign image
  2. Real-Time Webcam - live classification from the camera
Run:  streamlit run app.py
"""
import time

import streamlit as st
from PIL import Image

from utils.predictor import ISLPredictor, model_files_exist, MODEL_PATH

st.set_page_config(page_title="IndiSignLang | ISL Recognition", page_icon="🤟", layout="wide")

# ---------- Light-blue styling ----------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #EAF4FF 0%, #F5FAFF 100%); }
    [data-testid="stSidebar"] { background: #D6EBFF; }
    .hero {
        background: linear-gradient(135deg, #90CAF9 0%, #BBDEFB 60%, #E3F2FD 100%);
        padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(30,136,229,.18);
    }
    .hero h1 { margin: 0; color: #0D47A1; font-size: 2.1rem; }
    .hero p  { margin: .3rem 0 0; color: #1A4D7C; font-size: 1.05rem; }
    .card {
        background: #FFFFFF; border: 1px solid #BBDEFB; border-radius: 16px;
        padding: 1.2rem 1.4rem; box-shadow: 0 2px 10px rgba(30,136,229,.10);
    }
    .big-letter {
        font-size: 5.5rem; font-weight: 800; color: #1565C0; line-height: 1; text-align: center;
    }
    .conf { text-align: center; color: #1A4D7C; font-size: 1.1rem; margin-bottom: .6rem; }
    .bar-wrap { background: #E3F2FD; border-radius: 10px; height: 16px; overflow: hidden; }
    .bar { background: linear-gradient(90deg, #42A5F5, #1E88E5); height: 16px; border-radius: 10px; }
    .row { display: flex; align-items: center; gap: .7rem; margin: .35rem 0; }
    .row .lab { width: 2rem; font-weight: 700; color: #0D47A1; }
    .row .val { width: 4.2rem; text-align: right; color: #1A4D7C; font-size: .9rem; }
    div.stButton > button { background: #1E88E5; color: white; border-radius: 10px; border: 0; }
    div.stButton > button:hover { background: #1565C0; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🤟 IndiSignLang</h1>
      <p>Indian Sign Language recognition with a CNN — upload a photo or sign live on your webcam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the CNN model…")
def get_predictor():
    return ISLPredictor()


def result_card(top):
    """HTML card with the predicted letter and top-k confidence bars."""
    label, conf = top[0]
    bars = "".join(
        f'<div class="row"><span class="lab">{l}</span>'
        f'<div class="bar-wrap" style="flex:1"><div class="bar" style="width:{p*100:.1f}%"></div></div>'
        f'<span class="val">{p*100:.1f}%</span></div>'
        for l, p in top
    )
    return (
        f'<div class="card"><div class="big-letter">{label}</div>'
        f'<div class="conf">Confidence: <b>{conf*100:.1f}%</b></div>{bars}</div>'
    )


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Choose input mode", ["📤 Upload Photo", "🎥 Real-Time Webcam"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Recognised signs (24)**  \n`A B C D E F G H I K L M N O P Q R S T U V W X Y`  \n"
    "*J and Z need motion, so they are not included.*"
)
st.sidebar.info(
    "Tips for best results:\n- Plain, light background\n- One hand, well lit\n- Hand clearly in frame"
)

# ---------- Model check ----------
if not model_files_exist():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.markdown(
        "**Fix it in one of two ways:**\n"
        "1. Run the notebook and copy `IndiSignLang_CNN_Model.keras` and "
        "`IndiSignLang_class_names.json` into the `models/` folder, **or**\n"
        "2. Train here: `python train_model.py` (downloads the dataset and saves the model to `models/`)."
    )
    st.stop()

predictor = get_predictor()

# ---------- Mode 1: Upload photo ----------
if mode == "📤 Upload Photo":
    st.subheader("📤 Upload a hand-sign photo")
    file = st.file_uploader("JPG / JPEG / PNG", type=["jpg", "jpeg", "png"])
    if file is None:
        st.info("Upload an image to get a prediction.")
    else:
        img = Image.open(file).convert("RGB")
        left, right = st.columns([1.1, 1])
        with left:
            st.image(img, caption="Uploaded image", use_container_width=True)
        with right:
            with st.spinner("Predicting…"):
                top = predictor.predict_pil(img, top_k=3)
            st.markdown(result_card(top), unsafe_allow_html=True)
            if top[0][1] < 0.60:
                st.warning("Low confidence — try a clearer photo on a plain background.")

# ---------- Mode 2: Real-time webcam ----------
else:
    st.subheader("🎥 Real-time webcam recognition")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        from utils.webcam import SignProcessor
    except Exception as e:  # pragma: no cover
        st.error(f"Real-time mode needs `streamlit-webrtc` and `av`: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    use_roi = c1.checkbox("Use centre box (ROI)", value=False,
                          help="Predict only on the highlighted square. Helpful if the background is busy.")
    min_conf = c2.slider("Min confidence", 0.30, 0.95, 0.60, 0.05)
    smooth_n = c3.slider("Smoothing (frames)", 1, 15, 7)

    col_video, col_res = st.columns([1.6, 1])
    with col_video:
        ctx = webrtc_streamer(
            key="isl-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=SignProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
    with col_res:
        placeholder = st.empty()
        placeholder.info("Click **START**, allow camera access, and show a sign to the camera.")

    if ctx.state.playing:
        while ctx.state.playing:
            proc = ctx.video_processor
            if proc is not None:
                proc.predictor = predictor
                proc.use_roi = use_roi
                proc.min_conf = min_conf
                proc.smooth_n = smooth_n
                if proc.top3:
                    top = [(proc.label if i == 0 else l, proc.confidence if i == 0 else p)
                           for i, (l, p) in enumerate(proc.top3)]
                    placeholder.markdown(result_card(top), unsafe_allow_html=True)
            time.sleep(0.25)

st.markdown(
    "<hr><p style='text-align:center;color:#5B7FA3'>PRAICP-1000 · Indian Sign Language Recognition · CNN (TensorFlow/Keras)</p>",
    unsafe_allow_html=True,
)
