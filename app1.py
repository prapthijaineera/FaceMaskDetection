import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time
import pandas as pd

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ──────────────────────────────────────────────────────────────────
IMG_HEIGHT   = 224
IMG_WIDTH    = 224

# Keras sorts class folders alphabetically:
#   'with_mask' → 0  |  'without_mask' → 1
# Verify with: print(train_generator.class_indices)
CLASS_LABELS = ["With Mask", "Without Mask"]

CLASS_INFO = {
    "With Mask": {
        "icon"       : "😷",
        "emoji_large": "✅",
        "color"      : "#28a745",
        "bg"         : "#d4edda",
        "verdict"    : "Mask Detected",
        "message"    : (
            "Great job! The person is **wearing a face mask** correctly. "
            "This helps reduce the spread of airborne diseases and protects "
            "both the wearer and those around them."
        ),
        "tips": [
            "Ensure the mask covers both the nose and mouth fully.",
            "Use a well-fitted mask with no gaps at the sides.",
            "Replace disposable masks after each use.",
            "Wash reusable cloth masks after every use.",
            "Avoid touching the mask while wearing it.",
        ],
    },
    "Without Mask": {
        "icon"       : "🚫",
        "emoji_large": "❌",
        "color"      : "#dc3545",
        "bg"         : "#f8d7da",
        "verdict"    : "No Mask Detected",
        "message"    : (
            "⚠️ The person is **not wearing a face mask**. "
            "In crowded or enclosed spaces, wearing a mask significantly "
            "reduces the risk of transmitting or contracting respiratory illnesses."
        ),
        "tips": [
            "Wear a surgical or N95 mask in public spaces.",
            "Maintain at least 1 metre distance from others.",
            "Wash hands frequently with soap for at least 20 seconds.",
            "Avoid touching your face, eyes, nose, or mouth.",
            "Follow local health guidelines and regulations.",
        ],
    },
}

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-banner {
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 18px;
    }
    .result-title  { font-size: 2rem; font-weight: 700; margin: 0 0 4px; }
    .result-sub    { font-size: 1.1rem; margin: 0; }
    .tip-item      { padding: 6px 0; font-size: 0.97rem; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "facemask_model.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ─── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 😷 Face Mask Detector")
    st.markdown("---")
    st.markdown(
        """
        This app uses a **CNN deep learning model** trained on the
        [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
        to detect whether a person is wearing a face mask.

        ---
        ### 🧠 Model Details
        | Property | Value |
        |---|---|
        | Architecture | CNN (4 × Conv2D) |
        | Input Size | 224 × 224 px |
        | Classes | 2 |
        | Optimizer | Adam (lr=0.001) |
        | Epochs | 30 |

        ---
        ### 📌 Classes
        | Label | Index |
        |---|---|
        | ✅ With Mask | 0 |
        | ❌ Without Mask | 1 |

        ---
        ### 🔍 How to Use
        1. Upload a face image (JPG/PNG).
        2. Click **Detect Mask**.
        3. View prediction + confidence.

        ---
        *Built with TensorFlow & Streamlit*
        """
    )

# ─── Header ─────────────────────────────────────────────────────────────────────
st.title("😷 Face Mask Detection")
st.markdown(
    "Upload a **face image** and the AI model will instantly detect whether "
    "the person is **wearing a mask** or **not wearing a mask**."
)
st.divider()

# ─── Load Model ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading model…"):
    model = load_model()

if model is None:
    st.error(
        "**Model file `face_mask_model.h5` not found.** "
        "Please place it in the same directory as `app.py` and restart.",
        icon="🚫",
    )
    st.info(
        "**Save your model** by adding this to the end of your notebook:\n"
        "```python\ncnn_model.save('face_mask_model.h5')\n"
        "# In Colab, download it:\n"
        "from google.colab import files\n"
        "files.download('face_mask_model.h5')\n```",
        icon="💡",
    )
    st.stop()

st.success("✅ Model loaded and ready!", icon="🧠")
st.divider()

# ─── Upload & Preview ───────────────────────────────────────────────────────────
col_up, col_prev = st.columns([1, 1], gap="large")

with col_up:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader(
        "Choose an image with a face",
        type=["jpg", "jpeg", "png", "webp"],
        help="Best results with a clear, frontal face photo.",
    )

    if uploaded:
        img = Image.open(uploaded)
        detect_btn = st.button("🔍 Detect Mask", type="primary", use_container_width=True)
    else:
        st.info("👆 Upload an image to get started.", icon="📁")
        detect_btn = False

with col_prev:
    if uploaded:
        st.subheader("🖼️ Uploaded Image")
        st.image(img, use_container_width=False, caption=uploaded.name)

# ─── Prediction ─────────────────────────────────────────────────────────────────
if uploaded and detect_btn:
    st.divider()
    st.subheader("🧪 Detection Results")

    with st.spinner("Analysing image…"):
        time.sleep(0.3)
        arr         = preprocess(img)
        preds       = model.predict(arr, verbose=0)[0]

    pred_idx   = int(np.argmax(preds))
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(preds[pred_idx]) * 100
    info       = CLASS_INFO[pred_label]

    # ── Result Banner ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-banner" style="
            background: linear-gradient(135deg, {info['bg']}, #ffffff);
            border-left: 7px solid {info['color']};
        ">
            <p class="result-title" style="color:{info['color']};">
                {info['emoji_large']} {info['verdict']}
            </p>
            <p class="result-sub">
                Confidence: <strong>{confidence:.1f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Detail Columns ─────────────────────────────────────────────────────────
    col_detail, col_chart = st.columns([1.2, 1], gap="large")

    with col_detail:
        st.markdown("#### 📋 Analysis")
        st.markdown(info["message"])

        st.markdown("#### 💡 Health Tips")
        for tip in info["tips"]:
            st.markdown(f"- {tip}")

    with col_chart:
        st.markdown("#### 📊 Confidence Scores")
        chart_df = pd.DataFrame(
            {
                "Class": CLASS_LABELS,
                "Confidence (%)": [round(float(p) * 100, 2) for p in preds],
            }
        ).set_index("Class")
        st.bar_chart(chart_df, height=240, color="#1f77b4")

        st.markdown("#### 🔢 Raw Probabilities")
        for label, prob in zip(CLASS_LABELS, preds):
            pct = float(prob) * 100
            st.markdown(f"**{label}** &nbsp; `{pct:.2f}%`")
            st.progress(int(pct))

    # ── Metrics Row ────────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Prediction",  pred_label)
    m2.metric("📈 Confidence",  f"{confidence:.1f}%")
    m3.metric("🔢 Class Index", str(pred_idx))

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ *This tool is intended for educational and demonstration purposes only. "
        "It is not a substitute for certified safety or health compliance systems.*"
    )
