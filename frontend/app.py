import pickle
import re
import textwrap
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IMDb AI Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "imdb_rnn_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
MAX_LEN = 200


# ============================================================
# HELPER
# ============================================================
# IMPORTANT: st.markdown()/Markdown treats a 4-space indent as a
# CODE BLOCK, not HTML. Since the strings below live inside indented
# Python code, textwrap.dedent() strips that common leading
# whitespace before it ever reaches the Markdown parser — this is
# what makes unsafe_allow_html actually take effect instead of the
# tags being printed as plain text.


def render(html: str) -> None:
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


def clean_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-zA-Z ]", "", text)


@st.cache_resource(show_spinner=False)
def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder


def check_model_status():
    try:
        model, tokenizer, encoder = load_resources()
        return True, int(len(encoder.classes_))
    except Exception:
        return False, 0


def predict_genre(overview: str):
    model, tokenizer, encoder = load_resources()

    cleaned = clean_text(overview)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_genre = str(encoder.inverse_transform([predicted_index])[0])
    confidence = float(probabilities[predicted_index])

    top_indices = np.argsort(probabilities)[-5:][::-1]
    top_predictions = [
        {
            "genre": str(encoder.inverse_transform([int(i)])[0]),
            "confidence": round(float(probabilities[i]), 6),
        }
        for i in top_indices
    ]

    return {
        "genre": predicted_genre,
        "confidence": round(confidence, 6),
        "top_predictions": top_predictions,
    }


# ============================================================
# CUSTOM CSS
# ============================================================

render(
    """
    <style>

    /* =========================
       GLOBAL
    ========================= */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(99,102,241,0.14), transparent 30%),
            radial-gradient(circle at 85% 15%, rgba(168,85,247,0.12), transparent 30%),
            linear-gradient(135deg, #050816 0%, #0b1020 50%, #050816 100%);
        color: #f8fafc;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        max-width: 1180px;
        padding-top: 30px;
        padding-bottom: 50px;
    }

    /* =========================
       NAVBAR
    ========================= */

    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 24px;
        margin-bottom: 70px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 18px;
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    }

    .logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .logo-highlight { color: #a78bfa; margin-left: 6px; }

    .api-status {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 8px 15px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
    }

    .api-status.online {
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac;
    }

    .api-status.offline {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }

    .status-dot.online { background: #22c55e; box-shadow: 0 0 12px rgba(34,197,94,0.9); }
    .status-dot.offline { background: #ef4444; box-shadow: 0 0 12px rgba(239,68,68,0.9); }

    /* =========================
       HERO
    ========================= */

    .hero { text-align: center; max-width: 850px; margin: 0 auto; }

    .hero-badge {
        display: inline-block;
        padding: 9px 17px;
        margin-bottom: 24px;
        border-radius: 30px;
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(129,140,248,0.25);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(46px, 7vw, 80px);
        line-height: 1.02;
        font-weight: 700;
        letter-spacing: -3px;
        margin-bottom: 24px;
    }

    .hero-title span {
        background: linear-gradient(90deg, #818cf8, #a855f7, #d8b4fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 690px;
        margin: auto;
        color: #94a3b8;
        font-size: 17px;
        line-height: 1.8;
    }

    /* =========================
       SECTION HEADINGS
    ========================= */

    .section { margin-top: 90px; margin-bottom: 30px; }

    .section-label {
        color: #818cf8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .section-description { color: #64748b; font-size: 15px; line-height: 1.7; }

    /* =========================
       ANALYSIS CARD
    ========================= */

    .analysis-card {
        padding: 32px;
        background: linear-gradient(145deg, rgba(30,41,59,0.82), rgba(15,23,42,0.65));
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 24px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.30);
        margin-bottom: 20px;
    }

    /* =========================
       TEXT AREA
    ========================= */

    .stTextArea label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    .stTextArea textarea {
        background: #080d1c !important;
        color: #f8fafc !important;
        border: 1px solid rgba(129,140,248,0.22) !important;
        border-radius: 15px !important;
        padding: 17px !important;
        font-size: 15px !important;
        min-height: 180px !important;
    }

    .stTextArea textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 1px #818cf8, 0 0 25px rgba(129,140,248,0.12) !important;
    }

    /* =========================
       BUTTON
    ========================= */

    .stButton > button {
        width: 100%;
        margin-top: 12px;
        padding: 15px 20px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 15px;
        font-weight: 700;
        box-shadow: 0 12px 30px rgba(99,102,241,0.25);
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 40px rgba(139,92,246,0.38);
    }

    /* =========================
       RESULT
    ========================= */

    .result-card {
        margin-top: 30px;
        padding: 35px;
        text-align: center;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(168,85,247,0.08));
        border: 1px solid rgba(129,140,248,0.25);
    }

    .result-label { color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 2px; }

    .result-genre {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 42px;
        font-weight: 700;
        margin-top: 8px;
        background: linear-gradient(90deg, #818cf8, #d8b4fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .confidence { color: #cbd5e1; margin-top: 8px; font-size: 15px; }

    /* =========================
       PREDICTIONS
    ========================= */

    .prediction-card {
        padding: 18px 20px;
        margin-top: 12px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.11);
        border-radius: 15px;
    }

    .prediction-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .prediction-name { color: #e2e8f0; font-size: 14px; font-weight: 600; }
    .prediction-score { color: #a5b4fc; font-size: 14px; font-weight: 700; }

    .prediction-track {
        height: 7px;
        background: #1e293b;
        border-radius: 20px;
        overflow: hidden;
    }

    .prediction-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        border-radius: 20px;
    }

    /* =========================
       WORKFLOW
    ========================= */

    .workflow-card {
        min-height: 210px;
        padding: 28px;
        background: rgba(15,23,42,0.62);
        border: 1px solid rgba(148,163,184,0.11);
        border-radius: 20px;
        transition: 0.25s ease;
    }

    .workflow-card:hover {
        transform: translateY(-5px);
        border-color: rgba(129,140,248,0.35);
        box-shadow: 0 20px 45px rgba(0,0,0,0.22);
    }

    .workflow-icon { font-size: 30px; margin-bottom: 16px; }
    .workflow-number { color: #818cf8; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 8px; }
    .workflow-title { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; margin-bottom: 10px; }
    .workflow-text { color: #64748b; font-size: 14px; line-height: 1.7; }

    /* =========================
       ARCHITECTURE
    ========================= */

    .architecture-card {
        padding: 38px;
        margin-top: 30px;
        text-align: center;
        background: rgba(15,23,42,0.62);
        border: 1px solid rgba(148,163,184,0.11);
        border-radius: 22px;
    }

    .arch-title { font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700; margin-bottom: 32px; }

    .layer-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .layer {
        min-width: 150px;
        padding: 18px 20px;
        border-radius: 15px;
        background: linear-gradient(145deg, rgba(99,102,241,0.16), rgba(168,85,247,0.08));
        border: 1px solid rgba(129,140,248,0.24);
        color: #e2e8f0;
        font-size: 13px;
        font-weight: 600;
        line-height: 1.5;
    }

    .arrow { color: #818cf8; font-size: 22px; font-weight: 700; }

    /* =========================
       TECHNOLOGY
    ========================= */

    .tech-card {
        min-height: 150px;
        padding: 25px;
        text-align: center;
        background: rgba(15,23,42,0.58);
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 18px;
        transition: 0.25s ease;
    }

    .tech-card:hover { transform: translateY(-4px); border-color: rgba(129,140,248,0.30); }
    .tech-icon { font-size: 28px; margin-bottom: 10px; }
    .tech-name { color: #f1f5f9; font-size: 15px; font-weight: 700; }
    .tech-desc { color: #64748b; font-size: 12px; margin-top: 5px; }

    /* =========================
       FOOTER
    ========================= */

    .footer {
        margin-top: 90px;
        padding-top: 30px;
        border-top: 1px solid rgba(148,163,184,0.10);
        text-align: center;
        color: #64748b;
        font-size: 13px;
        line-height: 2;
    }

    .footer strong { color: #a5b4fc; }

    /* =========================
       MOBILE
    ========================= */

    @media(max-width: 700px) {
        .block-container { padding-left: 18px; padding-right: 18px; }
        .navbar { margin-bottom: 45px; flex-wrap: wrap; gap: 12px; }
        .hero-title { font-size: 40px; letter-spacing: -2px; }
        .hero-description { font-size: 15px; }
        .layer { width: 100%; }
        .arrow { transform: rotate(90deg); }
    }

    </style>
    """
)


# ============================================================
# NAVBAR
# ============================================================

api_online, num_classes = check_model_status()
status_class = "online" if api_online else "offline"
status_text = "Model Loaded" if api_online else "Model Error"

render(
    f"""
    <div class="navbar">
        <div class="logo">
            🎬 IMDb
            <span class="logo-highlight">AI Classifier</span>
        </div>
        <div class="api-status {status_class}">
            <span class="status-dot {status_class}"></span>
            {status_text}
        </div>
    </div>
    """
)
if not api_online:
    st.error("⚠️ The genre classification model could not be loaded.")


# ============================================================
# HERO
# ============================================================

render(
    """
    <div class="hero">
        <div class="hero-badge">ARTIFICIAL INTELLIGENCE • NLP • DEEP LEARNING</div>
        <div class="hero-title">Movie <span>Genre</span><br>Classifier</div>
        <div class="hero-description">
            Enter a movie overview and let our trained SimpleRNN model
            analyze the text and predict its most likely genre.
        </div>
    </div>
    """
)


# ============================================================
# MOVIE ANALYSIS
# ============================================================

render(
    """
    <div class="section">
        <div class="section-label">🎬 Movie Analysis</div>
        <div class="section-title">Describe your movie</div>
        <div class="section-description">
            Give the AI model a short movie overview to classify its genre.
        </div>
    </div>
    """
)

render('<div class="analysis-card">')

overview = st.text_area(
    "Movie Overview",
    height=190,
    placeholder=(
        "Example: A young detective investigates a mysterious murder "
        "and discovers a dangerous conspiracy..."
    ),
    label_visibility="collapsed",
)

predict_clicked = st.button(
    "⚡ Analyze Movie Genre",
    type="primary",
    disabled=not api_online,
)

render("</div>")


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:
    if not overview.strip():
        st.warning("Please enter a movie overview first.")
    else:
        try:
            with st.spinner("🧠 AI is analyzing the movie overview..."):
                data = predict_genre(overview)

            if data:
                genre = data.get("genre", "Unknown")
                confidence = float(data.get("confidence", 0))

                render(
                    f"""
                    <div class="result-card">
                        <div class="result-label">AI PREDICTION</div>
                        <div class="result-genre">{genre}</div>
                        <div class="confidence">
                            Model Confidence: <strong>{confidence * 100:.2f}%</strong>
                        </div>
                    </div>
                    """
                )

                render(
                    """
                    <div class="section">
                        <div class="section-label">🎭 Prediction Results</div>
                        <div class="section-title">Top 5 Predictions</div>
                        <div class="section-description">
                            Most probable genre classes generated by the neural network.
                        </div>
                    </div>
                    """
                )

                predictions = data.get("top_predictions", [])

                for index, item in enumerate(predictions, start=1):
                    item_genre = item.get("genre", "Unknown")
                    item_confidence = float(item.get("confidence", 0))
                    percentage = item_confidence * 100

                    render(
                        f"""
                        <div class="prediction-card">
                            <div class="prediction-header">
                                <div class="prediction-name">#{index}&nbsp;&nbsp;{item_genre}</div>
                                <div class="prediction-score">{percentage:.2f}%</div>
                            </div>
                            <div class="prediction-track">
                                <div class="prediction-fill" style="width: {min(percentage, 100):.2f}%;"></div>
                            </div>
                        </div>
                        """
                    )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ============================================================
# HOW IT WORKS
# ============================================================

render(
    """
    <div class="section">
        <div class="section-label">⚡ How It Works</div>
        <div class="section-title">From Text to Intelligence</div>
        <div class="section-description">
            Three simple stages transform natural language into a genre prediction.
        </div>
    </div>
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    render(
        """
        <div class="workflow-card">
            <div class="workflow-icon">📝</div>
            <div class="workflow-number">STEP 01</div>
            <div class="workflow-title">Input</div>
            <div class="workflow-text">Enter a movie overview written in natural language.</div>
        </div>
        """
    )

with col2:
    render(
        """
        <div class="workflow-card">
            <div class="workflow-icon">🧠</div>
            <div class="workflow-number">STEP 02</div>
            <div class="workflow-title">RNN Analysis</div>
            <div class="workflow-text">
                The tokenizer converts text into sequences processed
                by the trained SimpleRNN.
            </div>
        </div>
        """
    )

with col3:
    render(
        """
        <div class="workflow-card">
            <div class="workflow-icon">🎭</div>
            <div class="workflow-number">STEP 03</div>
            <div class="workflow-title">Prediction</div>
            <div class="workflow-text">
                The neural network calculates the most probable
                movie genre classes.
            </div>
        </div>
        """
    )


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

render(
    f"""
    <div class="section">
        <div class="section-label">🧠 Deep Learning Model</div>
        <div class="section-title">SimpleRNN Architecture</div>
        <div class="section-description">
            The neural network pipeline used for movie genre classification.
        </div>
    </div>

    <div class="architecture-card">
        <div class="arch-title">SimpleRNN Genre Classification Pipeline</div>
        <div class="layer-container">
            <div class="layer">Embedding<br>5,000 × 64</div>
            <div class="arrow">→</div>
            <div class="layer">SimpleRNN<br>64 Units</div>
            <div class="arrow">→</div>
            <div class="layer">Dense<br>32 Units</div>
            <div class="arrow">→</div>
            <div class="layer">Softmax<br>{num_classes or 202} Classes</div>
        </div>
    </div>
    """
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

render(
    """
    <div class="section">
        <div class="section-label">⚙️ Technology Stack</div>
        <div class="section-title">Built With Modern AI Technologies</div>
        <div class="section-description">Technologies powering the application.</div>
    </div>
    """
)

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    render(
        """
        <div class="tech-card">
            <div class="tech-icon">🐍</div>
            <div class="tech-name">Python</div>
            <div class="tech-desc">Programming</div>
        </div>
        """
    )

with tech2:
    render(
        """
        <div class="tech-card">
            <div class="tech-icon">🧠</div>
            <div class="tech-name">TensorFlow</div>
            <div class="tech-desc">Deep Learning</div>
        </div>
        """
    )

with tech3:
    render(
        """
        <div class="tech-card">
            <div class="tech-icon">⚡</div>
            <div class="tech-name">FastAPI</div>
            <div class="tech-desc">REST API</div>
        </div>
        """
    )

with tech4:
    render(
        """
        <div class="tech-card">
            <div class="tech-icon">🎨</div>
            <div class="tech-name">Streamlit</div>
            <div class="tech-desc">Web Interface</div>
        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

render(
    """
    <div class="footer">
        🎬 <strong>IMDb AI Genre Classifier</strong><br>
        Built with Python • TensorFlow • Keras • FastAPI • Streamlit<br>
        Deep Learning NLP Project • SimpleRNN • 202 Genre Classes
    </div>
    """
)

