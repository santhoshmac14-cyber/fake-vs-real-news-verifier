"""Streamlit interface for the Fake vs Real News text classifier."""

from html import escape
from pathlib import Path
import re

import joblib
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "models" / "fake_real_model.joblib"
NEWSROOM_IMAGE = (
    "https://images.unsplash.com/photo-1651465531201-7e430660fd82"
    "?auto=format&fit=crop&w=1200&q=85"
)

# This is a simple keyword-based topic suggestion. Topic labels are not learned
# by the Fake/Real machine-learning model.
TOPIC_KEYWORDS = {
    "Politics": [
        "president", "congress", "election", "senate", "government",
        "politician", "parliament", "campaign", "vote", "white house",
    ],
    "World News": [
        "war", "international", "border", "foreign", "israel", "russia",
        "ukraine", "china", "united nations", "overseas", "diplomat",
    ],
    "Business": [
        "market", "stock", "company", "business", "economy", "trade",
        "profit", "bank", "financial", "jobs", "price", "oil",
    ],
    "Health": [
        "health", "hospital", "disease", "vaccine", "medical", "doctor",
        "patient", "virus", "medicine", "clinical", "wellness",
    ],
    "Technology": [
        "technology", "tech", "internet", "software", "google", "apple",
        "artificial intelligence", "computer", "cyber", "smartphone", "app",
    ],
    "Science": [
        "research", "scientist", "science", "study", "laboratory", "space",
        "astronomy", "discovery", "experiment", " nasa ",
    ],
    "Sports": [
        "sports", "football", "basketball", "game", "team", "player",
        "coach", "league", "match", "tournament", "championship",
    ],
    "Entertainment": [
        "film", "movie", "actor", "actress", "music", "celebrity",
        "concert", "television", "singer", "album", "hollywood",
    ],
    "Environment": [
        "climate", "environment", "carbon", "wildlife", "conservation",
        "pollution", "renewable", "forest", "drought", "flood",
    ],
}

TOPIC_ICONS = {
    "Politics": "🏛️",
    "World News": "🌍",
    "Business": "📈",
    "Health": "🩺",
    "Technology": "💻",
    "Science": "🔬",
    "Sports": "🏆",
    "Entertainment": "🎬",
    "Environment": "🌱",
    "General / Unclear": "🗞️",
}


@st.cache_resource
def load_model():
    """Load the trained model once and reuse it when Streamlit reruns the app."""
    return joblib.load(MODEL_FILE)


def get_key_ngrams(model, text: str, predicted_label: str, limit: int = 6):
    """Return matching words/phrases that support the predicted class most."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    text_vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    # For binary logistic regression, positive weights point to classes_[1].
    direction = 1 if predicted_label == classifier.classes_[1] else -1

    ranked = []
    for feature_index, tfidf_value in zip(text_vector.indices, text_vector.data):
        contribution = tfidf_value * weights[feature_index] * direction
        if contribution > 0:
            ranked.append((feature_names[feature_index], float(contribution)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def suggest_topic(text: str) -> str:
    """Choose the topic with the most matching keywords in the pasted story."""
    normalized_text = text.lower()
    topic_counts = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        count = 0
        for keyword in keywords:
            if keyword.startswith(" ") or keyword.endswith(" "):
                count += normalized_text.count(keyword.strip())
            else:
                count += len(re.findall(rf"\b{re.escape(keyword)}\b", normalized_text))
        topic_counts[topic] = count

    best_topic = max(topic_counts, key=topic_counts.get)
    if topic_counts[best_topic] == 0:
        return "General / Unclear"
    return best_topic


st.set_page_config(
    page_title="Fake vs Real News Verifier",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(ellipse at 15% 0%, #143554 0%, #0a1728 45%, #07101d 100%);
        color: #eff6ff;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(6, 16, 29, .82); }
    .block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }
    h1 { font-size: clamp(2.3rem, 5vw, 4rem) !important; line-height: 1.05 !important; }
    .live-strip {
        padding: .62rem 1rem; margin-bottom: 1.4rem; border-radius: 8px;
        background: linear-gradient(90deg, #c52d3d, #982236 70%, #73223a);
        color: white; font-size: .82rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; box-shadow: 0 8px 24px rgba(197,45,61,.18);
    }
    .eyebrow { color: #7dd3fc; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .lede { color: #b8c8d9; font-size: 1.07rem; line-height: 1.65; max-width: 52rem; }
    .hero-image img { border: 1px solid rgba(148, 189, 220, .22); border-radius: 18px; box-shadow: 0 20px 52px rgba(0,0,0,.3); }
    .section-label { color: #93a9bf; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .result-banner {
        padding: 1.1rem 1.3rem; margin: 1.1rem 0 1rem; border-radius: 14px;
        background: linear-gradient(110deg, rgba(20,53,84,.96), rgba(13,31,51,.95));
        border: 1px solid rgba(125,211,252,.24);
    }
    .result-label { color: #9eb4c9; font-size: .8rem; text-transform: uppercase; letter-spacing: .12em; }
    .result-value { color: #fff; font-family: 'Manrope', sans-serif; font-size: 2rem; font-weight: 800; }
    .topic-card {
        padding: 1rem 1.2rem; border-radius: 12px; margin: .9rem 0;
        background: rgba(16, 35, 55, .86); border: 1px solid rgba(125,211,252,.2);
    }
    .topic-title { color: #9eb4c9; font-size: .77rem; text-transform: uppercase; letter-spacing: .1em; }
    .topic-value { color: #e9f5ff; font-size: 1.25rem; font-weight: 700; margin-top: .22rem; }
    .explain-note { color: #9eb4c9; font-size: .87rem; line-height: 1.55; }
    .term-chip { display: inline-block; margin: .22rem .28rem .22rem 0; padding: .45rem .72rem;
        border: 1px solid rgba(125,211,252,.28); border-radius: 999px; background: rgba(19,50,76,.85); color: #d9f1ff; }
    [data-testid="stTextArea"] textarea { background: rgba(11,26,43,.92); border: 1px solid #294761; color: #f2f7fb; border-radius: 12px; }
    [data-testid="stTextArea"] textarea:focus { border-color: #38bdf8; box-shadow: 0 0 0 1px #38bdf8; }
    .stButton button[kind="primary"] { background: linear-gradient(90deg,#e1464d,#b92c44); border: 0; border-radius: 10px; min-height: 3rem; font-weight: 700; }
    [data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#38bdf8,#34d399); }
    [data-testid="stProgress"] { background: #1b3044; border-radius: 20px; }
    hr { border-color: rgba(148, 189, 220, .16); }
    </style>
    <div class="live-strip">● &nbsp; Verification desk &nbsp; • &nbsp; Text-pattern analysis &nbsp; • &nbsp; Not a live fact-check</div>
    """,
    unsafe_allow_html=True,
)

hero_text, hero_photo = st.columns([1.2, 1], vertical_alignment="center")
with hero_text:
    st.markdown('<div class="eyebrow">Newsroom • Analysis</div>', unsafe_allow_html=True)
    st.title("Fake vs Real News Verifier")
    st.markdown(
        '<div class="lede">Paste a headline or article to see how the model reads its language, '
        'which topic it may cover, and which phrases influenced the result.</div>',
        unsafe_allow_html=True,
    )
with hero_photo:
    st.image(
        NEWSROOM_IMAGE,
        caption="Broadcast studio photo by Brad Weaver on Unsplash",
        use_container_width=True,
    )

st.markdown("---")
st.markdown(
    '<div class="explain-note">This classroom project recognizes writing patterns from its training dataset. '
    'It does not check sources or independently confirm current events.</div>',
    unsafe_allow_html=True,
)

if not MODEL_FILE.exists():
    st.error("The trained model is not here yet. Run python3 src/train_model.py first.")
    st.stop()

with st.spinner("Loading the trained model..."):
    model = load_model()

st.markdown("### 📝 &nbsp;Paste a story for analysis")
news_text = st.text_area(
    "News text",
    height=220,
    label_visibility="collapsed",
    placeholder="Paste a news headline or article here...",
)

analyze = st.button("Analyze text  →", type="primary", use_container_width=True)

if analyze:
    if not news_text.strip():
        st.warning("Paste a headline or article before analyzing.")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        class_scores = dict(zip(model.classes_, probabilities))
        topic = suggest_topic(news_text)

        st.markdown("---")
        st.markdown('<div class="section-label">Analysis desk • Result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-banner"><div class="result-label">Model prediction</div>'
            f'<div class="result-value">{escape(str(prediction))}</div></div>',
            unsafe_allow_html=True,
        )

        score_col1, score_col2 = st.columns(2)
        with score_col1:
            fake_score = float(class_scores.get("Fake", 0.0))
            st.markdown(f"**Fake score** &nbsp; {fake_score:.1%}")
            st.progress(fake_score, text="Model estimate")
        with score_col2:
            real_score = float(class_scores.get("Real", 0.0))
            st.markdown(f"**Real score** &nbsp; {real_score:.1%}")
            st.progress(real_score, text="Model estimate")

        st.markdown(
            f'<div class="topic-card"><div class="topic-title">Suggested news topic</div>'
            f'<div class="topic-value">{TOPIC_ICONS[topic]} &nbsp; {escape(topic)}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="explain-note">Topic is suggested using simple keyword matching; '
            'it is separate from the trained Fake/Real model.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Key words and phrases")
        key_ngrams = get_key_ngrams(model, news_text, prediction)
        if key_ngrams:
            chips = "".join(
                f'<span class="term-chip">{escape(phrase)}</span>'
                for phrase, _score in key_ngrams
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("No learned words or phrases from the training vocabulary matched this text.")

        st.caption(
            "Confidence bars are model estimates, not proof that a story is true or false. "
            "Highlighted phrases show patterns the model used; they are not evidence."
        )

st.markdown("---")
st.caption("FAKE VS REAL NEWS VERIFIER  •  Academic NLP project  •  TF-IDF + Logistic Regression")
