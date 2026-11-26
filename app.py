"""Streamlit interface for the Fake vs Real News text classifier."""

from pathlib import Path

import joblib
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "models" / "fake_real_model.joblib"


@st.cache_resource
def load_model():
    """Load the trained model once and reuse it when Streamlit reruns the app."""
    return joblib.load(MODEL_FILE)


def get_key_ngrams(model, text: str, predicted_label: str, limit: int = 6):
    """Return matching words/phrases that support the predicted class most."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    # Convert only this pasted text using the vocabulary learned during training.
    text_vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    # For binary logistic regression, positive weights point to classes_[1].
    direction = 1
    if predicted_label == classifier.classes_[0]:
        direction = -1

    # Visit only phrases present in this text and calculate their contribution.
    ranked = []
    for feature_index, tfidf_value in zip(text_vector.indices, text_vector.data):
        learned_weight = weights[feature_index]
        contribution = tfidf_value * learned_weight * direction
        if contribution > 0:
            ranked.append((feature_names[feature_index], float(contribution)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


st.set_page_config(
    page_title="Fake vs Real News Verifier",
    page_icon="📰",
    layout="centered",
)

st.title("Fake vs Real News Verifier")
st.caption(
    "A text-pattern classifier trained on the provided dataset. "
    "It does not independently fact-check current events."
)

if not MODEL_FILE.exists():
    st.error("The trained model is not here yet. Run `python3 src/train_model.py` first.")
    st.stop()

with st.spinner("Loading the trained model..."):
    model = load_model()

news_text = st.text_area(
    "Paste a news headline or article",
    height=220,
    placeholder="Paste the headline or article text here...",
)

if st.button("Analyze text", type="primary"):
    if not news_text.strip():
        st.warning("Paste a headline or article before analyzing.")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        class_scores = dict(zip(model.classes_, probabilities))

        st.subheader(f"Prediction: {prediction}")

        fake_column, real_column = st.columns(2)
        with fake_column:
            st.metric("Fake score", f"{class_scores.get('Fake', 0):.1%}")
        with real_column:
            st.metric("Real score", f"{class_scores.get('Real', 0):.1%}")

        st.subheader("Key n-grams for this prediction")
        key_ngrams = get_key_ngrams(model, news_text, prediction)
        if key_ngrams:
            for phrase, score in key_ngrams:
                st.write(f"• **{phrase}** — contribution score {score:.3f}")
        else:
            st.write("No learned words or phrases from the training vocabulary matched this text.")

        st.caption(
            "Scores show the model's estimates, not proof that a story is true or false. "
            "The n-grams explain patterns the model used; they are not evidence."
        )
