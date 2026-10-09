
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
DATA_PATH = BASE_DIR / "data" / "reviews.csv"

st.set_page_config(
    page_title="SentiScope AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #141b33 100%);
    color: #f4f7ff;
}
[data-testid="stSidebar"] {
    background-color: #10172a;
}
.metric-card {
    background: #1c2742;
    border: 1px solid #344466;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 10px;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(110deg, #4338ca, #7c3aed, #0e7490);
    margin-bottom: 22px;
}
.hero h1 { color: white; font-size: 38px; }
.hero p { color: #e5e7eb; font-size: 16px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        artifacts = pickle.load(file)
    return artifacts["model"], artifacts["metrics"]


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

model, metrics = load_artifacts()

st.markdown("""
<div class="hero">
    <h1>🧠 SentiScope AI</h1>
    <p>Intelligent review analytics powered by Natural Language Processing
    and Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.caption("SyntecxHub AI Internship Project | NLP • TF-IDF • Logistic Regression")

with st.sidebar:
    st.title("SentiScope AI")
    page = st.radio(
        "Navigation",
        ["Overview", "Analyze Text", "Batch Analysis", "About Model"]
    )
    st.divider()
    st.caption("Built with Python, scikit-learn, Streamlit and Plotly")


if page == "Overview":
    st.subheader("Model Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Macro F1-Score", f"{metrics['macro_f1']:.3f}")
    c3.metric("Test Samples", f"{metrics['test_samples']:,}")

    st.divider()
    st.subheader("Explore the Dataset")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        left, right = st.columns(2)

        with left:
            counts = df["sentiment"].value_counts().rename_axis(
                "Sentiment"
            ).reset_index(name="Reviews")

            fig = px.bar(
                counts, x="Sentiment", y="Reviews",
                color="Sentiment",
                title="Sentiment Distribution",
                color_discrete_map={
                    "positive": "#22c55e",
                    "negative": "#f43f5e"
                }
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            source_counts = df["source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Reviews"]
            fig2 = px.pie(
                source_counts, names="Source", values="Reviews",
                title="Reviews by Dataset Source", hole=0.55
            )
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        st.write("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)


elif page == "Analyze Text":
    st.subheader("Analyze a Review")
    st.write("Enter a product, service, or movie review.")

    text = st.text_area(
        "Review text",
        placeholder="Example: The quality is excellent and I love this product!",
        height=140
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
            confidence = float(max(probabilities))

            if prediction == "positive":
                st.success("Predicted sentiment: POSITIVE 😊")
            else:
                st.error("Predicted sentiment: NEGATIVE 😞")

            st.metric("Model confidence", f"{confidence:.1%}")
            prob_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities
            })
            fig = px.bar(
                prob_df, x="Sentiment", y="Probability",
                color="Sentiment",
                title="Prediction Probabilities",
                color_discrete_map={
                    "positive": "#22c55e",
                    "negative": "#f43f5e"
                }
            )
            fig.update_layout(template="plotly_dark", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee of correctness."
            )


elif page == "Batch Analysis":
    st.subheader("Batch Review Analysis")
    st.write("Upload a CSV file containing a column named `text`.")

    uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            if "text" not in batch.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                batch = batch.dropna(subset=["text"]).copy()
                batch["text"] = batch["text"].astype(str)
                batch["sentiment"] = model.predict(batch["text"].tolist())

                probabilities = model.predict_proba(batch["text"].tolist())
                batch["confidence"] = probabilities.max(axis=1)

                st.metric("Reviews analyzed", len(batch))
                counts = batch["sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Reviews"]

                fig = px.pie(
                    counts, names="Sentiment", values="Reviews",
                    title="Batch Sentiment Breakdown", hole=0.5,
                    color="Sentiment",
                    color_discrete_map={
                        "positive": "#22c55e",
                        "negative": "#f43f5e"
                    }
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(batch, use_container_width=True)

                csv = batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Analysis CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
        except Exception as error:
            st.error(f"Could not process the file: {error}")


else:
    st.subheader("About the Model")

    st.markdown("""
    **Pipeline**
    1. Input review text
    2. TF-IDF converts text into numerical features using unigrams and bigrams
    3. Logistic Regression predicts positive or negative sentiment
    4. The app displays the prediction and estimated class probabilities

    **Training dataset**
    - 3,000 labeled sentences
    - 1,000 Amazon reviews
    - 1,000 IMDb reviews
    - 1,000 Yelp reviews

    **Evaluation**
    - 80% training split
    - 20% held-out test split
    - Stratified split with random state 42
    - Accuracy and macro F1-score reported from the test set

    **Limitations**
    - The dataset is relatively small and consists of labeled sentences.
    - The model predicts only positive or negative sentiment.
    - Sarcasm, mixed opinions, and domain-specific language can cause errors.
    - Estimated probabilities may not be calibrated confidence scores.
    """)

    st.info(
        "This is a learning and demonstration project. "
        "Predictions should not be treated as guaranteed assessments."
    )

st.divider()
st.caption("SentiScope AI • Educational NLP project")
