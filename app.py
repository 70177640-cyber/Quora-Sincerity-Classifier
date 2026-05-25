import streamlit as st
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Quora Sincere vs Insincere Classifier",
    page_icon="🧠",
    layout="centered"
)

# =========================
# LOAD MODEL & VECTORIZER
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# =========================
# UI HEADER
# =========================
st.title("🧠 Quora Question Classifier")
st.write("Detect whether a question is **Sincere or Insincere** using Machine Learning.")

# =========================
# USER INPUT
# =========================
question = st.text_area("✍️ Enter your question here:")

# =========================
# PREDICTION FUNCTION
# =========================
def predict(text):
    if not text.strip():
        return None

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    return int(prediction)

# =========================
# BUTTON ACTION
# =========================
if st.button("Predict 🚀"):
    result = predict(question)

    if result is None:
        st.warning("Please enter a question before predicting.")
    else:
        if result == 1:
            st.error("❌ Insincere Question Detected")
            st.write("This question appears fake, misleading, or not genuine.")
        else:
            st.success("✅ Sincere Question Detected")
            st.write("This question appears genuine and meaningful.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + Machine Learning")