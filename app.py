import os
os.system("pip install joblib scikit-learn pandas numpy streamlit")
import streamlit as st
import joblib

model = joblib.load("quora_model.pkl")

st.title("Quora Question Sincerity Detector")

st.write("Enter a question below.")

question = st.text_area("Question")

if st.button("Check"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        prediction = model.predict([question])[0]

        if prediction == 0:
            st.success("✅ Sincere Question")
        else:
            st.error("❌ Insincere Question")