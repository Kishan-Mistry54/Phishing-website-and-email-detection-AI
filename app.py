import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch


@st.cache_resource
def load_model():
    model_path = "saved_phishing_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "❌ Phishing" if pred == 1 else "✅ Not Phishing, It is Legit"


st.title("🔐 Phishing Detector")

tab1, tab2, tab3 = st.tabs(["📧 Email Check", "🌐 URL Check", "🧠 Combined Check"])


with tab1:
    st.subheader("Email Phishing Detection")
    subject = st.text_input("Email Subject", key="subj1")
    sender = st.text_input("Sender Email", key="send1")
    if st.button("Check Email", key="check_email"):
        if subject and sender:
            text = f"[EMAIL] {subject} FROM {sender}"
            st.success("Prediction: " + predict(text))
        else:
            st.warning("Please enter both subject and sender.")


with tab2:
    st.subheader("URL Phishing Detection")
    url = st.text_input("Website URL", key="url1")
    if st.button("Check URL", key="check_url"):
        if url:
            text = f"[URL] {url}"
            st.success("Prediction: " + predict(text))
        else:
            st.warning("Please enter a URL.")

with tab3:
    st.subheader("Full Email + URL Check")
    subject_c = st.text_input("Email Subject", key="subj2")
    sender_c = st.text_input("Sender Email", key="send2")
    url_c = st.text_input("Website URL", key="url2")
    if st.button("Check All", key="check_all"):
        if subject_c and sender_c and url_c:
            full_text = f"[EMAIL] {subject_c} FROM {sender_c} [URL] {url_c}"
            st.success("Prediction: " + predict(full_text))
        else:
            st.warning("Please fill out all fields.")
