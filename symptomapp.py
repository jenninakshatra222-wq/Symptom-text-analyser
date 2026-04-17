import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import numpy as np
import re
from scipy.special import softmax
from google import genai


MODEL_PATH = "./bert_symptom_model"
LABELS_PATH = "label_classes.pkl"
import os
from google import genai
client = genai.Client(api_key="YOUR_API_KEY")

for m in client.models.list():
    print(m.name)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

with open(LABELS_PATH, "rb") as f:
    label_classes = pickle.load(f)


def predict(text):
    cleaned = re.sub(r'[^a-z0-9\s]', '', text.lower())
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = softmax(outputs.logits.cpu().numpy(), axis=1)[0]
    top_idx = np.argsort(probs)[::-1][:3]
    top_probs = [min(probs[i] * 3, 1.0) for i in top_idx]
    diseases = [label_classes[i] for i in top_idx]

    return diseases, top_probs


def get_medical_response(disease, symptoms):
    prompt = f"""
You are a medical assistant.

Symptoms: {symptoms}
Disease: {disease}

Give SPECIFIC medical advice for {disease} only.

Format:
Condition: {disease}

Medical Advice:
(3–4 sentences specific to this disease)

Recommended Specialist:
(only one specialist, e.g., Dermatologist, Cardiologist)
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

EMERGENCY_CONDITIONS = {
    "pneumonia", "dengue", "hypertension", "heart attack",
    "stroke", "meningitis", "sepsis"
}

def check_emergency(disease):
    if disease.lower() in EMERGENCY_CONDITIONS:
        return "\n> 🚨 **This condition may require urgent care. Go to an emergency room or call emergency services immediately.**"
    return ""


st.set_page_config(page_title="AI Medical Chatbot", page_icon="🩺")

st.title("🩺 AI Medical Symptom Checker")
st.caption("Describe your symptoms and I'll help identify possible conditions and next steps.")
st.warning("⚠️ This tool is for informational purposes only and does not replace professional medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Greeting
    greeting = "Hi! I'm your AI symptom checker. Please describe what you're experiencing — for example: *'I have a headache, fever, and body aches.'*"
    st.session_state.messages.append({"role": "assistant", "content": greeting})


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


user_input = st.chat_input("Describe your symptoms here...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Analyzing your symptoms..."):
        diseases, probs = predict(user_input)
        main_disease = diseases[0]
        main_conf = probs[0]

        medical_response = get_medical_response(main_disease, user_input)
        emergency_alert = check_emergency(main_disease)

    
    reply = f"""{medical_response}

---


🔎 **Other possibilities:**
- {diseases[1]} 
- {diseases[2]} 
{emergency_alert}
---
*Not a diagnosis. Please consult a licensed healthcare professional for proper evaluation.*"""

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
st.write("Testing Gemini...")

