
---
# AI-Based Patient Symptom Analyzer

###  BERT + Generative AI Healthcare Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Transformers](https://img.shields.io/badge/HuggingFace-BERT-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview

This project is a **hybrid AI-powered healthcare assistant** that predicts diseases from user-entered symptoms and generates **context-aware medical advice and specialist recommendations** through a chatbot interface.

It integrates:

*  **BERT (Transformer Model)** → Disease Prediction
*  **Generative AI (Gemini)** → Medical Advice Generation
*  **Streamlit Chat UI** → User Interaction

---

##  Features

✔ Symptom-based disease prediction
✔ Chatbot-style interface
✔ AI-generated medical advice
✔ Dynamic specialist recommendation
✔ Top-3 predictions with confidence scores
✔ Emergency alert detection 
✔ Clean and interactive UI

---

##  System Architecture

```text
User Input (Symptoms)
        ↓
BERT Model (Prediction)
        ↓
Predicted Disease
        ↓
Generative AI (LLM)
        ↓
Advice + Specialist
        ↓
Streamlit Chatbot UI
```

---

##  Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Model:** BERT (HuggingFace Transformers)
* **Generative AI:** Gemini (Google GenAI SDK)
* **Libraries:** PyTorch, NumPy, Scikit-learn

---

##  Project Structure

```bash
symptom_app/
│
├── app.py
├── label_classes.pkl
└── bert_symptom_model/
      ├── config.json
      ├── model.safetensors
      ├── tokenizer.json
      ├── tokenizer_config.json
```

---

##  Dataset

* **Symptom2Disease Dataset (Kaggle)**
* ~1200 samples
* 24 disease classes

 Dataset is used during training and is **not required during deployment**.

---

##  Installation

### 1️ Clone Repository

```bash
git clone https://github.com/your-username/symptom-analyzer.git
cd symptom-analyzer
```

---

### 2️ Install Dependencies

```bash
pip install streamlit torch transformers numpy scipy scikit-learn google-genai
```

---

### 3️ Set API Key

```bash
set GEMINI_API_KEY=your_api_key_here   # Windows
```

---

### 4️ Run App

```bash
streamlit run app.py
```

---

##  Example

**Input:**

```text
fever headache vomiting
```

**Output:**

* Predicted Disease
* Confidence Score
* AI-generated Advice
* Recommended Specialist

---

##  Disclaimer

This application is for **educational purposes only** and is not a substitute for professional medical advice.

---

##  Future Improvements

*  Use BioBERT / ClinicalBERT
*  Improve dataset size
*  Voice input support
*  Multi-turn conversation memory
*  Deploy as web app
*  Multi-label prediction

---

## Highlights

* Hybrid AI system (Predictive + Generative AI)
* Real-time chatbot interaction
* ~97% classification accuracy
* Practical healthcare application


## Acknowledgements

* HuggingFace Transformers
* Kaggle Dataset
* Google GenAI SDK


