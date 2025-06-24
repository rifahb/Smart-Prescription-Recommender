# 💊 Smart Prescription Recommender

An AI-powered system that recommends possible treatment classes based on user-reported symptoms and disease traits. Built using a custom deep learning model trained on real-world medical data from Kaggle, this project showcases applied multi-label classification in healthcare.

---

## 📌 Project Objective

To build a practical, explainable, and human-interactive tool that:

- Accepts symptoms and disease attributes as input
- Predicts relevant treatment categories (e.g. medications, therapies, surgeries)
- Groups and presents results clearly for real-world use (e.g. hospitals, chatbots, self-diagnosis tools)

---

## 🧠 How It Works

1. **Input**:  
   Users enter symptoms (e.g. `"itchy skin, small blisters"`) and select traits like contagious/chronic.

2. **Processing**:  
   Symptoms are vectorized using **TF-IDF**, combined with binary disease traits.

3. **Model**:  
   A custom **multi-label MLP (Neural Network)** trained on labeled treatment data from Kaggle.

4. **Prediction**:  
   Outputs top treatment suggestions grouped by category and shown in a structured UI.

---

## 🚀 Demo

👉 **Try it live (hosted on Streamlit)**  
[https://your-streamlit-app-link](#) ← *(Replace with your actual Streamlit Cloud link)*

---

## 📂 Dataset

Sourced from Kaggle (medical conditions, symptoms, treatments).  
Dataset fields include:

- `Name` – Disease name  
- `Symptoms` – Comma-separated symptoms  
- `Treatments` – List of recommended treatments  
- `Contagious`, `Chronic` – Binary flags  
- `Disease_Code` – Optional code

> Sensitive personal data is NOT included. All data is for academic/demo purposes only.

---

## 📊 Model Performance

| Metric         | Score |
|----------------|-------|
| Micro F1 Score | 0.91  |
| Macro F1 Score | 0.92  |
| Weighted Avg   | 0.90  |
| Samples Avg    | 0.86  |

✅ Model performs excellently across frequent and rare treatment classes.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit  
- **Model**: Keras (Multi-label Neural Network)  
- **Processing**: Scikit-learn, Pandas  
- **Deployment**: Streamlit Cloud (or local)

---

## 📦 How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/smart-prescription-recommender.git
   cd smart-prescription-recommender
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   streamlit run app.py
   ```

   ---
## Sample Output
✅ Recommended Treatments:

🔹 Medication
- Antibiotics (oral or IV)
- Pain relievers
- Immunotherapy

🔹 Surgery
- Surgical drainage
- Abscess removal

🔹 Supportive Care
- Warm compresses
- Hydration

---
## Future Improvements
- **Model Optimization**: Fine-tune the model for better accuracy and efficiency.
- **User Feedback**: Implement a feedback mechanism to improve the model over time.
- **Multi-Domain Integration**: Expand the model to cover more domains beyond just medicine.

---
## 📝 License
  This project is licensed for academic and demo use.
  Original data belongs to respective Kaggle sources.

 ---
 ## 👩‍⚕️ Built With ❤️ For Healthcare + AI
   Made by Rifah Balquees – rifahb03@gmail.com 

