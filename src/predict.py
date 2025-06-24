import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.sparse import hstack
import os
import os, base64

def predict(symptom_text, contagious, chronic):
    model = load_model("models/recommender_model.h5")
    class_names = joblib.load("models/class_names.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Input processing
    X_symptom = vectorizer.transform([symptom_text])
    X_traits = np.array([[int(contagious), int(chronic)]])
    X_input = hstack([X_symptom, X_traits]).toarray()

    # Predict treatment classes
    prediction = model.predict(X_input)[0]
    result = [class_names[i] for i, p in enumerate(prediction) if p > 0.5]
    return result
def load_model_from_b64file():
    model_path = "models/disease_model.keras"
    b64_path = "disease_model_b64.txt"

    if not os.path.exists(model_path):
        with open(b64_path, "r") as b64file:
            decoded = base64.b64decode(b64file.read())
        with open(model_path, "wb") as model_file:
            model_file.write(decoded)

    return load_model(model_path)


def predict_disease(symptom_text, contagious, chronic, top_k=3):
    #model = load_model(os.path.join("models", "disease_model.keras"))
    

    model = load_model_from_b64file()
    print("✅ Exists:", os.path.exists(model))
    print("📦 Size:", os.path.getsize(model) if os.path.exists(model) else "Missing!")
    class_names = joblib.load("models/disease_class_names.pkl")
    vectorizer = joblib.load("models/disease_vectorizer.pkl")

    # Input processing
    X_symptom = vectorizer.transform([symptom_text])
    X_traits = np.array([[int(contagious), int(chronic)]])
    X_input = hstack([X_symptom, X_traits]).toarray()

    # Predict probabilities
    prediction = model.predict(X_input)[0]
    top_indices = prediction.argsort()[-top_k:][::-1]

    top_diseases = [(class_names[i], round(prediction[i] * 100, 2)) for i in top_indices]
    return top_diseases  # returns list of (disease, confidence%)
