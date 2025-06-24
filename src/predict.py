import numpy as np
import joblib
from tensorflow.keras.models import load_model

def predict(symptom_text, contagious, chronic):
    model = load_model("models/recommender_model.h5")
    class_names = joblib.load("models/class_names.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Convert input
    X_symptom = vectorizer.transform([symptom_text])
    X_traits = np.array([[int(contagious), int(chronic)]])
    
    from scipy.sparse import hstack
    X_input = hstack([X_symptom, X_traits]).toarray()

    prediction = model.predict(X_input)[0]
    print("Model raw predictions:", prediction)  # Add this line for debugging
    result = [class_names[i] for i, p in enumerate(prediction) if p > 0.01]
    return result
