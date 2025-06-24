from tensorflow.keras.models import load_model

model = load_model("models/disease_model.keras")
model.summary()
