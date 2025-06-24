import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.preprocess import load_disease_data

# Load data
X, y, class_names, vectorizer, label_binarizer = load_disease_data()

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')  # softmax for multiclass
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

# Train
history = model.fit(X.toarray(), y, epochs=50, batch_size=8, validation_split=0.1)

# Save model and assets
os.makedirs("models", exist_ok=True)
model.save("models/disease_model.keras")
joblib.dump(vectorizer, "models/disease_vectorizer.pkl")
joblib.dump(class_names, "models/disease_class_names.pkl")
joblib.dump(label_binarizer, "models/disease_label_binarizer.pkl")

print("Disease model and assets saved!")