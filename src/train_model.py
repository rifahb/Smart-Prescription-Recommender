# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from src.preprocess import load_data
# import joblib
# import os
# from sklearn.metrics import classification_report  # <-- Add this import


# X, y, class_names, vectorizer = load_data()

# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X.shape[1],)),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(len(class_names), activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
# model.fit(X.toarray(), y, epochs=50, batch_size=8)  # Convert sparse to dense

# y_pred = model.predict(X.toarray()) > 0.5
# print("\nClassification Report:\n")
# print(classification_report(y, y_pred, target_names=class_names))
# # Save model + assets
# os.makedirs("models", exist_ok=True)
# model.save("models/recommender_model.h5")
# joblib.dump(class_names, "models/class_names.pkl")
# joblib.dump(vectorizer, "models/vectorizer.pkl")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from src.preprocess import load_data
import joblib, os
from sklearn.metrics import classification_report, f1_score
import numpy as np

X, y, class_names, vectorizer = load_data()

model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X.toarray(), y, epochs=50, batch_size=8)

# Tune threshold
best_t = 0.5
best_f1 = 0
for t in np.arange(0.3, 0.7, 0.05):
    preds = (model.predict(X.toarray()) > t).astype(int)
    f1 = f1_score(y, preds, average='micro')
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"\nBest Threshold: {best_t} | F1 Score: {best_f1:.4f}\n")

# Final Predictions
y_pred = model.predict(X.toarray()) > best_t
print("Classification Report:\n")
print(classification_report(y, y_pred, target_names=class_names))

# Save
os.makedirs("models", exist_ok=True)
model.save("models/recommender_model.keras")
joblib.dump(class_names, "models/class_names.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

