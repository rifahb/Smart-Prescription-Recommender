import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def symptom_tokenizer(x):
    # Handles empty string or NaN inputs safely
    if not isinstance(x, str):
        return []
    return [s.strip().lower() for s in x.split(',') if s.strip()]

def treatment_splitter(x):
    if not isinstance(x, str):
        return []
    return [t.strip() for t in str(x).split(',') if t.strip()]

def load_data(path='data/Diseases_Symptoms.csv'):
    df = pd.read_csv(path)

    # Drop rows with missing critical data
    df = df.dropna(subset=['Symptoms', 'Treatments'])

    # Convert symptoms into vectorized TF-IDF features
    vectorizer = TfidfVectorizer(tokenizer=symptom_tokenizer)
    symptom_features = vectorizer.fit_transform(df['Symptoms'])

    # Use Contagious and Chronic as binary features
    df['Contagious'] = df['Contagious'].astype(int)
    df['Chronic'] = df['Chronic'].astype(int)
    traits = df[['Contagious', 'Chronic']].values

    # Combine TF-IDF symptom features and traits
    X = hstack([symptom_features, traits])

    # Process treatments for multi-label classification
    df['Treatments'] = df['Treatments'].apply(treatment_splitter)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['Treatments'])

    return X, y, mlb.classes_, vectorizer
