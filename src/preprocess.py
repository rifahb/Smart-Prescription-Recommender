import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def symptom_tokenizer(x):
    if not isinstance(x, str):
        return []
    return [s.strip().lower() for s in x.split(',') if s.strip()]

def treatment_splitter(x):
    if not isinstance(x, str):
        return []
    return [t.strip() for t in str(x).split(',') if t.strip()]

def load_data(path='data/Diseases_Symptoms.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Symptoms', 'Treatments'])
    vectorizer = TfidfVectorizer(tokenizer=symptom_tokenizer)
    symptom_features = vectorizer.fit_transform(df['Symptoms'])
    df['Contagious'] = df['Contagious'].astype(int)
    df['Chronic'] = df['Chronic'].astype(int)
    traits = df[['Contagious', 'Chronic']].values
    X = hstack([symptom_features, traits])
    df['Treatments'] = df['Treatments'].apply(treatment_splitter)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['Treatments'])
    return X, y, mlb.classes_, vectorizer

def load_disease_data(path='data/Diseases_Symptoms.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Symptoms', 'Name'])
    vectorizer = TfidfVectorizer(tokenizer=symptom_tokenizer)
    symptom_features = vectorizer.fit_transform(df['Symptoms'])
    df['Contagious'] = df['Contagious'].astype(int)
    df['Chronic'] = df['Chronic'].astype(int)
    traits = df[['Contagious', 'Chronic']].values
    X = hstack([symptom_features, traits])
    lb = LabelBinarizer()
    y = lb.fit_transform(df['Name'])
    return X, y, lb.classes_, vectorizer, lb

def get_disease_to_treatments(path='data/Diseases_Symptoms.csv'):
    df = pd.read_csv(path)
    mapping = df.groupby('Name')['Treatments'].apply(
        lambda x: list(set([t.strip() for tx in x for t in str(tx).split(',') if t.strip()]))
    ).to_dict()
    return mapping