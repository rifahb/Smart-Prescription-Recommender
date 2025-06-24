import streamlit as st
import sys
import os
import re

# Add parent path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_disease
from src.preprocess import get_disease_to_treatments

# Page setup
st.set_page_config(page_title="Smart Prescription Recommender", page_icon="💊")
st.title("💊 Smart Prescription Recommender")
st.markdown("Enter patient symptoms and traits below to receive **AI-suggested disease and treatment classes.**")

# Input
symptom_input = st.text_input("📝 Enter symptoms (comma-separated):", "itchy skin, small blisters")
col1, col2 = st.columns(2)
with col1:
    contagious = st.checkbox("🦠 Contagious?")
with col2:
    chronic = st.checkbox("♾️ Chronic?")

# Load disease → treatments map
disease_to_treatments = get_disease_to_treatments()

# Helper: simplify messy treatment text
def simplify_treatment(t):
    if not isinstance(t, str): return ""
    t = t.strip().lower()
    t = re.sub(r'\(.*?\)', '', t)              # Remove text inside parentheses
    t = re.sub(r'[^a-zA-Z0-9\s]', '', t)       # Remove punctuation
    t = re.sub(r'\s+', ' ', t).strip()         # Normalize spaces

    if "antibiotic" in t:
        return "Antibiotics"
    if "antifungal" in t or "fluconazole" in t or "amphotericin" in t:
        return "Antifungal medication"
    if "pain reliever" in t or "ibuprofen" in t or "acetaminophen" in t:
        return "Pain relievers"
    if "rest" in t or "supportive" in t or "hydration" in t or "fluids" in t:
        return "Supportive care"
    if len(t) < 5:
        return ""
    return t.title()

# Recommend button
if st.button("🔍 Recommend Disease & Treatments"):
    top_diseases = predict_disease(symptom_input, contagious, chronic)

    for disease, score in top_diseases:
        st.markdown(f"---")
        st.info(f"🩺 **Possible Disease:** {disease} ({score}%)")

        treatments = disease_to_treatments.get(disease, [])
        if not treatments:
            st.warning("⚠️ No treatments found for this disease.")
            continue

        # Clean & normalize
        seen = set()
        cleaned_result = []
        for t in treatments:
            t = simplify_treatment(t)
            if len(t) > 120:
                t = t[:117] + "..."
            if t and t.lower() not in seen:
                seen.add(t.lower())
                cleaned_result.append(t)

        # Group treatments
        categories = {
            "Medication": [],
            "Surgery": [],
            "Therapy": [],
            "Supportive Care": [],
            "Lifestyle": [],
            "Diagnosis": [],
            "Other": []
        }

        for t in cleaned_result:
            tl = t.lower()
            if any(w in tl for w in ["medication", "drug", "antibiotic", "antifungal", "corticosteroid", "chemotherapy", "immuno", "fluconazole", "amphotericin"]):
                categories["Medication"].append(t)
            elif any(w in tl for w in ["surgery", "surgical", "removal", "abortion", "intervention", "drainage"]):
                categories["Surgery"].append(t)
            elif any(w in tl for w in ["therapy", "rehab", "speech", "physical"]):
                categories["Therapy"].append(t)
            elif any(w in tl for w in ["rest", "hydration", "supportive", "care", "pain", "hospitalization", "management", "fluids"]):
                categories["Supportive Care"].append(t)
            elif any(w in tl for w in ["lifestyle", "exercise", "diet", "avoid", "modification"]):
                categories["Lifestyle"].append(t)
            elif any(w in tl for w in ["diagnose", "imaging", "test", "scan", "urine", "culture"]):
                categories["Diagnosis"].append(t)
            else:
                categories["Other"].append(t)

        # Display treatments per category
        st.success(f"✅ **Recommended Treatments for {disease}:**")
        for group, items in categories.items():
            if items:
                st.subheader(f"🔹 {group}")
                top_items = sorted(items)[:5]
                for item in top_items:
                    st.markdown(f"- {item}")
                remaining = sorted(items)[5:]
                if remaining:
                    with st.expander(f"See all {len(remaining)} additional {group.lower()} options"):
                        for item in remaining:
                            st.markdown(f"- {item}")

        st.caption(f"🧾 Showing top 5 per category for **{disease}**. Total unique suggestions: {len(cleaned_result)}")

