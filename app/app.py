import streamlit as st
import sys
import os

# Add parent path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict

# Streamlit page setup
st.set_page_config(page_title="Smart Prescription Recommender", page_icon="💊")
st.title("💊 Smart Prescription Recommender")
st.markdown("Enter patient symptoms and traits below to receive AI-suggested treatment classes.")

# Input fields
symptom_input = st.text_input("📝 Enter symptoms (comma-separated):", "itchy skin, small blisters")
contagious = st.checkbox("Contagious?")
chronic = st.checkbox("Chronic?")

# On button click
if st.button("🔍 Recommend Treatments"):
    result = predict(symptom_input, contagious, chronic)

    if result:
        # Pre-clean: truncate long entries and dedupe by lowercase string
        seen = set()
        cleaned_result = []
        for t in result:
            t = t.strip()
            if len(t) > 140:
                t = t[:137] + "..."
            key = t.lower()
            if key not in seen:
                seen.add(key)
                cleaned_result.append(t)

        # Grouping logic
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
            if any(word in tl for word in ["medication", "drug", "insulin", "immuno", "antibiotic", "antiviral", "chemotherapy"]):
                categories["Medication"].append(t)
            elif any(word in tl for word in ["surgery", "surgical", "removal", "abortion", "intervention"]):
                categories["Surgery"].append(t)
            elif any(word in tl for word in ["therapy", "speech", "physical"]):
                categories["Therapy"].append(t)
            elif any(word in tl for word in ["rest", "hydration", "supportive", "compresses", "care", "pain relief", "hospitalization"]):
                categories["Supportive Care"].append(t)
            elif any(word in tl for word in ["lifestyle", "exercise", "diet", "avoid", "modification"]):
                categories["Lifestyle"].append(t)
            elif any(word in tl for word in ["diagnose", "imaging", "urine", "culture", "test", "scan"]):
                categories["Diagnosis"].append(t)
            else:
                categories["Other"].append(t)

        # Display grouped output
        st.success("✅ **Recommended Treatments:**")

        for group, treatments in categories.items():
            if treatments:
                st.subheader(f"🔹 {group}")
                for item in sorted(treatments):
                    st.markdown(f"- {item}")

        st.caption(f"🧾 Total unique suggestions: {len(cleaned_result)}")

    else:
        st.warning("⚠️ No confident recommendation found.")
