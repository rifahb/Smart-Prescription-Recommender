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
        # Pre-clean: truncate long entries and dedupe
        seen = set()
        cleaned_result = []
        for t in result:
            t = t.strip()
            if len(t) > 120:
                t = t[:117] + "..."
            key = t.lower()
            if key and key not in seen:
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
            if any(w in tl for w in ["medication", "drug", "insulin", "immuno", "antibiotic", "antiviral", "chemotherapy"]):
                categories["Medication"].append(t)
            elif any(w in tl for w in ["surgery", "surgical", "removal", "abortion", "intervention"]):
                categories["Surgery"].append(t)
            elif any(w in tl for w in ["therapy", "speech", "physical"]):
                categories["Therapy"].append(t)
            elif any(w in tl for w in ["rest", "hydration", "supportive", "compresses", "care", "hospitalization"]):
                categories["Supportive Care"].append(t)
            elif any(w in tl for w in ["lifestyle", "exercise", "diet", "avoid", "modification"]):
                categories["Lifestyle"].append(t)
            elif any(w in tl for w in ["diagnose", "imaging", "urine", "culture", "test", "scan"]):
                categories["Diagnosis"].append(t)
            else:
                categories["Other"].append(t)

        # Display limited output
        st.success("✅ **Top Recommended Treatments:**")
        for group, treatments in categories.items():
            if treatments:
                st.subheader(f"🔹 {group}")
                short_list = sorted(treatments)[:5]
                for item in short_list:
                    st.markdown(f"- {item}")
                if len(treatments) > 5:
                    with st.expander(f"See all {len(treatments)} {group.lower()} options"):
                        for item in sorted(treatments[5:]):
                            st.markdown(f"- {item}")
        st.caption(f"🧾 Showing top 5 per category. Total unique suggestions: {len(cleaned_result)}")

    else:
        st.warning("⚠️ No confident recommendation found.")
