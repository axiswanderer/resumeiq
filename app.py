import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from src.parser import extract_text_from_pdf
from src.features import extract_skills

# 1. Load the Trained Model
model = xgb.XGBClassifier()
model.load_model("resume_model.json")

# 2. App Layout
st.title("AI Resume Screener")
st.markdown("Upload a resume and job description to see if they match.")

# 3. Input: Job Description
st.subheader("1. Job Description")
default_jd = "We are looking for a Python Engineer with SQL, Docker, and Machine Learning experience."
jd_text = st.text_area("Paste JD here:", value=default_jd, height=100)

# Extract required skills from JD 
# In a real app, you'd use NER here too.
required_skills = [s.strip() for s in ["Python", "SQL", "Docker", "Machine Learning"] if s in jd_text]
st.write(f"**Required Skills Detected:** {required_skills}")

# 4. Input: Resume Upload
st.subheader("2. Upload Resume (PDF)")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    # Save file temporarily to parse it
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 5. Process the Resume
    resume_text = extract_text_from_pdf(save_path)
    st.text_area("Resume Text Preview:", value=resume_text[:500] + "...", height=100)
    
    # 6. Extract Features (Must match training columns EXACTLY)
    # We reconstruct the dataframe row just like in training
    found_skills = extract_skills(resume_text, required_skills)
    skill_count = len(found_skills)
    text_len = len(resume_text)
    
    # Create the feature row
    input_data = {
        "skill_count": [skill_count],
        "text_len": [text_len]
    }
    # Add the specific skill flags (One-Hot Encoding)
    for skill in ["Python", "Machine Learning", "SQL", "Docker"]: # These must match training order
        input_data[f"has_{skill}"] = [1 if skill in found_skills else 0]
        
    df_input = pd.DataFrame(input_data)
    
    # 7. Prediction
    prob = model.predict_proba(df_input)[0][1] # Probability of "1" (Hired)
    
    # 8. Display Results
    st.divider()
    st.subheader("Start Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Match Score", value=f"{prob * 100:.1f}%")
    
    with col2:
        if prob > 0.7:
            st.success("✅ Recommendation: Interview")
        elif prob > 0.4:
            st.warning("⚠️ Recommendation: Review Manually")
        else:
            st.error("❌ Recommendation: Reject")
            
    # Why
    st.write("### Why this score?")
    st.write(f"- **Skills Found:** {len(found_skills)} of {len(required_skills)}")
    st.write(f"- **Missing Skills:** {list(set(required_skills) - set(found_skills))}")