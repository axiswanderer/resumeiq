import spacy
import pandas as pd
import os
from src.parser import extract_text_from_pdf

# Load English language model (make sure to run: python -m spacy download en_core_web_sm)
# If you haven't, run that command in terminal first!
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_skills(text, skill_list):
    """
    Simple keyword matching. 
    In a real app, this would use Entity Recognition (NER).
    """
    text_lower = text.lower()
    found_skills = [skill for skill in skill_list if skill.lower() in text_lower]
    return found_skills

def process_resumes(resume_dir, required_skills):
    """
    Loops through all PDFs and creates a DataFrame of features.
    """
    rows = []
    files = os.listdir(resume_dir)
    
    for f in files:
        if f.endswith(".pdf"):
            path = os.path.join(resume_dir, f)
            text = extract_text_from_pdf(path)
            
            # Feature 1: Which skills are present?
            found = extract_skills(text, required_skills)
            
            # Feature 2: Count of matched skills
            skill_count = len(found)
            
            # Feature 3: Text length (heuristic)
            text_len = len(text)
            
            rows.append({
                "resume_id": f,
                "text": text,
                "skill_count": skill_count,
                "text_len": text_len,
                # Create One-Hot Encoding for each required skill
                **{f"has_{s}": (1 if s in found else 0) for s in required_skills}
            })
            
    return pd.DataFrame(rows)