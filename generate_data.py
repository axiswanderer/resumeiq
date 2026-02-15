import os
import random
import pandas as pd
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker()

# Configuration
NUM_CANDIDATES = 50
SKILL_POOL = ["Python", "SQL", "Java", "Machine Learning", "Docker", "AWS", "React", "Node.js", "Kubernetes", "Excel"]

# Setup paths
RESUME_DIR = "data/raw/resumes"
JD_FILE = "data/raw/job_descriptions/software_engineer.txt"
os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs("data/raw/job_descriptions", exist_ok=True)

# 1. Target Job Description
job_required_skills = ["Python", "Machine Learning", "SQL", "Docker"]
job_description = f"""
Job Title: AI Engineer
Description: We are looking for a candidate with strong experience in {', '.join(job_required_skills)}.
The ideal candidate will build predictive models and deploy them to cloud infrastructure.
"""
with open(JD_FILE, "w") as f:
    f.write(job_description)

print(f"Generated Job Description with required skills: {job_required_skills}")

# 2. Synthetic Resumes & Labels
data = []

for i in range(NUM_CANDIDATES):
    # Randomly assign skills to candidate
    num_skills = random.randint(2, 6)
    candidate_skills = random.sample(SKILL_POOL, num_skills)
    
    # Calculate "Ground Truth" Score (Simple overlap logic)
    # If candidate has a required skill, they get points.
    match_count = sum(1 for skill in candidate_skills if skill in job_required_skills)
    score = match_count / len(job_required_skills)
    is_hired = 1 if score >= 0.5 else 0  # Hire if matched > 50%
    
    # Create PDF content
    c_name = fake.name()
    c_email = fake.email()
    c_text = f"""
    Name: {c_name}
    Email: {c_email}
    
    Summary:
    Passionate developer interested in technology and data.
    
    Skills:
    {', '.join(candidate_skills)}
    
    Experience:
    Worked at {fake.company()} as a Software Engineer.
    """
    
    # Save as PDF
    file_name = f"resume_{i}.pdf"
    file_path = os.path.join(RESUME_DIR, file_name)
    
    c = canvas.Canvas(file_path, pagesize=letter)
    text_obj = c.beginText(40, 750)
    for line in c_text.split('\n'):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()
    
    # Log data for training later
    data.append({
        "resume_id": file_name,
        "skills": ", ".join(candidate_skills),
        "score": score,
        "hired_label": is_hired
    })

# Save the Ground Truth CSV
df = pd.DataFrame(data)
df.to_csv("data/processed_data.csv", index=False)
print(f"Generated {NUM_CANDIDATES} resumes in {RESUME_DIR}")
print("Saved ground truth labels to data/processed_data.csv")