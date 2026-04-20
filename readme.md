---

# AI Resume Screening System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=Streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-FLAML-green?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge\&logo=spacy)

> **Automating the recruitment funnel with NLP and Machine Learning.**

This project is an end-to-end **AI-powered Applicant Tracking System (ATS)** prototype. It automates the initial screening process by parsing PDF resumes, extracting key technical skills using Natural Language Processing (NLP), and ranking candidates against job descriptions using an **XGBoost** classifier.

---

## Demo

### Interactive Dashboard

The system features a user-friendly web interface built with **Streamlit** for real-time scoring and feedback.

![Dashboard Preview](https://github.com/axiswanderer/resumeiq/blob/main/assets/dashboard_demo.png)

---

## Architecture & Workflow

The system follows a standard MLOps pipeline structure:

1. **Ingestion:** Raw PDF resumes are parsed using `PyMuPDF` to extract unstructured text.
2. **Feature Engineering:**

   * **NER (Named Entity Recognition):** `spaCy` extracts skills, degrees, and experience.
   * **Vectorization:** Resumes are converted into numerical feature vectors (Skill Match Count, Text Length, Keyword Density).
3. **Scoring Model:** A trained **XGBoost Classifier** predicts the probability of a candidate being "Hired" based on historical/synthetic data patterns.
4. **Visualization:** The results are displayed via a Streamlit dashboard, providing immediate feedback on missing skills.

---

## Tech Stack

* **Core Logic:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **NLP:** spaCy (Entity Extraction)
* **Data Processing:** Pandas, NumPy
* **PDF Parsing:** PyMuPDF (Fitz)
* **Visualization/UI:** Streamlit

---

## Setup & Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/axiswanderer/resumeiq.git
cd resumeiq
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost spacy reportlab faker pymupdf
```

### 4. (Optional) Generate Synthetic Data

If you don't have real resumes, generate a synthetic dataset for training:

```bash
python generate_data.py
```

### 5. Train the Model

Train the XGBoost ranker on the data:

```bash
python train_model.py
```

*Output: `Model saved to resume_model.json`*

---

## Usage

Launch the dashboard to test the model:

```bash
streamlit run app.py
```

1. Open your browser to the local URL provided (usually `http://localhost:8501`).
2. Paste a **Job Description** (or use the default).
3. **Upload a PDF Resume**.
4. View the **Match Score** and missing skills analysis.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs or feature enhancements.

---

## License

This project is licensed under the MIT License.

---

**Made with love by @axiswanderer**
