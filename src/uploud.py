import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline
import re

# --- Page Config ---
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #2E86C1;
        }
        .sub-title {
            font-size: 20px;
            font-weight: bold;
            color: #117A65;
        }
        .stProgress > div > div > div > div {
            background-color: #2E86C1;
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            margin: 4px;
            background-color: #D6EAF8;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 500;
            color: #154360;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load CSV data ---
df_resumes = pd.read_csv("/mnt/d/data/results/resume_matching_results.csv")
df_jobs = pd.read_csv("/mnt/d/data/cleaned/job_descriptions_clean_half.csv")

# --- NER pipeline ---
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# --- Functions ---
def extract_text(file):
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages])
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    return ""

def clean_text(text):
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_entities(text):
    entities = ner_pipeline(text)
    skills = [e['word'] for e in entities if e['entity_group'] == "SKILL"]
    experience = [e['word'] for e in entities if e['entity_group'] == "EXPERIENCE"]
    projects = [e['word'] for e in entities if e['entity_group'] == "ORG"]
    return list(set(skills)), list(set(experience)), list(set(projects))

def compute_score(skills, experience, projects, job_category):
    job_data = df_jobs[df_jobs['Role'].str.lower() == job_category.lower()]
    required_skills = []
    for s in job_data['skills']:
        if pd.notna(s):
            required_skills.extend([x.strip().lower() for x in s.split(',')])
    skill_match = len(set(skills).intersection(set(required_skills))) / (len(required_skills)+1e-5)
    exp_match = min(len(experience)/5, 1)
    proj_match = min(len(projects)/3, 1)
    score = 0.5 * skill_match + 0.3 * exp_match + 0.2 * proj_match
    return score * 100

# --- Sidebar ---
st.sidebar.header("📂 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Resume (.pdf, .docx, .txt)")
job_category_input = st.sidebar.text_input("Job Category", "AI Engineering")

# --- Main Title ---
st.markdown('<p class="main-title">📄 AI-Powered Resume Screening</p>', unsafe_allow_html=True)

if uploaded_file and job_category_input:
    text = extract_text(uploaded_file)
    cleaned_text = clean_text(text)
    skills, experience, projects = extract_entities(cleaned_text)
    score = compute_score(skills, experience, projects, job_category_input)

    # --- Match Score Section ---
    st.subheader("📊 Match Score")
    st.progress(int(score))
    st.metric("Overall Score", f"{score:.2f}%")

    # --- Skills ---
    st.markdown('<p class="sub-title">🛠️ Skills Extracted</p>', unsafe_allow_html=True)
    st.markdown("".join([f"<span class='badge'>{s}</span>" for s in skills]), unsafe_allow_html=True)

    # --- Experience ---
    st.markdown('<p class="sub-title">💼 Experience Extracted</p>', unsafe_allow_html=True)
    st.write(", ".join(experience) if experience else "No experiences detected")

    # --- Projects ---
    st.markdown('<p class="sub-title">📁 Projects Extracted</p>', unsafe_allow_html=True)
    st.write(", ".join(projects) if projects else "No projects detected")

    # --- Result Message ---
    if score >= 80:
        st.success("✅ Excellent Match! Candidate is highly suitable.")
    elif score >= 50:
        st.warning("⚠️ Moderate Match. Candidate might need some improvements.")
    else:
        st.error("❌ Weak Match. Candidate does not meet most requirements.")
else:
    st.info("👆 Upload a resume and enter job category to start screening.")
