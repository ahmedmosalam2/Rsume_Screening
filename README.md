# 📝 AI-Powered Resume Screening System  

An intelligent system that helps recruiters automatically match resumes with job descriptions using **NLP** and **Machine Learning**.  
This tool ranks candidates based on similarity scores and provides a clean interactive **Streamlit dashboard** for recruiters.  

---

## 🚀 Features  
- Upload and process job descriptions & resumes.  
- Extract features using **TF-IDF** (or embeddings).  
- Compute **similarity scores** between resumes and jobs.  
- Display **Top N candidates** for each job title.  
- Interactive dashboard built with **Streamlit**.  
- Visualization of candidate match scores.  

---

## 📂 Project Structure  
```bash
Resume-Screening/
│── data/                # Raw & processed data (resumes + jobs)
│   ├── resume_features.csv
│   ├── job_descriptions.csv
│   └── results/
│       └── resume_matching_results.csv
│
│── models/              # Saved models or vectorizers
│
│── src/                 
│   ├── preprocessing.py # Data cleaning & feature extraction
│   ├── matching.py      # Similarity computation
│   ├── inference.py     # Streamlit app (UI)
│
│── README.md            # Project documentation
│── requirements.txt     # Dependencies
