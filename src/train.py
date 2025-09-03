import streamlit as st
import pandas as pd
import plotly.express as px

# ---- Page Config ----
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📝",
    layout="wide"
)

# ---- Load Data ----
@st.cache_data
def load_data():
    return pd.read_csv("/mnt/d/Rsume_Screening/data/results/resume_matching_results.csv")

df = load_data()

# ---- Header ----
st.title("📝 AI-Powered Resume Screening System")
st.markdown(
    """
    This tool helps recruiters quickly identify the most suitable candidates for a given job title.  
    Enter the **job title** and select how many top candidates you’d like to view.
    """
)

# ---- Sidebar Filters ----
st.sidebar.header("⚙️ Filters")
job_input = st.sidebar.text_input("Enter Job Title (e.g., Data Scientist):")
top_n = st.sidebar.slider("Number of top candidates:", min_value=1, max_value=20, value=5)

# ---- Main Logic ----
if job_input:
    job_df = df[df['Job Title'].str.lower() == job_input.lower()]
    
    if job_df.empty:
        st.warning(f"No candidates found for **{job_input}**.")
    else:
        # Select Top N Candidates
        top_candidates = job_df.nlargest(top_n, 'Similarity')
        top_candidates = top_candidates[['Resume Index', 'Resume Text', 'Similarity']]
        
        # ---- Summary Metrics ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Job Title", job_input)
        col2.metric("Total Applicants", len(job_df))
        col3.metric("Top Candidates Shown", top_n)
        
        st.markdown("---")
        st.subheader(f"Top {top_n} Candidates for **{job_input}**")
        
        # ---- Candidate Table ----
        st.dataframe(top_candidates.style.format({"Similarity": "{:.2f}"}))
        
        # ---- Visualization ----
        fig = px.bar(
            top_candidates,
            x="Resume Index",
            y="Similarity",
            text="Similarity",
            title="Candidate Similarity Scores",
            labels={"Resume Index": "Candidate ID", "Similarity": "Match Score"},
            color="Similarity",
            color_continuous_scale="Blues"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please enter a job title from the sidebar to start screening.")
