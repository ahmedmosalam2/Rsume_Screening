import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import re
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("📥 Reading datasets...")
df_resume = pd.read_csv("/mnt/d/Rsume_Screening/data/resume_features.csv")
df_job = pd.read_csv("/mnt/d/Rsume_Screening/data/job_descriptions_clean_half.csv")
df_job = df_job.head(500_000)  # تقدر تعدل الرقم حسب قدرتك
print(f"Resumes: {len(df_resume)}, Jobs: {len(df_job)}")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), " ", text)
    text = re.sub(r"[^x00-x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("🧹 Cleaning text columns...")
df_resume['cleaned_resume'] = df_resume['cleaned_resume'].fillna('').apply(clean_text)
df_job['cleaned_description'] = df_job['cleaned_description'].fillna('').apply(clean_text)
df_job['cleaned_skills'] = df_job['cleaned_skills'].fillna('').apply(clean_text)
df_job['full_text'] = df_job['cleaned_description'] + " " + df_job['cleaned_skills']
print("✅ Text cleaning done")

print("🤖 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("✅ Model loaded")

resume_batch_size = 64
job_batch_size = 16
top_k = 5
results = []

print("⚡ Encoding resume embeddings...")
resume_embeddings = model.encode(df_resume['cleaned_resume'].tolist(),
                                 batch_size=resume_batch_size,
                                 convert_to_tensor=True,
                                 device=device)
print("✅ Resume embeddings done")

print("⚡ Processing similarity in batches...")
num_job_batches = (len(df_job) + job_batch_size - 1) // job_batch_size

for i in range(num_job_batches):
    start_idx = i * job_batch_size
    end_idx = min((i + 1) * job_batch_size, len(df_job))
    job_batch = df_job.iloc[start_idx:end_idx]
    job_embeddings = model.encode(job_batch['full_text'].tolist(),
                                  batch_size=len(job_batch),
                                  convert_to_tensor=True,
                                  device=device)

    for j, job_emb in enumerate(job_embeddings):
        sim = F.cosine_similarity(resume_embeddings, job_emb.unsqueeze(0), dim=1)
        k = min(top_k, len(sim))
        top_values, top_indices = torch.topk(sim, k=k)

        for rank, idx in enumerate(top_indices):
            resume_idx = idx.item()
            results.append({
                "Job Title": job_batch.iloc[j]['Job Title'],
                "Resume Index": int(resume_idx),
                "Resume Text": df_resume.iloc[resume_idx]['cleaned_resume'][:200],
                "Similarity": top_values[rank].item(),
                "Rank": rank + 1
            })
    print(f"Processed job batch {i+1}/{num_job_batches}")

os.makedirs("/mnt/d/Rsume_Screening/data/results", exist_ok=True)
df_results = pd.DataFrame(results)
df_results.to_csv("/mnt/d/Rsume_Screening/data/results/resume_matching_results.csv", index=False)
print("✅ Done, results saved at data/results/resume_matching_results.csv")
