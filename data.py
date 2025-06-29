import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ========== Load & Clean Data ==========
@st.cache_data
def load_data():
    df = pd.read_csv("./assets/clean_data.csv")

    # Rename biar seragam
    df.rename(columns={
        "r": "Skills",
        "title": "Job Title"
    }, inplace=True)

    # Bersihkan data
    df["Skills"] = df["Skills"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["Skills", "Job Title"])
    
    return df

df = load_data()

# ========== Train Model ==========
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Skills"])
y = df["Job Title"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Job Recommender", page_icon="🧠")
st.title("🧠 Rekomendasi Job Role Berdasarkan Skill")

with st.form("recommend_form"):
    skill_input = st.text_input("🛠️ Masukkan skill kamu (pisahkan koma):", "java, android")
    top_n = st.slider("🔢 Jumlah rekomendasi:", 1, 5, 3)
    submitted = st.form_submit_button("🔍 Cari Rekomendasi")

if submitted:
    if not skill_input.strip():
        st.error("❗ Harap masukkan skill terlebih dahulu.")
    else:
        skill_input = skill_input.strip().lower()
        input_vec = vectorizer.transform([skill_input])
        proba = model.predict_proba(input_vec)[0]
        classes = model.classes_

        results = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:top_n]

        st.subheader("🎯 Rekomendasi Teratas:")
        for i, (job, score) in enumerate(results):
            st.markdown(f"### {i+1}. 🧩 {job} — **{round(score * 100, 2)}%** keyakinan")
            job_url = f"https://id.jobstreet.com/id/{job.replace(' ', '-')}-jobs"
            st.markdown(f"[🌐 Cari lowongan {job} di JobStreet]({job_url})", unsafe_allow_html=True)
            st.markdown("---")
