import pandas as pd
import streamlit as st
import joblib

# ========== Load Model ==========
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ========== Load Data untuk Referensi ==========
@st.cache_data
def load_data():
    df = pd.read_csv("./assets/clean_data.csv")
    df.rename(columns={"r": "Skills", "title": "Job Title"}, inplace=True)
    df["Skills"] = df["Skills"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["Skills", "Job Title"])
    return df

df = load_data()

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
