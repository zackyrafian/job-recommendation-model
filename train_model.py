import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./assets/clean_data.csv")
df.rename(columns={"r": "Skills", "title": "Job Title"}, inplace=True)
df["Skills"] = df["Skills"].astype(str).str.lower().str.strip()
df = df.dropna(subset=["Skills", "Job Title"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Skills"])
y = df["Job Title"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
