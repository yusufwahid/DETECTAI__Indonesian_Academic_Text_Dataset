# -*- coding: utf-8 -*-
"""
Demo skripsi: Detektor teks generatif AI berbahasa Indonesia
Skema fitur  : TF-IDF, Stylometry (21 fitur), TF-IDF + Stylometry
Algoritma    : Logistic Regression, SVM (LinearSVC), Random Forest, XGBoost
Pembersihan, fitur, dan hiperparameter sama dengan notebook penelitian.
"""
import os
import re
import string

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from xgboost import XGBClassifier

st.set_page_config(page_title="Detektor Teks AI Berbahasa Indonesia", page_icon="🔎", layout="wide")

# ------------------------------------------------------------------ konfigurasi
CSV_PATH = "DETECTAI__Indonesian_Academic_Text_Dataset.csv"   # opsional, jika ada dipakai lebih dulu
HF_DATASET = "shouwiku/detectai-indonesian"
TFIDF_MAX_FEATURES = 3000
SCHEMES = ["TF-IDF", "Stylometry", "TF-IDF+Stylometry"]
ALGOS = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]

# Hasil skripsi (rata-rata 5 fold x 3 seed, validasi per topik) - Tabel 4.2
RESULTS = {
    ("TF-IDF", "Logistic Regression"): (0.9565, 0.9734, 0.9384, 0.9553, 0.9937),
    ("TF-IDF", "SVM"): (0.9624, 0.9842, 0.9393, 0.9611, 0.9958),
    ("TF-IDF", "Random Forest"): (0.9484, 0.9670, 0.9286, 0.9471, 0.9882),
    ("TF-IDF", "XGBoost"): (0.9443, 0.9667, 0.9203, 0.9428, 0.9862),
    ("Stylometry", "Logistic Regression"): (0.9267, 0.9235, 0.9304, 0.9264, 0.9797),
    ("Stylometry", "SVM"): (0.9287, 0.9256, 0.9323, 0.9284, 0.9802),
    ("Stylometry", "Random Forest"): (0.9347, 0.9491, 0.9187, 0.9329, 0.9825),
    ("Stylometry", "XGBoost"): (0.9397, 0.9474, 0.9309, 0.9384, 0.9834),
    ("TF-IDF+Stylometry", "Logistic Regression"): (0.9513, 0.9514, 0.9512, 0.9511, 0.9901),
    ("TF-IDF+Stylometry", "SVM"): (0.9768, 0.9808, 0.9729, 0.9767, 0.9969),
    ("TF-IDF+Stylometry", "Random Forest"): (0.9670, 0.9830, 0.9503, 0.9661, 0.9951),
    ("TF-IDF+Stylometry", "XGBoost"): (0.9648, 0.9753, 0.9536, 0.9642, 0.9942),
}
# Akurasi validasi per topik vs acak (seed 42) - Tabel 4.4
GAP = {
    ("TF-IDF", "Logistic Regression"): (0.9558, 0.9759), ("TF-IDF", "SVM"): (0.9588, 0.9848),
    ("TF-IDF", "Random Forest"): (0.9467, 0.9656), ("TF-IDF", "XGBoost"): (0.9396, 0.9571),
    ("Stylometry", "Logistic Regression"): (0.9258, 0.9294), ("Stylometry", "SVM"): (0.9268, 0.9285),
    ("Stylometry", "Random Forest"): (0.9356, 0.9402), ("Stylometry", "XGBoost"): (0.9382, 0.9460),
    ("TF-IDF+Stylometry", "Logistic Regression"): (0.9497, 0.9562), ("TF-IDF+Stylometry", "SVM"): (0.9768, 0.9866),
    ("TF-IDF+Stylometry", "Random Forest"): (0.9663, 0.9795), ("TF-IDF+Stylometry", "XGBoost"): (0.9630, 0.9723),
}

# ------------------------------------------------------------------ pembersihan teks (sama dengan notebook)
def trim_to_last_sentence(t, min_words=15):
    m = list(re.finditer(r"[.!?](?=\s|$)", t))
    if not m:
        return t
    cut = t[: m[-1].end()]
    return cut if len(cut.split()) >= min_words else t


def clean_text(t):
    t = str(t)
    t = re.sub(r"^\s*\*\*[^\n]*?\*\*\s*\n+", "", t)
    t = re.sub(r"\s*={2,}[^=\n]*={2,}\s*", " ", t)
    t = re.sub(r"={2,}", " ", t)
    t = re.sub(r"[*#`]+", "", t)
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return trim_to_last_sentence(t)


# ------------------------------------------------------------------ 21 fitur stylometri (sama dengan notebook)
STOPWORDS_ID = set(StopWordRemoverFactory().get_stop_words())
PUNCT = set(string.punctuation)


def split_words(text):
    return re.findall(r"\b[\wÀ-ÿ]+\b", text.lower(), flags=re.UNICODE)


def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def safe_ratio(a, b):
    return float(a) / float(b) if b else 0.0


def extract_stylometry(texts):
    rows = []
    for text in texts:
        text = str(text)
        words = split_words(text)
        sentences = split_sentences(text)
        wl = np.array([len(w) for w in words], dtype=float)
        sl = np.array([len(split_words(s)) for s in sentences], dtype=float)
        n_chars, n_words, n_sent = len(text), len(words), len(sentences)
        uniq = set(words)
        hapax = sum(1 for w in uniq if words.count(w) == 1)
        alpha = sum(ch.isalpha() for ch in text)
        rows.append({
            "char_count": n_chars, "word_count": n_words, "sentence_count": n_sent,
            "avg_word_length": wl.mean() if n_words else 0.0,
            "std_word_length": wl.std() if n_words else 0.0,
            "avg_sentence_length": sl.mean() if n_sent else 0.0,
            "std_sentence_length": sl.std() if n_sent else 0.0,
            "type_token_ratio": safe_ratio(len(uniq), n_words),
            "hapax_ratio": safe_ratio(hapax, n_words),
            "stopword_ratio": safe_ratio(sum(1 for w in words if w in STOPWORDS_ID), n_words),
            "long_word_ratio": safe_ratio(sum(1 for w in words if len(w) >= 8), n_words),
            "punctuation_ratio": safe_ratio(sum(ch in PUNCT for ch in text), n_chars),
            "digit_ratio": safe_ratio(sum(ch.isdigit() for ch in text), n_chars),
            "uppercase_ratio": safe_ratio(sum(ch.isupper() for ch in text), alpha),
            "whitespace_ratio": safe_ratio(text.count(" "), n_chars),
            "comma_ratio": safe_ratio(text.count(","), n_chars),
            "semicolon_ratio": safe_ratio(text.count(";"), n_chars),
            "colon_ratio": safe_ratio(text.count(":"), n_chars),
            "question_ratio": safe_ratio(text.count("?"), n_chars),
            "exclamation_ratio": safe_ratio(text.count("!"), n_chars),
            "parenthesis_ratio": safe_ratio(text.count("(") + text.count(")"), n_chars),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ data dan model
@st.cache_data(show_spinner="Memuat dan membersihkan dataset DETECTAI...")
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        sumber = "CSV lokal"
    else:
        from datasets import load_dataset
        df = load_dataset(HF_DATASET)["train"].to_pandas()
        sumber = "Hugging Face"
    df["clean"] = df["text"].apply(clean_text)
    df["n_words"] = df["clean"].str.split().str.len()
    df = df[df["n_words"] >= 15].drop_duplicates(subset="clean").reset_index(drop=True)
    style_df = extract_stylometry(df["clean"].tolist())
    return df, style_df, sumber


def make_algo(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    if name == "SVM":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             eval_metric="logloss", n_jobs=-1, verbosity=0)
    raise ValueError(name)


@st.cache_resource(show_spinner="Melatih model (hanya sekali per kombinasi)...")
def train_model(scheme, algo):
    df, style_df, _ = load_data()
    parts, vec, sc = [], None, None
    if "TF-IDF" in scheme:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
                              max_features=TFIDF_MAX_FEATURES)
        parts.append(vec.fit_transform(df["clean"]))
    if "Stylometry" in scheme:
        sc = StandardScaler().fit(style_df.values)
        parts.append(csr_matrix(sc.transform(style_df.values)))
    X = parts[0] if len(parts) == 1 else hstack(parts).tocsr()
    model = make_algo(algo).fit(X, df["label"].values)
    return {"model": model, "vec": vec, "sc": sc}


def predict(bundle, raw_text):
    clean = clean_text(raw_text)
    feats = extract_stylometry([clean])
    parts = []
    if bundle["vec"] is not None:
        parts.append(bundle["vec"].transform([clean]))
    if bundle["sc"] is not None:
        parts.append(csr_matrix(bundle["sc"].transform(feats.values)))
    X = parts[0] if len(parts) == 1 else hstack(parts).tocsr()
    m = bundle["model"]
    if hasattr(m, "predict_proba"):
        p_ai = float(m.predict_proba(X)[0, 1])
        kind = "prob"
    else:  # LinearSVC: skor jarak ke bidang pemisah, diubah ke 0-1 hanya untuk tampilan
        d = float(m.decision_function(X)[0])
        p_ai = float(1 / (1 + np.exp(-d)))
        kind = "margin"
    return clean, feats, p_ai, kind


# ------------------------------------------------------------------ antarmuka
st.title("Detektor Teks Generatif AI Berbahasa Indonesia")
st.caption("Demo skripsi: Analisis Robustness TF-IDF dan Stylometry pada Algoritma Machine Learning "
           "untuk Deteksi Teks Generatif AI Berbahasa Indonesia")

with st.sidebar:
    st.header("Pengaturan model")
    scheme = st.selectbox("Skema fitur", SCHEMES, index=2)
    algo = st.selectbox("Algoritma", ALGOS, index=1)
    acc, prec, rec, f1, auc = RESULTS[(scheme, algo)]
    st.markdown("**Kinerja pada skripsi** (validasi per topik, rata-rata 5 fold × 3 seed)")
    c1, c2 = st.columns(2)
    c1.metric("Akurasi", f"{acc:.4f}")
    c2.metric("F1-score", f"{f1:.4f}")
    c1.metric("Presisi", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    st.metric("AUC", f"{auc:.4f}")
    st.info("Model dilatih ulang pada seluruh dataset untuk demo ini. Angka di atas adalah estimasi "
            "dari validasi silang, bukan akurasi pada teks yang Anda masukkan.")

tab_detect, tab_result, tab_about = st.tabs(["Deteksi teks", "Hasil penelitian", "Tentang dan batasan"])

# ------------------------------- tab 1: deteksi
with tab_detect:
    df, style_df, sumber = load_data()
    bundle = train_model(scheme, algo)

    def ambil_contoh():
        row = df.sample(1).iloc[0]
        st.session_state["input_text"] = row["clean"]
        st.session_state["label_asli"] = "Teks AI" if int(row["label"]) == 1 else "Teks manusia (Wikipedia)"

    st.write("Tempel teks berbahasa Indonesia (15 sampai sekitar 200 kata), lalu tekan **Deteksi**.")
    st.text_area("Teks yang akan diperiksa", key="input_text", height=220,
                 placeholder="Tempel teks di sini...")
    b1, b2, _ = st.columns([1, 2, 4])
    run = b1.button("Deteksi", type="primary")
    b2.button("Ambil contoh acak dari dataset", on_click=ambil_contoh)

    if st.session_state.get("label_asli"):
        st.caption(f"Label asli contoh dari dataset (jangan dilihat dulu jika ingin menguji): "
                   f"{st.session_state['label_asli']}")

    if run:
        text = st.session_state.get("input_text", "").strip()
        n_words = len(text.split())
        if n_words < 15:
            st.warning(f"Teks terlalu pendek ({n_words} kata). Model dilatih pada teks minimal 15 kata.")
        else:
            if n_words > 200:
                st.warning("Teks lebih panjang dari data latih (maks. sekitar 200 kata). "
                           "Hasil kurang dapat diandalkan; coba periksa per bagian.")
            clean, feats, p_ai, kind = predict(bundle, text)
            label = "Teks AI" if p_ai >= 0.5 else "Teks manusia"
            st.subheader(f"Prediksi: {label}")
            st.progress(min(max(p_ai, 0.0), 1.0), text=f"Skor AI: {p_ai:.2%}")
            if kind == "margin":
                st.caption("SVM tidak menghasilkan peluang. Skor ini hanya jarak ke bidang pemisah yang "
                           "diubah ke rentang 0 sampai 1, jadi bukan peluang terkalibrasi.")
            else:
                st.caption("Skor adalah peluang kelas AI menurut model, bukan kepastian.")
            st.warning("Gunakan hasil ini sebagai pertimbangan pendukung, bukan keputusan tunggal, "
                       "terutama untuk penilaian akademik.")

            st.markdown("**Fitur stylometri teks ini dibandingkan rata-rata dataset**")
            means = style_df.assign(label=df["label"]).groupby("label").mean().T
            comp = pd.DataFrame({
                "Teks ini": feats.iloc[0],
                "Rata-rata manusia": means[0],
                "Rata-rata AI": means[1],
            }).round(4)
            st.dataframe(comp)
            with st.expander("Teks setelah dibersihkan (yang dimasukkan ke model)"):
                st.write(clean)
    st.caption(f"Sumber data latih: {sumber} | jumlah teks setelah pembersihan: {len(df):,}".replace(",", "."))

# ------------------------------- tab 2: hasil penelitian
with tab_result:
    st.subheader("Kinerja 12 kombinasi skema fitur dan algoritma")
    rows = [dict(Skema=s, Algoritma=a, Akurasi=v[0], Presisi=v[1], Recall=v[2], F1=v[3], AUC=v[4])
            for (s, a), v in RESULTS.items()]
    res = pd.DataFrame(rows)
    st.dataframe(res, hide_index=True)
    st.markdown("**F1-score per kombinasi**")
    st.bar_chart(res.pivot(index="Algoritma", columns="Skema", values="F1").loc[ALGOS, SCHEMES])
    st.success("Kombinasi terbaik: SVM + TF-IDF + Stylometry (F1 0,9767, AUC 0,9969, koefisien variasi 0,0116).")

    st.subheader("Seberapa optimistis validasi acak dibanding validasi per topik")
    gap = pd.DataFrame([dict(Skema=s, Algoritma=a, **{"Per topik": g[0], "Acak": g[1], "Selisih": round(g[1] - g[0], 4)})
                        for (s, a), g in GAP.items()])
    st.dataframe(gap, hide_index=True)
    ringkas = gap.groupby("Skema", sort=False)["Selisih"].mean().round(4).rename("Rata-rata selisih")
    st.dataframe(ringkas.to_frame())
    st.caption("Selisih terbesar pada TF-IDF (berbasis isi kata), terkecil pada Stylometry (berbasis gaya tulis). "
               "Perbandingan memakai satu seed (42).")

# ------------------------------- tab 3: tentang
with tab_about:
    st.markdown("""
**Tentang aplikasi.** Aplikasi ini mendemonstrasikan hasil skripsi pada deteksi teks berbahasa Indonesia
sebagai tulisan manusia atau hasil AI. Pembersihan teks, 21 fitur stylometri, konfigurasi TF-IDF
(unigram dan bigram, maksimum 3.000 fitur), dan hiperparameter algoritma sama dengan notebook penelitian.

**Batasan yang perlu dipahami**
1. Teks manusia pada data latih berasal dari Wikipedia dan teks AI dari keluarga model Gemma, sehingga
   kinerja pada gaya tulis lain (opini, media sosial, esai mahasiswa) atau generator lain belum teruji.
2. Data latih berukuran sekitar 16 sampai 199 kata. Teks yang jauh lebih pendek atau panjang kurang andal.
3. Ketahanan terhadap parafrase tidak diuji pada penelitian ini.
4. Hasil berupa perkiraan statistik. Teks manusia dapat salah ditandai sebagai AI dan sebaliknya.
   Jangan dipakai sebagai satu-satunya dasar sanksi akademik.
""")
