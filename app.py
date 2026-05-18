import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG ──────────────────────────────────────────────────────────────────
K_NEIGHBORS     = 3
LSA_COMPONENTS  = 200   # ~30% reduksi dimensi (sesuai thesis)
THRESHOLD_C     = 70    # T = (c/100) * max(sim_lsa)
MIN_SIMILARITY  = 0.10

STOPWORDS_PATH  = "data/term_stopwords_id.txt"
DATASET_PATH    = "data/datasetfix.csv"

INTENT_LABELS = {
    "administratif":     "Administratif",
    "jadwal":            "Jadwal",
    "fasilitas":         "Fasilitas",
    "kegiatan":          "Kegiatan",
    "peraturan akademik":"Peraturan Akademik",
    "layanan akademik":  "Layanan Akademik",
    "pelaksana":         "Pelaksana Akademik",
    "greetings":         "Sapaan",
    "dialog":            "Dialog",
    "bot":               "Info Bot",
}

# ─── PREPROCESSING ───────────────────────────────────────────────────────────
def preprocess(text: str, stemmer, stopwords: set) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 1]
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# ─── LOAD SEMUA RESOURCE (di-cache, hanya jalan sekali) ─────────────────────
@st.cache_resource(show_spinner=False)
def load_all():
    # 1. Stemmer (PySastrawi)
    stemmer = None
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stemmer = StemmerFactory().createStemmer()
    except Exception:
        pass  # fallback tanpa stemmer

    # 2. Stopwords
    stopwords: set = set()
    try:
        with open(STOPWORDS_PATH, encoding="utf-8") as f:
            stopwords = set(f.read().split())
    except Exception:
        pass

    # 3. Dataset (pertanyaan, jawaban, kategori)
    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        quotechar='"',
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = ["pertanyaan", "jawaban", "kategori"]
    df = df.dropna(subset=["pertanyaan", "jawaban", "kategori"]).reset_index(drop=True)
    df["pertanyaan"] = df["pertanyaan"].astype(str).str.strip()
    df["jawaban"]    = df["jawaban"].astype(str).str.strip()
    df["kategori"]   = df["kategori"].astype(str).str.strip().str.lower()

    # 4. Preprocessing semua pertanyaan training
    processed = [preprocess(q, stemmer, stopwords) for q in df["pertanyaan"]]

    # 5. TF-IDF
    vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(processed)

    # 6. LSA via TruncatedSVD
    n_comp = min(LSA_COMPONENTS, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    lsa_matrix = normalize(svd.fit_transform(tfidf_matrix))

    # 7. KNN Classifier (cosine similarity)
    knn = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        metric="cosine",
        algorithm="brute",
    )
    knn.fit(lsa_matrix, df["kategori"].values)

    return {
        "df":         df,
        "stemmer":    stemmer,
        "stopwords":  stopwords,
        "vectorizer": vectorizer,
        "svd":        svd,
        "lsa_matrix": lsa_matrix,
        "knn":        knn,
        "processed":  processed,
    }

# ─── INFERENCE ───────────────────────────────────────────────────────────────
def answer_query(query: str, r: dict) -> tuple[str, str]:
    stemmer    = r["stemmer"]
    stopwords  = r["stopwords"]
    df         = r["df"]
    vectorizer = r["vectorizer"]
    svd        = r["svd"]
    lsa_matrix = r["lsa_matrix"]
    knn        = r["knn"]
    processed  = r["processed"]

    # Preprocess query
    q_proc = preprocess(query, stemmer, stopwords)
    if not q_proc.strip():
        return (
            "Maaf, saya tidak memahami pertanyaan Anda. "
            "Coba gunakan kata-kata yang lebih jelas ya! 😊",
            "–"
        )

    # TF-IDF & LSA projection
    q_tfidf = vectorizer.transform([q_proc])
    q_lsa   = normalize(svd.transform(q_tfidf))

    # Prediksi intent
    intent = knn.predict(q_lsa)[0]

    # Cosine similarity (LSA space) semua training docs
    sim_lsa = cosine_similarity(q_lsa, lsa_matrix)[0]

    # Filter kandidat berdasarkan intent
    mask = (df["kategori"] == intent).values
    if not mask.any():
        return "Maaf, saya belum bisa menjawab pertanyaan ini. 🙏", intent

    cand_idx      = np.where(mask)[0]
    sim_lsa_cands = sim_lsa[cand_idx]

    # Threshold T = (c/100) * max(sim_lsa_candidates)
    max_sim = sim_lsa_cands.max()
    T       = (THRESHOLD_C / 100) * max_sim

    # Final similarity = sim_lsa + sim_vsm (jika sim_lsa > T)
    final_sims = sim_lsa_cands.copy()
    for i, idx in enumerate(cand_idx):
        if sim_lsa_cands[i] > T:
            cand_tfidf  = vectorizer.transform([processed[idx]])
            sim_vsm     = cosine_similarity(q_tfidf, cand_tfidf)[0][0]
            final_sims[i] += sim_vsm

    best_i     = int(np.argmax(final_sims))
    best_score = sim_lsa_cands[best_i]

    if best_score < MIN_SIMILARITY:
        return (
            "Maaf, saya belum punya jawaban untuk pertanyaan tersebut. "
            "Silakan hubungi jurusan secara langsung ya! 🙏",
            intent
        )

    answer = df.iloc[cand_idx[best_i]]["jawaban"]
    return answer, intent

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="IFVA-BOT",
        page_icon="🤖",
        layout="centered",
    )

    # Header
    st.title("🤖 IFVA-BOT")
    st.caption(
        "**Informatika Virtual Assistant Bot** — "
        "Jurusan Informatika UPN \"Veteran\" Yogyakarta"
    )
    st.divider()

    # Load model (hanya sekali, lalu di-cache)
    with st.spinner("⏳ Memuat model IFVA-BOT, mohon tunggu..."):
        resources = load_all()

    stemmer_status = "✅ PySastrawi" if resources["stemmer"] else "⚠️ Tanpa stemmer"
    with st.sidebar:
        st.header("Info Sistem")
        st.markdown(f"""
        - **Dataset:** {len(resources['df'])} pasang tanya-jawab
        - **Kategori:** {len(resources['df']['kategori'].unique())} intent
        - **Vocabulary:** {len(resources['vectorizer'].vocabulary_)} term
        - **LSA k:** {resources['svd'].n_components} dimensi
        - **KNN k:** {K_NEIGHBORS} tetangga
        - **Stemmer:** {stemmer_status}
        """)
        st.divider()
        st.markdown("**Topik yang bisa ditanyakan:**")
        for label in INTENT_LABELS.values():
            st.markdown(f"- {label}")
        st.divider()
        if st.button("🗑️ Bersihkan percakapan"):
            st.session_state.messages = _welcome_messages()
            st.rerun()

    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = _welcome_messages()

    # Tampilkan riwayat chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("intent"):
                label = INTENT_LABELS.get(msg["intent"], msg["intent"])
                st.caption(f"🏷️ Intent: **{label}**")

    # Input pengguna
    if prompt := st.chat_input("Ketik pertanyaan Anda..."):
        # Tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Dapatkan jawaban
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                answer, intent = answer_query(prompt, resources)
            st.markdown(answer)
            if intent and intent != "–":
                label = INTENT_LABELS.get(intent, intent)
                st.caption(f"🏷️ Intent: **{label}**")

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "intent":  intent,
        })


def _welcome_messages() -> list:
    return [{
        "role":    "assistant",
        "content": (
            "Halo! Saya **IFVA-BOT**, asisten virtual Jurusan Informatika "
            "UPN \"Veteran\" Yogyakarta. 😊\n\n"
            "Saya bisa membantu menjawab pertanyaan seputar:\n"
            "- 📅 Jadwal & kegiatan akademik\n"
            "- 📋 Administrasi & peraturan akademik\n"
            "- 🏫 Fasilitas jurusan\n"
            "- 👨‍🏫 Dosen & pelaksana akademik\n\n"
            "Silakan ketik pertanyaan Anda!"
        ),
        "intent": None,
    }]


if __name__ == "__main__":
    main()
