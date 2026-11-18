"""Aplikasi demo interaktif Streamlit untuk Klasifikasi Artikel Berita Arab (Medan Makna & Semantik Frame)."""

import html
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import unicodedata

# Resolve root repositori secara relatif agar berjalan dari direktori kerja mana pun
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from arabic_news.acquisition import find_file_unicode_safe, load_or_compute_corpus_stats
from arabic_news.config import (
    LABELS,
    LABELS_ID,
    LEXICONS_DIR,
    METRICS_DIR,
    STATS_DIR,
)
from arabic_news.evaluation import parse_evaluation_metrics_txt, parse_resource_log_txt
from arabic_news.models import analyze_article_dual_svm, load_compact_svm_bundles
from arabic_news.preprocessing import load_stopwords, preprocess_text, reshape_arabic_for_plot
from arabic_news.visualization import find_arabic_font_path

JOURNAL_DOI_URL = "https://doi.org/10.24815/siele.v11i3.38090"
THESIS_REPO_URL = "https://digilib.uin-suka.ac.id/id/eprint/75147/"

MODELS_EVAL_CONFIG = [
    ("SVC", "SVM", "svm.txt", "SVM"),
    ("LogisticRegression", "Logistic Regression", "logistic.txt", "LogReg"),
    ("MultinomialNB", "Naïve Bayes", "naive.txt", "Naïve Bayes"),
    ("RandomForest", "Random Forest", "forest.txt", "Random Forest"),
    ("GradientBoosting", "Gradient Boosting", "gradient.txt", "GradBoost"),
]


def load_evaluation_summary_tables(metrics_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Memuat tabel efektivitas, efisiensi, laporan per kategori, dan confusion matrix dari results/metrics/."""
    m_dir = Path(metrics_dir) if metrics_dir else METRICS_DIR
    eff_rows: List[Dict[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []
    short_labels: List[str] = []
    cm_by_scheme: Dict[str, Optional[Dict[str, Any]]] = {"frame": None, "medan": None}

    for key, label_name, res_file, short_name in MODELS_EVAL_CONFIG:
        mf_path = m_dir / "frame_semantik" / f"no_lemma_evaluation_metrics_{key}.txt"
        mn_path = m_dir / "medan_semantik" / f"no_lemma_evaluation_metrics_{key}.txt"
        if not mf_path.exists() or not mn_path.exists():
            continue

        mf = parse_evaluation_metrics_txt(mf_path)
        mn = parse_evaluation_metrics_txt(mn_path)

        rf_path = m_dir / "cpu_and_memory" / "frame_semantik" / res_file
        rn_path = m_dir / "cpu_and_memory" / "medan_semantik" / res_file
        rf = parse_resource_log_txt(rf_path) if rf_path.exists() else {"max_memory_mb": 0.0, "avg_cpu_percent": 0.0}
        rn = parse_resource_log_txt(rn_path) if rn_path.exists() else {"max_memory_mb": 0.0, "avg_cpu_percent": 0.0}

        mem_inc_pct = (
            round((rf["max_memory_mb"] - rn["max_memory_mb"]) / rn["max_memory_mb"] * 100, 2)
            if rn["max_memory_mb"]
            else 0.0
        )
        eff_rows.append(
            {
                "Model": label_name,
                "Waktu Frame (s)": round(mf["execution_time_seconds"], 4),
                "Waktu Medan (s)": round(mn["execution_time_seconds"], 4),
                "RAM Frame (MB)": rf["max_memory_mb"],
                "RAM Medan (MB)": rn["max_memory_mb"],
                "Kenaikan RAM (%)": mem_inc_pct,
                "CPU Frame (%)": rf["avg_cpu_percent"],
                "CPU Medan (%)": rn["avg_cpu_percent"],
            }
        )
        perf_rows.append(
            {
                "Model": label_name,
                "Akurasi Frame": mf["accuracy"],
                "Prec Frame": mf["precision"],
                "Rec Frame": mf["recall"],
                "F1 Frame": mf["f1_score"],
                "Support Frame": mf["support"],
                "Akurasi Medan": mn["accuracy"],
                "Prec Medan": mn["precision"],
                "Rec Medan": mn["recall"],
                "F1 Medan": mn["f1_score"],
                "Support Medan": mn["support"],
            }
        )
        short_labels.append(short_name)
        if key == "SVC":
            cm_by_scheme["frame"] = mf
            cm_by_scheme["medan"] = mn

    return {
        "efficiency": pd.DataFrame(eff_rows),
        "performance": pd.DataFrame(perf_rows),
        "short_names": short_labels,
        "svm_frame_detail": cm_by_scheme["frame"],
        "svm_medan_detail": cm_by_scheme["medan"],
    }


def load_category_lexicon_snapshot(
    category_name: str,
    top_k: int = 15,
    lexicons_dir: Optional[Path] = None,
    stats_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Memuat kamus Semantik Frame dan daftar kata/frasa frekuensi tertinggi untuk satu kategori Arab."""
    norm_cat = unicodedata.normalize("NFC", category_name)
    l_dir = Path(lexicons_dir) if lexicons_dir else LEXICONS_DIR
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR

    try:
        frame_df = (
            pd.read_csv(find_file_unicode_safe(l_dir / "tfidf_top1000", f"ngram_tfidf_weights_{norm_cat}", ".csv"))
            .head(top_k)
            .rename(columns={"N-gram": "Frasa Semantik Frame", "TF-IDF Weight": "Bobot TF-IDF"})
        )
    except FileNotFoundError:
        frame_df = pd.DataFrame()

    word_rows = []
    try:
        words_path = find_file_unicode_safe(s_dir / "nonstop_words", f"nonstop_words_{norm_cat}", ".txt")
        for line in words_path.read_text(encoding="utf-8").splitlines()[1 : top_k + 1]:
            if ":" in line:
                word, freq = line.split(":", 1)
                word_rows.append({"Kata": word.strip(), "Frekuensi": int(freq.strip())})
    except FileNotFoundError:
        pass

    try:
        phrase2_df = (
            pd.read_csv(find_file_unicode_safe(l_dir / "bigram_top1000", f"bigram_{norm_cat}", ".csv"))
            .head(top_k)
            .rename(columns={"Bigram N-gram": "Frasa 2 Kata", "Frequency": "Frekuensi"})
        )
    except FileNotFoundError:
        phrase2_df = pd.DataFrame()

    try:
        phrase3_df = (
            pd.read_csv(find_file_unicode_safe(l_dir / "trigram_top1000", f"trigram_{norm_cat}", ".csv"))
            .head(top_k)
            .rename(columns={"Trigram N-gram": "Frasa 3 Kata", "Frequency": "Frekuensi"})
        )
    except FileNotFoundError:
        phrase3_df = pd.DataFrame()

    return {
        "frame_tfidf": frame_df,
        "core_words": pd.DataFrame(word_rows),
        "phrase_2": phrase2_df,
        "phrase_3": phrase3_df,
    }


def load_sample_test_articles(stats_dir: Optional[Path] = None) -> pd.DataFrame:
    """Memuat 11 artikel uji hasil Notebook 02 dari results/stats/sample_11_test_articles.csv."""
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR
    sample_path = s_dir / "sample_11_test_articles.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return pd.DataFrame(columns=["label", "category", "filename", "source", "text"])


def load_dashboard_corpus_stats(stats_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Memuat ringkasan korpus yang sudah dihasilkan di results/stats/ tanpa membebani startup UI."""
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR
    summary_json = s_dir / "corpus_summary.json"
    if summary_json.exists():
        return load_or_compute_corpus_stats(stats_dir=s_dir, prefer_summary=True)

    dist_json = s_dir / "corpus_distributions.json"
    if dist_json.exists():
        data = json.loads(dist_json.read_text(encoding="utf-8"))
        return {
            "source_mode": "partial_from_notebook_01",
            "total_raw_rows": data.get("total_raw_rows", 0),
            "empty_text_rows": data.get("empty_text_rows", 0),
            "total_valid_articles": data.get("total_valid_articles", 0),
            "category_distribution": pd.DataFrame(data.get("category_distribution", [])),
            "source_distribution": pd.DataFrame(data.get("source_distribution", [])),
            "tokens_per_category": {},
        }

    return {"source_mode": "not_generated_yet"}


def highlight_arabic_terms_html(
    text: str,
    positive_terms: List[str],
    negative_terms: List[str],
) -> str:
    """Merender HTML RTL dengan sorotan hijau untuk kata berkoefisien positif (+) dan merah untuk negatif (-)."""
    pos_set = set(positive_terms)
    neg_set = set(negative_terms)
    tokens = str(text).split()
    rendered: List[str] = []
    for raw_tok in tokens:
        cleaned_tok = preprocess_text(raw_tok, stopwords=None)
        safe_tok = html.escape(raw_tok)
        if cleaned_tok in pos_set:
            rendered.append(f'<span class="ar-pos" title="Koefisien Positif (+)">{safe_tok}</span>')
        elif cleaned_tok in neg_set:
            rendered.append(f'<span class="ar-neg" title="Koefisien Negatif (-)">{safe_tok}</span>')
        else:
            rendered.append(safe_tok)
    return " ".join(rendered)


@st.cache_resource(show_spinner="Memuat model Medan Makna & Semantik Frame...")
def get_cached_bundles() -> Dict[str, Dict[str, Any]]:
    return load_compact_svm_bundles()


@st.cache_data
def get_cached_stopwords() -> List[str]:
    return load_stopwords()


@st.cache_data
def get_cached_sample_articles() -> pd.DataFrame:
    return load_sample_test_articles()


@st.cache_data
def get_cached_corpus_stats() -> Dict[str, Any]:
    return load_dashboard_corpus_stats()


@st.cache_data
def get_cached_evaluation_tables() -> Dict[str, Any]:
    return load_evaluation_summary_tables()


def _get_font_prop() -> Optional[fm.FontProperties]:
    font_path = find_arabic_font_path()
    return fm.FontProperties(fname=font_path) if font_path else None


def _inject_rtl_styles() -> None:
    st.markdown(
        """
        <style>
        .arabic-rtl-box {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif;
            font-size: 1.15rem;
            line-height: 2.0;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-right: 5px solid #1d4ed8;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            color: #0f172a;
        }
        .arabic-clean-box {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif;
            font-size: 1.02rem;
            line-height: 1.85;
            background-color: #f1f5f9;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            color: #1e293b;
        }
        .ar-pos {
            background-color: #dcfce7;
            color: #166534;
            padding: 0.08rem 0.35rem;
            border-radius: 4px;
            font-weight: 600;
        }
        .ar-neg {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 0.08rem 0.35rem;
            border-radius: 4px;
        }
        textarea[aria-label*="artikel berita Arab"] {
            direction: rtl !important;
            text-align: right !important;
            unicode-bidi: plaintext !important;
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif !important;
            font-size: 1.1rem !important;
            line-height: 1.8 !important;
        }
        [data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(corpus_stats: Dict[str, Any], eval_tables: Dict[str, Any]) -> None:
    svm_f = eval_tables.get("svm_frame_detail")
    svm_m = eval_tables.get("svm_medan_detail")
    total_valid = corpus_stats.get("total_valid_articles")

    with st.sidebar:
        st.header("Tentang Penelitian")
        st.markdown(
            """
            Sistem klasifikasi artikel berita berbahasa Arab berbasis **Medan Makna (*Semantic Field*)**
            dan **Semantik *Frame* (*Frame Semantics*)**.
            """
        )
        st.metric("Total Artikel Valid Korpus", f"{total_valid:,}" if total_valid else "Belum dihitung (NB 01/03)")
        st.metric("Akurasi Semantik Frame", f"{svm_f['accuracy'] * 100:.2f}%" if svm_f else "Belum dilatih (NB 04)")
        st.metric("Akurasi Medan Makna", f"{svm_m['accuracy'] * 100:.2f}%" if svm_m else "Belum dilatih (NB 04)")

        st.divider()
        st.subheader("Publikasi Ilmiah")
        st.markdown(
            f"""
            - **Artikel Jurnal (Medan Makna)**: *Studies in English Language and Education*, 11(3), 1686–1709 (2024) — [DOI: 10.24815/siele.v11i3.38090]({JOURNAL_DOI_URL})
            - **Tesis Magister (Semantik *Frame*)**: UIN Sunan Kalijaga Yogyakarta (2024) — [Repositori UIN]({THESIS_REPO_URL})
            """
        )
        st.caption("11 Kategori | 6 Portal Berita Timur Tengah & Afrika Utara")


def _render_classification_tab(
    bundles: Dict[str, Dict[str, Any]],
    stopwords: List[str],
    samples_df: pd.DataFrame,
    font_prop: Optional[fm.FontProperties],
) -> None:
    st.subheader("Uji Klasifikasi & Ekstraksi Ganda (Medan Makna + Semantik Frame)")

    if "medan_makna" not in bundles or "frame_semantik" not in bundles:
        st.warning(
            "Model terlatih belum tersedia di folder `models/`. "
            "Jalankan `notebooks/04_model_training_and_evaluation.ipynb` terlebih dahulu untuk melatih dan menyimpan "
            "`svm_medan_makna.joblib` dan `svm_frame_semantik.joblib`."
        )
        return

    CUSTOM_INPUT_LABEL = "Tulis / Tempel Teks Arab Sendiri"
    sample_options: List[str] = [CUSTOM_INPUT_LABEL]
    label_by_option: Dict[str, int] = {}
    if not samples_df.empty:
        for _, r in samples_df.sort_values("label").iterrows():
            cid = int(r["label"])
            opt_lbl = f"Contoh {cid}: {LABELS_ID[cid]} ({LABELS[cid]})"
            sample_options.append(opt_lbl)
            label_by_option[opt_lbl] = cid
    else:
        st.caption(
            "Sampel uji otomatis belum dibuat (`results/stats/sample_11_test_articles.csv`). "
            "Anda dapat menempel artikel berita Arab langsung di kotak di bawah ini."
        )

    if st.session_state.pop("pending_clear", False):
        st.session_state["article_input"] = ""
        st.session_state["submitted_text"] = ""
        st.session_state["sample_selector"] = CUSTOM_INPUT_LABEL
        st.session_state["last_sample_source"] = CUSTOM_INPUT_LABEL

    if "article_input" not in st.session_state:
        st.session_state["article_input"] = ""
    if "last_sample_source" not in st.session_state:
        st.session_state["last_sample_source"] = CUSTOM_INPUT_LABEL
    if "submitted_text" not in st.session_state:
        st.session_state["submitted_text"] = ""

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_source = st.selectbox(
            "Sumber Teks Artikel:",
            sample_options,
            key="sample_selector",
        )
    with col_sel2:
        top_n_features = st.slider("Jumlah Fitur Teratas Ditampilkan", min_value=5, max_value=30, value=15)

    if selected_source != st.session_state["last_sample_source"]:
        st.session_state["last_sample_source"] = selected_source
        if selected_source == CUSTOM_INPUT_LABEL:
            st.session_state["article_input"] = ""
            st.session_state["submitted_text"] = ""
        else:
            chosen_cid = label_by_option[selected_source]
            sample_txt = str(samples_df[samples_df["label"] == chosen_cid].iloc[0]["text"])
            st.session_state["article_input"] = sample_txt
            st.session_state["submitted_text"] = sample_txt

    true_label_info: Optional[int] = label_by_option.get(selected_source)

    st.text_area(
        "Teks artikel berita Arab (dukungan pengetikan Right-to-Left):",
        key="article_input",
        height=190,
        placeholder="الصق أو اكتب نص مقال إخباري باللغة العربية هنا... (Tempel atau tulis teks berita Arab di sini)",
    )

    btn_col1, btn_col2, _ = st.columns([1.6, 1, 2.4])
    with btn_col1:
        classify_clicked = st.button("Klasifikasikan & Ekstrak Fitur", type="primary", width="stretch")
    with btn_col2:
        clear_clicked = st.button("Bersihkan Teks", width="stretch")

    if clear_clicked:
        st.session_state["pending_clear"] = True
        st.rerun()

    if classify_clicked:
        if not st.session_state["article_input"].strip():
            st.warning("Silakan tulis atau tempel teks berita terlebih dahulu sebelum menekan tombol klasifikasi.")
            return
        st.session_state["submitted_text"] = st.session_state["article_input"]

    active_text = st.session_state.get("submitted_text", "").strip()
    if not active_text:
        st.info(
            "Tulis atau tempel artikel berita berbahasa Arab lalu klik **Klasifikasikan & Ekstrak Fitur**, "
            "atau pilih contoh kategori pada menu **Sumber Teks Artikel** di atas."
        )
        return

    with st.spinner("Mengekstrak fitur Medan Makna, frasa Semantik Frame, dan koefisien model..."):
        dual = analyze_article_dual_svm(active_text, bundles=bundles, stopwords=stopwords)

    medan_res = dual["medan_makna"]
    frame_res = dual["frame_semantik"]

    if true_label_info is not None:
        st.caption(
            f"Label Acuan Artikel Uji: **{LABELS[true_label_info]} ({LABELS_ID[true_label_info]})**"
        )

    both_uncat = frame_res.get("is_uncategorized") and medan_res.get("is_uncategorized")
    if both_uncat:
        st.warning(
            "**Tidak Terkategori (*Uncategorized*)** — Tidak ditemukan kata maupun frasa berita berbahasa Arab "
            "yang dikenali oleh kedua model (0 fitur aktif)."
        )

    c1, c2 = st.columns(2)
    with c1:
        if frame_res.get("is_uncategorized"):
            st.warning(
                "**Prediksi Semantik Frame**: `غير مصنف` — **Tidak Terkategori** *(Uncategorized — 0 frasa aktif)*"
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Skor Sigmoid", "—")
            m2.metric("Frasa Aktif", "0")
            m3.metric("Total Bobot TF-IDF", "0.0000")
        else:
            top_prob_f = frame_res["category_scores"][0]["sigmoid_percent"]
            st.success(
                f"**Prediksi Semantik Frame**: `{frame_res['predicted_category']}` "
                f"— **{frame_res['predicted_category_id']}** ({frame_res['predicted_category_en']})"
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Skor Sigmoid", f"{top_prob_f:.2f}%")
            m2.metric("Frasa Aktif", f"{frame_res['n_active_features']}")
            m3.metric("Total Bobot TF-IDF", f"{frame_res['total_tfidf_weight']:.4f}")

    with c2:
        if medan_res.get("is_uncategorized"):
            st.warning(
                "**Prediksi Medan Makna**: `غير مصنف` — **Tidak Terkategori** *(Uncategorized — 0 kata aktif)*"
            )
            n1, n2, n3 = st.columns(3)
            n1.metric("Skor Sigmoid", "—")
            n2.metric("Kata Aktif", "0")
            n3.metric("Total Bobot TF-IDF", "0.0000")
        else:
            top_prob_m = medan_res["category_scores"][0]["sigmoid_percent"]
            st.info(
                f"**Prediksi Medan Makna**: `{medan_res['predicted_category']}` "
                f"— **{medan_res['predicted_category_id']}** ({medan_res['predicted_category_en']})"
            )
            n1, n2, n3 = st.columns(3)
            n1.metric("Skor Sigmoid", f"{top_prob_m:.2f}%")
            n2.metric("Kata Aktif", f"{medan_res['n_active_features']}")
            n3.metric("Total Bobot TF-IDF", f"{medan_res['total_tfidf_weight']:.4f}")

    if both_uncat:
        return

    st.markdown("#### Tampilan Teks Berita Arab (RTL) & Sorotan Kata Medan Makna")
    st.caption(
        "Hijau = kata medan makna berkoefisien positif (+) untuk kategori prediksi; "
        "Merah = berkoefisien negatif (-)."
    )
    pos_terms = [d["term"] for d in medan_res["positive_coefs"]]
    neg_terms = [d["term"] for d in medan_res["negative_coefs"]]
    highlighted_html = highlight_arabic_terms_html(active_text, pos_terms, neg_terms)
    st.markdown(f'<div class="arabic-rtl-box">{highlighted_html}</div>', unsafe_allow_html=True)

    with st.expander("Lihat Hasil Pra-pemrosesan Teks (Medan Makna vs Semantik Frame)"):
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**1. Jalur Semantik Frame (Tanpa Penghapusan Stopwords / Preposisi Utuh):**")
            st.markdown(
                f'<div class="arabic-clean-box">{html.escape(frame_res["cleaned_text"] or "—")}</div>',
                unsafe_allow_html=True,
            )
        with e2:
            st.markdown("**2. Jalur Medan Makna (Setelah Stopword Removal):**")
            st.markdown(
                f'<div class="arabic-clean-box">{html.escape(medan_res["cleaned_text"] or "—")}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Distribusi Skor Probabilitas Sigmoid Lintas 11 Kategori")
    score_col1, score_col2 = st.columns(2)
    for col_box, title_txt, res_obj, bar_color in [
        (score_col1, "Semantik Frame", frame_res, "#0f766e"),
        (score_col2, "Medan Makna", medan_res, "#1d4ed8"),
    ]:
        with col_box:
            if res_obj.get("is_uncategorized"):
                st.info(f"{title_txt}: **Tidak Terkategori (Uncategorized)** karena 0 fitur aktif.")
                continue
            scores_df = pd.DataFrame(res_obj["category_scores"])
            scores_sorted = scores_df.sort_values("sigmoid_percent", ascending=True)
            fig, ax = plt.subplots(figsize=(7.6, 5.0))
            ylabels = [reshape_arabic_for_plot(r["category"]) for _, r in scores_sorted.iterrows()]
            bars = ax.barh(ylabels, scores_sorted["sigmoid_percent"], color=bar_color, height=0.68)
            for bar, val in zip(bars, scores_sorted["sigmoid_percent"]):
                ax.text(
                    min(float(val) + 1.5, 102.0),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=8.5,
                    color="#1e293b",
                )
            ax.set_xlim(0, 118)
            ax.set_xlabel("Skor Sigmoid (%)", fontsize=10)
            ax.set_title(title_txt, fontsize=11, pad=10)
            ax.tick_params(axis="y", labelsize=9.5)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            if font_prop:
                for lbl in ax.get_yticklabels():
                    lbl.set_fontproperties(font_prop)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.markdown("#### Detail Fitur Medan Makna, Frasa Semantik *Frame*, Bobot TF-IDF, dan Koefisien Model")
    ext_col1, ext_col2 = st.columns(2)

    with ext_col1:
        st.markdown("##### 1. Ekstraksi Medan Makna")
        if medan_res.get("is_uncategorized"):
            st.info("Tidak ada kata Medan Makna yang cocok pada input teks (**Tidak Terkategori**).")
        else:
            st.write(
                f"Koefisien (+) : `{len(medan_res['positive_coefs'])}` kata "
                f"(total **`+{medan_res['total_positive_coef']:.4f}`**) | "
                f"Koefisien (-) : `{len(medan_res['negative_coefs'])}` kata "
                f"(total **`{medan_res['total_negative_coef']:.4f}`**) | "
                f"Bias (`intercept`): `{medan_res['intercept']:.4f}`"
            )
            medan_df = pd.DataFrame(medan_res["features_by_tfidf"]).rename(
                columns={
                    "term": "Kata Medan Makna",
                    "tfidf_weight": "Bobot TF-IDF",
                    "svm_coef": "Koefisien Model",
                    "contribution": "Kontribusi",
                }
            )
            if not medan_df.empty:
                medan_top = medan_df.sort_values(
                    by="Koefisien Model", key=lambda s: s.abs(), ascending=False
                ).head(top_n_features)
                fig_h_m = max(4.6, 0.34 * len(medan_top) + 1.4)
                fig_m, ax_m = plt.subplots(figsize=(7.6, fig_h_m))
                plot_m = medan_top.sort_values("Koefisien Model", ascending=True)
                ax_m.barh(
                    [reshape_arabic_for_plot(t) for t in plot_m["Kata Medan Makna"]],
                    plot_m["Koefisien Model"],
                    color=["#16a34a" if v > 0 else "#dc2626" for v in plot_m["Koefisien Model"]],
                    height=0.68,
                )
                ax_m.axvline(0, color="black", linewidth=0.8)
                ax_m.set_title(f"Top {len(plot_m)} Koefisien Medan Makna (+/-)", fontsize=11, pad=10)
                ax_m.set_xlabel("Koefisien Hyperplane", fontsize=10)
                ax_m.tick_params(axis="y", labelsize=9.5)
                ax_m.grid(axis="x", linestyle="--", alpha=0.3)
                if font_prop:
                    for lbl in ax_m.get_yticklabels():
                        lbl.set_fontproperties(font_prop)
                fig_m.tight_layout()
                st.pyplot(fig_m)
                plt.close(fig_m)

                st.dataframe(
                    medan_df[["Kata Medan Makna", "Bobot TF-IDF", "Koefisien Model", "Kontribusi"]].head(
                        top_n_features
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with ext_col2:
        st.markdown("##### 2. Ekstraksi Semantik *Frame*")
        if frame_res.get("is_uncategorized"):
            st.info("Tidak ada frasa Semantik *Frame* yang aktif pada input teks (**Tidak Terkategori**).")
        else:
            st.write(
                f"Koefisien (+) : `{len(frame_res['positive_coefs'])}` frasa "
                f"(total **`+{frame_res['total_positive_coef']:.4f}`**) | "
                f"Koefisien (-) : `{len(frame_res['negative_coefs'])}` frasa "
                f"(total **`{frame_res['total_negative_coef']:.4f}`**) | "
                f"Bias (`intercept`): `{frame_res['intercept']:.4f}`"
            )
            frame_df = pd.DataFrame(frame_res["features_by_tfidf"]).rename(
                columns={
                    "term": "Frasa Semantik Frame",
                    "n_tokens": "Jumlah Kata",
                    "tfidf_weight": "Bobot TF-IDF",
                    "svm_coef": "Koefisien Model",
                    "contribution": "Kontribusi",
                }
            )
            if not frame_df.empty:
                frame_top = frame_df.head(top_n_features).iloc[::-1]
                fig_h_f = max(4.6, 0.34 * len(frame_top) + 1.4)
                fig_f, ax_f = plt.subplots(figsize=(7.6, fig_h_f))
                ax_f.barh(
                    [reshape_arabic_for_plot(t) for t in frame_top["Frasa Semantik Frame"]],
                    frame_top["Bobot TF-IDF"],
                    color="#0d9488",
                    height=0.68,
                )
                max_w = float(frame_top["Bobot TF-IDF"].max()) if len(frame_top) else 1.0
                ax_f.set_xlim(0, max_w * 1.15)
                ax_f.set_title(f"Top {len(frame_top)} Frasa Semantik Frame (Bobot TF-IDF)", fontsize=11, pad=10)
                ax_f.set_xlabel("Bobot TF-IDF", fontsize=10)
                ax_f.tick_params(axis="y", labelsize=9.5)
                ax_f.grid(axis="x", linestyle="--", alpha=0.3)
                if font_prop:
                    for lbl in ax_f.get_yticklabels():
                        lbl.set_fontproperties(font_prop)
                fig_f.tight_layout()
                st.pyplot(fig_f)
                plt.close(fig_f)

                st.dataframe(
                    frame_df[
                        ["Frasa Semantik Frame", "Jumlah Kata", "Bobot TF-IDF", "Koefisien Model", "Kontribusi"]
                    ].head(top_n_features),
                    width="stretch",
                    hide_index=True,
                )


def _render_corpus_and_lexicons_tab(
    corpus_stats: Dict[str, Any],
    font_prop: Optional[fm.FontProperties],
) -> None:
    st.subheader("Profil Korpus Berita Arab & Kamus Semantik 11 Kategori")

    source_mode = corpus_stats.get("source_mode", "not_generated_yet")
    if source_mode == "not_generated_yet":
        st.info(
            "Ringkasan statistik korpus di `results/stats/` belum tersedia. "
            "Jalankan `notebooks/01_data_collection_and_compilation.ipynb` dan "
            "`notebooks/03_corpus_trends_and_linguistic_stats.ipynb` untuk menghitungnya dari `data/basic_articles_dataset.csv`."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Artikel Valid", f"{corpus_stats['total_valid_articles']:,}")
        total_tok = corpus_stats.get("total_tokens")
        ar_types = corpus_stats.get("arabic_word_types")
        nonar_types = corpus_stats.get("nonarabic_word_types")
        m2.metric("Total Token Korpus", f"{total_tok:,}" if total_tok is not None else "Jalankan NB 03")
        m3.metric("Tipe Kata Unik Arab", f"{ar_types:,}" if ar_types is not None else "Jalankan NB 03")
        m4.metric("Tipe Kata Non-Arab", f"{nonar_types:,}" if nonar_types is not None else "Jalankan NB 03")

        cat_df: pd.DataFrame = corpus_stats["category_distribution"].copy()
        if corpus_stats.get("tokens_per_category"):
            cat_df["Total Token"] = cat_df["Kategori"].map(corpus_stats["tokens_per_category"])
        src_df: pd.DataFrame = corpus_stats["source_distribution"].copy()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_cat, ax_cat = plt.subplots(figsize=(8.6, 5.4))
            labels_cat = [reshape_arabic_for_plot(c) for c in cat_df["Kategori"].tolist()[::-1]]
            trans_cat = cat_df["Terjemahan"].tolist()[::-1]
            vals_cat = cat_df["Artikel Valid"].tolist()[::-1]
            bars_cat = ax_cat.barh(labels_cat, vals_cat, color="#2563eb", height=0.68)
            max_cat = max(vals_cat) if vals_cat else 1
            for bar, val, trans in zip(bars_cat, vals_cat, trans_cat):
                ax_cat.text(
                    val + max_cat * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{trans} ({val:,})",
                    va="center",
                    fontsize=8.5,
                    color="#1e293b",
                )
            ax_cat.set_xlim(0, max_cat * 1.48)
            ax_cat.set_title("Distribusi Artikel pada 11 Kategori", fontsize=11.5, pad=10)
            ax_cat.set_xlabel("Jumlah Artikel Valid", fontsize=10)
            ax_cat.tick_params(axis="y", labelsize=10)
            ax_cat.grid(axis="x", linestyle="--", alpha=0.3)
            if font_prop:
                for lbl in ax_cat.get_yticklabels():
                    lbl.set_fontproperties(font_prop)
            fig_cat.tight_layout()
            st.pyplot(fig_cat)
            plt.close(fig_cat)

        with col_c2:
            fig_src, ax_src = plt.subplots(figsize=(8.2, 5.4))
            total_src = float(src_df["Jumlah Artikel"].sum()) or 1.0
            colors_src = plt.cm.Set2(np.linspace(0.05, 0.85, len(src_df)))
            wedges, _, _ = ax_src.pie(
                src_df["Jumlah Artikel"],
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 4.0 else "",
                pctdistance=0.76,
                startangle=140,
                colors=colors_src,
                wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.5},
            )
            legend_labels = [
                f"{r['Portal Berita']} ({r['Asal Negara']}) — {r['Jumlah Artikel'] / total_src * 100:.1f}%"
                for _, r in src_df.iterrows()
            ]
            ax_src.legend(
                wedges,
                legend_labels,
                title="Portal Berita Arab",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=2,
                fontsize=8.5,
                frameon=False,
            )
            ax_src.set_title("Proporsi Sumber 6 Portal Berita Arab", fontsize=11.5, pad=10)
            fig_src.tight_layout()
            st.pyplot(fig_src)
            plt.close(fig_src)

        with st.expander("Lihat Tabel Rincian Kategori dan Portal Berita"):
            t1, t2 = st.columns(2)
            with t1:
                st.dataframe(cat_df, width="stretch", hide_index=True)
            with t2:
                st.dataframe(src_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Penjelajah Kamus Medan Makna & Semantik *Frame* per Kategori")
    cat_choices = [f"{cid}: {cname} ({LABELS_ID[cid]})" for cid, cname in LABELS.items()]
    chosen = st.selectbox("Pilih Kategori untuk Meninjau Leksikon:", cat_choices)
    chosen_cid = int(chosen.split(":")[0].strip())
    chosen_name = LABELS[chosen_cid]

    lex = load_category_lexicon_snapshot(chosen_name, top_k=15)
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"**Top-15 Kamus Semantik *Frame* TF-IDF (`{chosen_name}`)**")
        if lex["frame_tfidf"].empty:
            st.info("Belum tersedia. Jalankan Notebook 05 untuk menghasilkan kamus TF-IDF Semantik Frame.")
        else:
            st.dataframe(lex["frame_tfidf"], width="stretch", hide_index=True)

        st.markdown(f"**Top-15 Kata Inti Tanpa Stopwords (`{chosen_name}`)**")
        if lex["core_words"].empty:
            st.info("Belum tersedia. Jalankan Notebook 03 untuk menghasilkan daftar kata teratas.")
        else:
            st.dataframe(lex["core_words"], width="stretch", hide_index=True)

    with k2:
        st.markdown(f"**Top-15 Frasa 2 Kata Frekuensi Tertinggi (`{chosen_name}`)**")
        if lex["phrase_2"].empty:
            st.info("Belum tersedia. Jalankan Notebook 03 untuk menghasilkan daftar frasa teratas.")
        else:
            st.dataframe(lex["phrase_2"], width="stretch", hide_index=True)

        st.markdown(f"**Top-15 Frasa 3 Kata Frekuensi Tertinggi (`{chosen_name}`)**")
        if lex["phrase_3"].empty:
            st.info("Belum tersedia. Jalankan Notebook 03 untuk menghasilkan daftar frasa teratas.")
        else:
            st.dataframe(lex["phrase_3"], width="stretch", hide_index=True)


def _render_evaluation_tab(
    eval_tables: Dict[str, Any],
    font_prop: Optional[fm.FontProperties],
) -> None:
    st.subheader("Evaluasi Efektivitas & Efisiensi 5 Algoritma Klasifikasi")

    eff_df: pd.DataFrame = eval_tables["efficiency"]
    perf_df: pd.DataFrame = eval_tables["performance"]
    svm_frame = eval_tables.get("svm_frame_detail")
    svm_medan = eval_tables.get("svm_medan_detail")

    if perf_df.empty or svm_frame is None or svm_medan is None:
        st.info(
            "Laporan metrik evaluasi 5 model belum ditemukan di `results/metrics/`. "
            "Jalankan `notebooks/04_model_training_and_evaluation.ipynb` terlebih dahulu untuk melatih dan mengevaluasi kelima algoritma."
        )
        return

    st.markdown(
        f"""
        Perbandingan eksperimen pada {len(perf_df)} model antara pendekatan **Dengan Semantik *Frame*** (`N={svm_frame['support']:,}`)
        dan **Tanpa Semantik *Frame* / Medan Makna** (`N={svm_medan['support']:,}`).
        """
    )

    tab_eff, tab_perf = st.tabs(
        ["Efektivitas Klasifikasi", "Efisiensi Komputasi (Waktu, RAM, CPU)"]
    )
    with tab_eff:
        st.dataframe(perf_df, width="stretch", hide_index=True)
    with tab_perf:
        st.dataframe(eff_df, width="stretch", hide_index=True)

    x = np.arange(len(perf_df))
    w = 0.36
    short_names = eval_tables["short_names"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    min_acc = float(min(perf_df["Akurasi Frame"].min(), perf_df["Akurasi Medan"].min()) * 100)
    b1 = axes[0].bar(x - w / 2, perf_df["Akurasi Frame"] * 100, w, label="Semantik Frame", color="#0d9488")
    b2 = axes[0].bar(x + w / 2, perf_df["Akurasi Medan"] * 100, w, label="Medan Makna", color="#2563eb")
    axes[0].set_ylim(max(0.0, min_acc - 5.0), 96.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[0].set_title("Akurasi (%)", fontsize=11, pad=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x - w / 2, eff_df["Waktu Frame (s)"], w, color="#0d9488")
    axes[1].bar(x + w / 2, eff_df["Waktu Medan (s)"], w, color="#2563eb")
    max_t = float(max(eff_df["Waktu Frame (s)"].max(), eff_df["Waktu Medan (s)"].max()))
    axes[1].set_ylim(0, max_t * 1.15 if max_t else 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[1].set_title("Waktu Evaluasi (detik)", fontsize=11, pad=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    axes[2].bar(x - w / 2, eff_df["RAM Frame (MB)"], w, color="#0d9488")
    axes[2].bar(x + w / 2, eff_df["RAM Medan (MB)"], w, color="#2563eb")
    max_r = float(max(eff_df["RAM Frame (MB)"].max(), eff_df["RAM Medan (MB)"].max()))
    axes[2].set_ylim(0, max_r * 1.15 if max_r else 1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[2].set_title("Penggunaan RAM Puncak (MB)", fontsize=11, pad=10)
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    fig.legend(
        [b1[0], b2[0]],
        ["Semantik Frame", "Medan Makna"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=10.5,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Performa Per-Kategori & Confusion Matrix Model SVM")

    scheme_choice = st.radio(
        "Pilih Skema Evaluasi SVM:",
        [
            f"Dengan Semantik Frame — Akurasi {svm_frame['accuracy'] * 100:.2f}%",
            f"Tanpa Semantik Frame (Medan Makna) — Akurasi {svm_medan['accuracy'] * 100:.2f}%",
        ],
        horizontal=True,
    )
    active_svm = svm_frame if scheme_choice.startswith("Dengan Semantik Frame") else svm_medan

    per_class_df = pd.DataFrame(
        [
            {
                "ID": cid,
                "Kategori": LABELS_ID[cid],
                "Arab": LABELS[cid],
                "Precision": active_svm["per_class"][cid]["precision"],
                "Recall": active_svm["per_class"][cid]["recall"],
                "F1-Score": active_svm["per_class"][cid]["f1-score"],
                "Support": active_svm["per_class"][cid]["support"],
            }
            for cid in range(1, 12)
        ]
    )

    cm_col1, cm_col2 = st.columns([1.15, 1])
    with cm_col1:
        labels_ar = [reshape_arabic_for_plot(LABELS[i]) for i in range(1, 12)]
        fig_cm, ax_cm = plt.subplots(figsize=(9.8, 8.4))
        sns.heatmap(
            active_svm["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_ar,
            yticklabels=labels_ar,
            cbar_kws={"shrink": 0.82},
            ax=ax_cm,
        )
        ax_cm.set_title(
            f"Confusion Matrix SVM — Akurasi: {active_svm['accuracy'] * 100:.2f}% (N = {active_svm['support']:,})",
            fontsize=11.5,
            pad=14,
        )
        ax_cm.set_xlabel("Prediksi", fontsize=10.5, labelpad=8)
        ax_cm.set_ylabel("Aktual", fontsize=10.5, labelpad=8)
        ax_cm.set_xticklabels(labels_ar, rotation=32, ha="right", fontsize=10)
        ax_cm.set_yticklabels(labels_ar, rotation=0, fontsize=10)
        if font_prop:
            for lbl in ax_cm.get_xticklabels() + ax_cm.get_yticklabels():
                lbl.set_fontproperties(font_prop)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with cm_col2:
        st.dataframe(
            per_class_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall": st.column_config.NumberColumn(format="%.4f"),
                "F1-Score": st.column_config.NumberColumn(format="%.4f"),
            },
        )


def _render_methodology_tab() -> None:
    st.subheader("Metodologi Penelitian & Arsitektur Sistem")
    st.markdown(
        """
        Penelitian ini menggabungkan dua pendekatan representasi semantik dari sumber data tunggal
        `data/basic_articles_dataset.csv` dan `data/stopwords.txt`:

        1. **Tahap 1 — Pengumpulan Data (*Web Scraping*)**: Akuisisi otomatis dari 6 portal berita daring berbahasa Arab
           (`alsumaria.tv`, `al-sharq.com`, `assahraa.ma`, `akhbarelsyaha.com`, `sarayanews.com`, `addustour.com`)
           pada 11 kategori tematik.
        2. **Tahap 2 — Pra-pemrosesan Teks & Pembagian Dataset**:
           - Pembersihan angka, tanda baca Arab (`،؛؟`) & ASCII, konversi huruf kecil, dan tokenisasi.
           - **Jalur Medan Makna**: menghapus *stopwords* untuk menyorot kata leksikal inti (pembagian data `80% / 15% / 5%`).
           - **Jalur Semantik *Frame***: mempertahankan preposisi/partikel Arab (`في`, `من`, `على`, `إلى`, `أن`) sebagai unsur pembentuk struktur frasa kontekstual 2–3 kata (pembagian data `85% / 10% / 5%`).
        3. **Tahap 3 — Statistik Korpus & Distribusi Kata/Frasa**:
           - Menghitung distribusi token, tipe kosakata unik Arab vs Non-Arab, serta Top-1000 kata dan frasa berita tiap kategori.
        4. **Tahap 4 — Pelatihan, Efisiensi Komputasi & Evaluasi 5 Model**:
           - Melatih lima algoritma klasifikasi (*SVM*, *Logistic Regression*, *Naïve Bayes*, *Random Forest*, dan *Gradient Boosting*), memantau konsumsi RAM & CPU via `psutil`, serta menyimpan model **SVM** ke `models/`.
        5. **Tahap 5 — Kamus TF-IDF Semantik *Frame* & Ekstraksi Fitur**:
           - Menyusun kamus bobot TF-IDF frasa kontekstual per kategori dan memetakan skor keputusan melalui fungsi **Sigmoid** ($\\sigma(z) = \\frac{1}{1 + e^{-z}}$) untuk menampilkan kata/frasa berkoefisien positif (`+`) dan negatif (`-`).
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Klasifikasi Artikel Berita Arab — Medan Makna & Semantik Frame",
        page_icon="📰",
        layout="wide",
    )
    _inject_rtl_styles()

    st.title("Klasifikasi Artikel Berita Arab Berbasis Medan Makna dan Semantik *Frame*")
    st.caption(
        "Demo interaktif klasifikasi berita Arab (11 kategori), ekstraksi Medan Makna dan Semantik Frame, "
        "bobot kontribusi fitur, serta evaluasi efisiensi & efektivitas model."
    )

    bundles = get_cached_bundles()
    stopwords = get_cached_stopwords()
    samples_df = get_cached_sample_articles()
    corpus_stats = get_cached_corpus_stats()
    eval_tables = get_cached_evaluation_tables()
    font_prop = _get_font_prop()

    _render_sidebar(corpus_stats, eval_tables)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Klasifikasi & Ekstraksi Semantik",
            "Korpus & Kamus Semantik",
            "Evaluasi 5 Algoritma",
            "Metodologi Penelitian",
        ]
    )

    with tab1:
        _render_classification_tab(bundles, stopwords, samples_df, font_prop)
    with tab2:
        _render_corpus_and_lexicons_tab(corpus_stats, font_prop)
    with tab3:
        _render_evaluation_tab(eval_tables, font_prop)
    with tab4:
        _render_methodology_tab()


if __name__ == "__main__":
    main()
