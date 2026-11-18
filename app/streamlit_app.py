"""Aplikasi demo interaktif Streamlit untuk Klasifikasi Artikel Berita Arab (Medan Makna & Semantik Frame)."""

import html
import json
from pathlib import Path
import re
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
GITHUB_REPO_URL = "https://github.com/moel-ihsan/arabic_news_classification"

INDIGO_PRIMARY = "#4F46E5"
BLUE_ACCENT = "#2563EB"
TEAL_SECONDARY = "#0D9488"
INK_DARK = "#111111"

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


def _inject_design_system_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Amiri:wght@400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .stApp {
            background-color: #F4F5FA;
            color: #0F172A;
        }
        .block-container {
            max-width: 1220px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        /* Top Indigo Hero Card */
        .hero-bento {
            background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 55%, #2563EB 100%);
            border-radius: 22px;
            padding: 2rem 2.25rem;
            color: #FFFFFF;
            box-shadow: 0 18px 40px rgba(79, 70, 229, 0.22);
            margin-bottom: 1.4rem;
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 1.75rem;
            align-items: center;
        }
        .hero-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.28);
            color: #FFFFFF;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.32rem 0.8rem;
            border-radius: 9999px;
            margin-bottom: 0.85rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.18;
            letter-spacing: -0.025em;
            margin: 0 0 0.7rem 0;
            color: #FFFFFF;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            line-height: 1.55;
            color: rgba(255, 255, 255, 0.90);
            margin: 0;
        }
        .hero-widget {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 1.05rem;
            color: #0F172A;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16);
        }
        .hero-widget-label {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.7rem;
        }
        .hero-tiles {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.6rem;
        }
        .hero-tile {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 13px;
            padding: 0.7rem 0.8rem;
        }
        .hero-tile.active {
            background: #111111;
            border-color: #111111;
            color: #FFFFFF;
        }
        .hero-tile-val {
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
        }
        .hero-tile-cap {
            font-size: 0.74rem;
            color: #64748B;
            margin-top: 0.2rem;
            font-weight: 600;
        }
        .hero-tile.active .hero-tile-cap {
            color: rgba(255, 255, 255, 0.78);
        }

        /* 3-Step Bento Strip */
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin-bottom: 1.35rem;
        }
        .step-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 9px;
            background: #111111;
            color: #FFFFFF;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .step-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.25rem;
        }
        .step-desc {
            font-size: 0.83rem;
            line-height: 1.45;
            color: #475569;
            margin: 0;
        }

        /* Pill Tabs (menargetkan semantic [role="tab"] tanpa menyembunyikan tab itu sendiri) */
        .stTabs [role="tablist"] {
            background-color: #FFFFFF !important;
            padding: 8px 10px !important;
            border-radius: 9999px !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
            gap: 6px !important;
            margin-bottom: 1.15rem !important;
        }
        .stTabs [role="tablist"] > div:not([role="tab"]):not(:has([role="tab"])) {
            display: none !important;
            height: 0 !important;
            background: transparent !important;
        }
        .stTabs [role="tablist"] > div:has([role="tab"]) {
            display: inline-flex !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .stTabs [role="tablist"] .react-aria-SelectionIndicator {
            display: none !important;
        }
        .stTabs [role="tablist"] [role="tab"] {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 9999px !important;
            padding: 0.6rem 1.2rem !important;
            min-height: 42px !important;
            height: auto !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: #475569 !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stTabs [role="tablist"] [role="tab"] p {
            margin: 0 !important;
            font-weight: 600 !important;
        }
        .stTabs [role="tablist"] [role="tab"][aria-selected="true"] {
            background-color: #111111 !important;
            color: #FFFFFF !important;
            border-radius: 9999px !important;
        }
        .stTabs [role="tablist"] [role="tab"][aria-selected="true"] p {
            color: #FFFFFF !important;
        }

        /* Compact Bento Metrics inside Prediction Columns */
        .pred-metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.45rem;
            margin-bottom: 0.9rem;
        }
        .pred-metric-chip {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.03);
        }
        .pred-metric-label {
            font-size: 0.74rem;
            font-weight: 600;
            color: #64748B;
            margin-bottom: 0.2rem;
            white-space: nowrap;
        }
        .pred-metric-val {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }

        /* Elevated Pill CTA Buttons */
        .stButton > button {
            border-radius: 9999px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.35rem !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease !important;
        }
        .stButton > button[kind="primary"] {
            background-color: #111111 !important;
            border-color: #111111 !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 16px rgba(17, 17, 17, 0.18) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #312E81 !important;
            border-color: #312E81 !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            color: #0F172A !important;
        }

        /* Elevated Metric Cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        }

        /* Side-by-Side Comparison Bento Cards */
        .compare-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 0.6rem 0 1.4rem 0;
        }
        .compare-card-muted {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 1.35rem 1.5rem;
            color: #1E293B;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        }
        .compare-card-highlight {
            background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
            border-radius: 20px;
            padding: 1.35rem 1.5rem;
            color: #FFFFFF;
            box-shadow: 0 14px 30px rgba(79, 70, 229, 0.22);
        }
        .compare-tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.24rem 0.65rem;
            border-radius: 9999px;
            margin-bottom: 0.55rem;
        }
        .compare-card-muted .compare-tag {
            background: #F1F5F9;
            color: #475569;
        }
        .compare-card-highlight .compare-tag {
            background: rgba(255, 255, 255, 0.2);
            color: #FFFFFF;
        }
        .compare-list {
            list-style: none;
            padding: 0;
            margin: 0.75rem 0 0 0;
        }
        .compare-list li {
            padding: 0.42rem 0;
            font-size: 0.9rem;
            border-bottom: 1px dashed rgba(148, 163, 184, 0.28);
        }
        .compare-list li:last-child {
            border-bottom: none;
        }

        /* 2-Col FAQ / Methodology Cards */
        .method-band {
            background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
            border-radius: 22px;
            padding: 1.75rem;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
        }
        .method-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.95rem;
            margin-top: 1.1rem;
        }
        .method-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            color: #0F172A;
            position: relative;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);
        }
        .method-badge {
            position: absolute;
            top: 0.95rem;
            right: 1rem;
            background: #111111;
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.22rem 0.55rem;
            border-radius: 8px;
        }
        .method-card h4 {
            margin: 0 2.8rem 0.45rem 0;
            font-size: 0.98rem;
            font-weight: 700;
            color: #0F172A;
        }
        .method-card p {
            margin: 0;
            font-size: 0.86rem;
            line-height: 1.55;
            color: #475569;
        }

        /* Arabic RTL Containers */
        .arabic-rtl-box {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif;
            font-size: 1.16rem;
            line-height: 2.05;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-right: 5px solid #4F46E5;
            border-radius: 16px;
            padding: 1.15rem 1.4rem;
            color: #0F172A;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        .arabic-clean-box {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif;
            font-size: 1.02rem;
            line-height: 1.85;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.85rem 1.05rem;
            color: #1E293B;
        }
        .ar-pos {
            background-color: #DCFCE7;
            color: #166534;
            padding: 0.08rem 0.38rem;
            border-radius: 6px;
            font-weight: 600;
        }
        .ar-neg {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.08rem 0.38rem;
            border-radius: 6px;
        }
        textarea[aria-label*="artikel berita Arab"] {
            direction: rtl !important;
            text-align: right !important;
            unicode-bidi: plaintext !important;
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif !important;
            font-size: 1.1rem !important;
            line-height: 1.8 !important;
            background-color: #FFFFFF !important;
            border-radius: 14px !important;
        }

        /* Dark Showcase Footer */
        .dark-footer {
            background: #111318;
            color: #F8FAFC;
            border-radius: 22px;
            padding: 1.75rem 2rem;
            margin-top: 2.2rem;
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 1.5rem;
            align-items: center;
        }
        .dark-footer h3 {
            margin: 0 0 0.45rem 0;
            font-size: 1.25rem;
            font-weight: 800;
            color: #FFFFFF;
        }
        .dark-footer p {
            margin: 0;
            font-size: 0.88rem;
            color: #94A3B8;
            line-height: 1.55;
        }
        .footer-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            justify-content: flex-end;
        }
        .footer-btn {
            display: inline-block;
            background: #FFFFFF;
            color: #111111 !important;
            text-decoration: none !important;
            font-weight: 700;
            font-size: 0.82rem;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
        }
        .footer-btn.outline {
            background: rgba(255, 255, 255, 0.1);
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.22);
        }

        @media (max-width: 860px) {
            .hero-bento, .steps-grid, .compare-grid, .method-grid, .dark-footer {
                grid-template-columns: 1fr;
            }
            .footer-links {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_banner(corpus_stats: Dict[str, Any], eval_tables: Dict[str, Any]) -> None:
    svm_f = eval_tables.get("svm_frame_detail")
    svm_m = eval_tables.get("svm_medan_detail")
    total_valid = corpus_stats.get("total_valid_articles")

    valid_txt = f"{total_valid:,}" if total_valid else "—"
    frame_txt = f"{svm_f['accuracy'] * 100:.2f}%" if svm_f else "—"
    medan_txt = f"{svm_m['accuracy'] * 100:.2f}%" if svm_m else "—"

    st.markdown(
        f"""
        <div class="hero-bento">
            <div>
                <div class="hero-pill">Arabic NLP Research · 11 Kategori Berita</div>
                <h1 class="hero-title">Klasifikasi Berita Arab Berbasis Medan Makna &amp; Semantik <em>Frame</em></h1>
                <p class="hero-subtitle">
                    Demo interaktif klasifikasi artikel berita berbahasa Arab, ekstraksi kata inti Medan Makna
                    dan frasa kontekstual Semantik <em>Frame</em>, bobot kontribusi fitur, serta evaluasi 5 model pembelajaran mesin.
                </p>
            </div>
            <div class="hero-widget">
                <div class="hero-widget-label">Ringkasan Performa Sistem</div>
                <div class="hero-tiles">
                    <div class="hero-tile active">
                        <div class="hero-tile-val">{frame_txt}</div>
                        <div class="hero-tile-cap">Akurasi Semantik Frame</div>
                    </div>
                    <div class="hero-tile">
                        <div class="hero-tile-val">{medan_txt}</div>
                        <div class="hero-tile-cap">Akurasi Medan Makna</div>
                    </div>
                    <div class="hero-tile">
                        <div class="hero-tile-val">{valid_txt}</div>
                        <div class="hero-tile-cap">Artikel Valid Korpus</div>
                    </div>
                    <div class="hero-tile">
                        <div class="hero-tile-val">6 Portal</div>
                        <div class="hero-tile-cap">Timur Tengah &amp; Afrika Utara</div>
                    </div>
                </div>
            </div>
        </div>
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


def _render_footer() -> None:
    st.markdown(
        f"""
        <div class="dark-footer">
            <div>
                <h3>Arabic News Semantic Classification</h3>
                <p>
                    Dikembangkan oleh <strong>Maulana Ihsan Ahmad</strong> — Program Studi Magister Bahasa dan Sastra Arab,
                    UIN Sunan Kalijaga Yogyakarta. Dirilis di bawah lisensi MIT.
                </p>
            </div>
            <div class="footer-links">
                <a class="footer-btn" href="{JOURNAL_DOI_URL}" target="_blank" rel="noopener">Jurnal SiELE (2024)</a>
                <a class="footer-btn outline" href="{THESIS_REPO_URL}" target="_blank" rel="noopener">Tesis UIN</a>
                <a class="footer-btn outline" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener">GitHub Repo</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_classification_tab(
    bundles: Dict[str, Dict[str, Any]],
    stopwords: List[str],
    samples_df: pd.DataFrame,
    font_prop: Optional[fm.FontProperties],
) -> None:
    st.markdown(
        """
        <div class="steps-grid">
            <div class="step-card">
                <div class="step-badge">01</div>
                <div class="step-title">Masukkan Artikel Arab</div>
                <p class="step-desc">Tulis atau tempel berita Arab secara mandiri, atau pilih contoh berita dari 11 kategori.</p>
            </div>
            <div class="step-card">
                <div class="step-badge">02</div>
                <div class="step-title">Ekstraksi Ganda</div>
                <p class="step-desc">Sistem menyaring kata kunci Medan Makna dan frasa kontekstual Semantik <em>Frame</em> sekaligus.</p>
            </div>
            <div class="step-card">
                <div class="step-badge">03</div>
                <div class="step-title">Skor &amp; Interpretasi</div>
                <p class="step-desc">Lihat keyakinan prediksi lintas 11 kategori beserta sorotan kata pendorong (+) dan penekan (-).</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    with st.container(border=True):
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

    medan_res = dict(dual["medan_makna"])
    frame_res = dict(dual["frame_semantik"])
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", active_text))
    medan_res["is_uncategorized"] = bool(
        medan_res.get("is_uncategorized") or medan_res.get("n_active_features", 0) == 0 or (not has_arabic)
    )
    frame_res["is_uncategorized"] = bool(
        frame_res.get("is_uncategorized") or frame_res.get("n_active_features", 0) == 0 or (not has_arabic)
    )

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

    def _pred_metrics_html(sig_val: str, count_label: str, count_val: str, tfidf_val: str) -> str:
        return (
            '<div class="pred-metrics-grid">'
            f'<div class="pred-metric-chip"><div class="pred-metric-label">Skor Sigmoid</div><div class="pred-metric-val">{sig_val}</div></div>'
            f'<div class="pred-metric-chip"><div class="pred-metric-label">{count_label}</div><div class="pred-metric-val">{count_val}</div></div>'
            f'<div class="pred-metric-chip"><div class="pred-metric-label">Bobot TF-IDF</div><div class="pred-metric-val">{tfidf_val}</div></div>'
            "</div>"
        )

    c1, c2 = st.columns(2)
    with c1:
        if frame_res.get("is_uncategorized"):
            st.warning(
                "**Prediksi Semantik Frame**: `غير مصنف` — **Tidak Terkategori** *(Uncategorized — 0 frasa aktif)*"
            )
            st.markdown(_pred_metrics_html("—", "Frasa Aktif", "0", "0.0000"), unsafe_allow_html=True)
        else:
            top_prob_f = frame_res["category_scores"][0]["sigmoid_percent"]
            st.success(
                f"**Prediksi Semantik Frame**: `{frame_res['predicted_category']}` "
                f"— **{frame_res['predicted_category_id']}** ({frame_res['predicted_category_en']})"
            )
            st.markdown(
                _pred_metrics_html(
                    f"{top_prob_f:.2f}%",
                    "Frasa Aktif",
                    f"{frame_res['n_active_features']}",
                    f"{frame_res['total_tfidf_weight']:.4f}",
                ),
                unsafe_allow_html=True,
            )

    with c2:
        if medan_res.get("is_uncategorized"):
            st.warning(
                "**Prediksi Medan Makna**: `غير مصنف` — **Tidak Terkategori** *(Uncategorized — 0 kata aktif)*"
            )
            st.markdown(_pred_metrics_html("—", "Kata Aktif", "0", "0.0000"), unsafe_allow_html=True)
        else:
            top_prob_m = medan_res["category_scores"][0]["sigmoid_percent"]
            st.info(
                f"**Prediksi Medan Makna**: `{medan_res['predicted_category']}` "
                f"— **{medan_res['predicted_category_id']}** ({medan_res['predicted_category_en']})"
            )
            st.markdown(
                _pred_metrics_html(
                    f"{top_prob_m:.2f}%",
                    "Kata Aktif",
                    f"{medan_res['n_active_features']}",
                    f"{medan_res['total_tfidf_weight']:.4f}",
                ),
                unsafe_allow_html=True,
            )

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
        (score_col1, "Semantik Frame", frame_res, INDIGO_PRIMARY),
        (score_col2, "Medan Makna", medan_res, TEAL_SECONDARY),
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
                    color=INDIGO_PRIMARY,
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
            bars_cat = ax_cat.barh(labels_cat, vals_cat, color=INDIGO_PRIMARY, height=0.68)
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
            palette = ["#4F46E5", "#2563EB", "#0D9488", "#0EA5E9", "#6366F1", "#14B8A6"]
            wedges, _, _ = ax_src.pie(
                src_df["Jumlah Artikel"],
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 4.0 else "",
                pctdistance=0.76,
                startangle=140,
                colors=palette[: len(src_df)],
                wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.5},
                textprops={"color": "#FFFFFF", "fontweight": "bold", "fontsize": 8.5},
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
        <div class="compare-grid">
            <div class="compare-card-muted">
                <span class="compare-tag">Baseline Leksikal</span>
                <h4 style="margin:0 0 0.2rem 0; font-size:1.1rem; font-weight:800;">Medan Makna (Tanpa Semantik <em>Frame</em>)</h4>
                <ul class="compare-list">
                    <li>• Representasi kata leksikal inti (penghapusan kata henti)</li>
                    <li>• Pembagian data 80% Latih / 15% Validasi (N={svm_medan['support']:,}) / 5% Uji</li>
                    <li>• Akurasi SVM: <strong>{svm_medan['accuracy'] * 100:.2f}%</strong></li>
                    <li>• Kebutuhan memori sangat ringan dan inferensi cepat</li>
                </ul>
            </div>
            <div class="compare-card-highlight">
                <span class="compare-tag">Pendekatan Unggulan</span>
                <h4 style="margin:0 0 0.2rem 0; font-size:1.1rem; font-weight:800; color:#FFFFFF;">Dengan Semantik <em>Frame</em></h4>
                <ul class="compare-list">
                    <li>✓ Frasa kontekstual 2–3 kata dengan preposisi Arab dipertahankan</li>
                    <li>✓ Pembagian data 85% Latih / 10% Validasi (N={svm_frame['support']:,}) / 5% Uji</li>
                    <li>✓ Akurasi SVM Tertinggi: <strong>{svm_frame['accuracy'] * 100:.2f}%</strong> (F1 = {svm_frame['f1_score']:.4f})</li>
                    <li>✓ Menangkap hubungan sintaksis-semantik antar-konsep berita</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
    b1 = axes[0].bar(x - w / 2, perf_df["Akurasi Frame"] * 100, w, label="Semantik Frame", color=INDIGO_PRIMARY)
    b2 = axes[0].bar(x + w / 2, perf_df["Akurasi Medan"] * 100, w, label="Medan Makna", color=TEAL_SECONDARY)
    axes[0].set_ylim(max(0.0, min_acc - 5.0), 96.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[0].set_title("Akurasi (%)", fontsize=11, pad=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x - w / 2, eff_df["Waktu Frame (s)"], w, color=INDIGO_PRIMARY)
    axes[1].bar(x + w / 2, eff_df["Waktu Medan (s)"], w, color=TEAL_SECONDARY)
    max_t = float(max(eff_df["Waktu Frame (s)"].max(), eff_df["Waktu Medan (s)"].max()))
    axes[1].set_ylim(0, max_t * 1.15 if max_t else 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[1].set_title("Waktu Evaluasi (detik)", fontsize=11, pad=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    axes[2].bar(x - w / 2, eff_df["RAM Frame (MB)"], w, color=INDIGO_PRIMARY)
    axes[2].bar(x + w / 2, eff_df["RAM Medan (MB)"], w, color=TEAL_SECONDARY)
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

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Akurasi SVM", f"{active_svm['accuracy'] * 100:.2f}%")
    s2.metric("Total Sampel", f"{active_svm['support']:,}")
    s3.metric("Precision", f"{active_svm['precision']:.4f}")
    s4.metric("F1-Score", f"{active_svm['f1_score']:.4f}")

    cm_col1, cm_col2 = st.columns([1.5, 1])
    with cm_col1:
        labels_ar = [reshape_arabic_for_plot(LABELS[i]) for i in range(1, 12)]
        fig_cm, ax_cm = plt.subplots(figsize=(9.8, 8.4))
        sns.heatmap(
            active_svm["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Purples",
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
        best_row = per_class_df.sort_values("F1-Score", ascending=False).iloc[0]
        low_row = per_class_df.sort_values("F1-Score", ascending=True).iloc[0]
        st.markdown("##### Ringkasan Performa Kategori")
        st.success(
            f"**F1 Tertinggi**: {best_row['Kategori']} ({best_row['Arab']}) — `{best_row['F1-Score']:.4f}`"
        )
        st.info(
            f"**F1 Terendah**: {low_row['Kategori']} ({low_row['Arab']}) — `{low_row['F1-Score']:.4f}`"
        )
        st.caption(
            "Nilai diagonal pada matriks menunjukkan jumlah prediksi benar per kategori, "
            "sedangkan angka di luar diagonal menunjukkan jumlah artikel yang tertukar antarkelas."
        )

    st.markdown("##### Rincian Metrik Per-Kategori")
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
    st.markdown(
        """
        <div class="method-band">
            <h3 style="margin:0 0 0.35rem 0; font-size:1.35rem; font-weight:800; color:#FFFFFF;">
                Metodologi Penelitian &amp; Arsitektur Sistem
            </h3>
            <p style="margin:0; font-size:0.9rem; color:rgba(255,255,255,0.88);">
                Seluruh eksperimen direproduksi dari sumber data tunggal <code>data/basic_articles_dataset.csv</code> dan <code>data/stopwords.txt</code>.
            </p>
            <div class="method-grid">
                <div class="method-card">
                    <span class="method-badge">01</span>
                    <h4>Pengumpulan Data (Web Scraping)</h4>
                    <p>Akuisisi otomatis dari 6 portal berita daring Arab (<code>alsumaria.tv</code>, <code>al-sharq.com</code>, <code>assahraa.ma</code>, <code>akhbarelsyaha.com</code>, <code>sarayanews.com</code>, <code>addustour.com</code>) pada 11 kategori tematik.</p>
                </div>
                <div class="method-card">
                    <span class="method-badge">02</span>
                    <h4>Pra-pemrosesan &amp; Percabangan</h4>
                    <p>Pembersihan angka, tanda baca Arab &amp; ASCII, konversi huruf kecil, lalu bercabang menjadi jalur <strong>Medan Makna</strong> (hapus stopwords, 80/15/5) dan <strong>Semantik Frame</strong> (pertahankan preposisi Arab, 85/10/5).</p>
                </div>
                <div class="method-card">
                    <span class="method-badge">03</span>
                    <h4>Statistik Korpus &amp; Leksikon</h4>
                    <p>Menghitung total token sebelum dan sesudah penghapusan kata henti, kosakata unik Arab vs Non-Arab, serta menyusun Top-1000 kata inti dan frasa TF-IDF per kategori.</p>
                </div>
                <div class="method-card">
                    <span class="method-badge">04</span>
                    <h4>Pelatihan 5 Model &amp; Interpretasi</h4>
                    <p>Evaluasi efektivitas &amp; efisiensi komputasi via <code>psutil</code> pada <em>SVM, Logistic Regression, Naïve Bayes, Random Forest, dan Gradient Boosting</em>, serta pemetaan skor <strong>Sigmoid</strong> pada model SVM.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Klasifikasi Artikel Berita Arab — Medan Makna & Semantik Frame",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_design_system_styles()

    bundles = get_cached_bundles()
    stopwords = get_cached_stopwords()
    samples_df = get_cached_sample_articles()
    corpus_stats = get_cached_corpus_stats()
    eval_tables = get_cached_evaluation_tables()
    font_prop = _get_font_prop()

    _render_sidebar(corpus_stats, eval_tables)
    _render_hero_banner(corpus_stats, eval_tables)

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

    _render_footer()


if __name__ == "__main__":
    main()
