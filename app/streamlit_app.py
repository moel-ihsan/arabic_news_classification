"""Interactive Streamlit demo for Arabic News Classification (Semantic Fields & Frame Semantics)."""

import html
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
import unicodedata

# Resolve repository root dynamically so the app runs from any working directory
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
ASSETS_BRANDING_DIR = REPO_ROOT / "assets" / "branding"
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
    LABELS_EN,
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

VIOLET_PRIMARY = "#7B2CFF"
ORANGE_SECONDARY = "#F97316"
LIME_ACCENT = "#A6F40D"
INK_DARK = "#151618"
CHART_SURFACE = "#FAF9F5"

COUNTRY_EN_MAP = {
    "Irak": "Iraq",
    "Qatar": "Qatar",
    "Maroko": "Morocco",
    "Mesir": "Egypt",
    "Yordania": "Jordan",
}


def _apply_chart_theme(ax: Any, grid_axis: str = "x") -> None:
    """Harmonize chart canvas, spines, ticks, and grid with the Warm Alabaster palette."""
    ax.figure.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDD8CA")
    ax.spines["bottom"].set_color("#DDD8CA")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors="#575860")
    ax.title.set_color(INK_DARK)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_color("#575860")
    ax.yaxis.label.set_color("#575860")
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.7, color="#D6D1C4", alpha=0.75)
    ax.set_axisbelow(True)


MODELS_EVAL_CONFIG = [
    ("SVC", "SVM", "svm.txt", "SVM"),
    ("LogisticRegression", "Logistic Regression", "logistic.txt", "LogReg"),
    ("MultinomialNB", "Naïve Bayes", "naive.txt", "Naïve Bayes"),
    ("RandomForest", "Random Forest", "forest.txt", "Random Forest"),
    ("GradientBoosting", "Gradient Boosting", "gradient.txt", "GradBoost"),
]


def load_evaluation_summary_tables(metrics_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load effectiveness, efficiency, per-class metrics, and confusion matrices from results/metrics/."""
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
                "Frame Time (s)": round(mf["execution_time_seconds"], 4),
                "Field Time (s)": round(mn["execution_time_seconds"], 4),
                "Frame RAM (MB)": rf["max_memory_mb"],
                "Field RAM (MB)": rn["max_memory_mb"],
                "RAM Increase (%)": mem_inc_pct,
                "Frame CPU (%)": rf["avg_cpu_percent"],
                "Field CPU (%)": rn["avg_cpu_percent"],
            }
        )
        perf_rows.append(
            {
                "Model": label_name,
                "Frame Accuracy": mf["accuracy"],
                "Frame Prec": mf["precision"],
                "Frame Rec": mf["recall"],
                "Frame F1": mf["f1_score"],
                "Frame Support": mf["support"],
                "Field Accuracy": mn["accuracy"],
                "Field Prec": mn["precision"],
                "Field Rec": mn["recall"],
                "Field F1": mn["f1_score"],
                "Field Support": mn["support"],
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
    """Load Frame Semantics TF-IDF lexicons and highest-frequency word/phrase lists for one Arabic category."""
    norm_cat = unicodedata.normalize("NFC", category_name)
    l_dir = Path(lexicons_dir) if lexicons_dir else LEXICONS_DIR
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR

    try:
        frame_df = (
            pd.read_csv(find_file_unicode_safe(l_dir / "tfidf_top1000", f"ngram_tfidf_weights_{norm_cat}", ".csv"))
            .head(top_k)
            .rename(columns={"N-gram": "Frame Semantics Phrase", "TF-IDF Weight": "TF-IDF Weight"})
        )
    except FileNotFoundError:
        frame_df = pd.DataFrame()

    word_rows = []
    try:
        words_path = find_file_unicode_safe(s_dir / "nonstop_words", f"nonstop_words_{norm_cat}", ".txt")
        for line in words_path.read_text(encoding="utf-8").splitlines()[1 : top_k + 1]:
            if ":" in line:
                word, freq = line.split(":", 1)
                word_rows.append({"Word": word.strip(), "Frequency": int(freq.strip())})
    except FileNotFoundError:
        pass

    try:
        phrase2_df = (
            pd.read_csv(find_file_unicode_safe(l_dir / "bigram_top1000", f"bigram_{norm_cat}", ".csv"))
            .head(top_k)
            .rename(columns={"Bigram N-gram": "2-Word Phrase", "Frequency": "Frequency"})
        )
    except FileNotFoundError:
        phrase2_df = pd.DataFrame()

    try:
        phrase3_df = (
            pd.read_csv(find_file_unicode_safe(l_dir / "trigram_top1000", f"trigram_{norm_cat}", ".csv"))
            .head(top_k)
            .rename(columns={"Trigram N-gram": "3-Word Phrase", "Frequency": "Frequency"})
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
    """Load the 11 representative test articles exported by Notebook 02 from results/stats/."""
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR
    sample_path = s_dir / "sample_11_test_articles.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return pd.DataFrame(columns=["label", "category", "filename", "source", "text"])


def load_dashboard_corpus_stats(stats_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load precomputed corpus summary statistics from results/stats/ without heavy startup overhead."""
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
    """Render RTL HTML highlighting positive (+) coefficients in green and negative (-) coefficients in red."""
    pos_set = set(positive_terms)
    neg_set = set(negative_terms)
    tokens = str(text).split()
    rendered: List[str] = []
    for raw_tok in tokens:
        cleaned_tok = preprocess_text(raw_tok, stopwords=None)
        safe_tok = html.escape(raw_tok)
        if cleaned_tok in pos_set:
            rendered.append(f'<span class="ar-pos" title="Positive Coefficient (+)">{safe_tok}</span>')
        elif cleaned_tok in neg_set:
            rendered.append(f'<span class="ar-neg" title="Negative Coefficient (-)">{safe_tok}</span>')
        else:
            rendered.append(safe_tok)
    return " ".join(rendered)


@st.cache_resource(show_spinner="Loading Semantic Field & Frame Semantics models...")
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
        @import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700,900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Amiri:wght@400;700&display=swap');

        :root {
            --font: 'Satoshi', 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            --gdg-font-family: 'Satoshi', 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            --ink-dark: #151618;
            --ink-muted: #575860;
            --surface-bg: #F4F1EA;
            --surface-card: #FAF9F5;
            --surface-warm: #EFECE3;
            --border-hairline: #E3DFD3;
            --accent-violet: #7B2CFF;
            --accent-lilac: #A688FF;
            --accent-orange: #FF7A1A;
            --accent-lime: #A6F40D;
        }

        /* Sembunyikan Header, Toolbar, dan Dekorasi Bawaan Streamlit */
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        /* Tipografi Non-Arab: Satoshi / DM Sans (menimpa Source Sans bawaan Streamlit) */
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"] p,
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"],
        [data-baseweb="select"],
        [role="tab"], [role="tab"] p,
        button, button p, input, label,
        .top-nav, .hero-bento, .pred-card, .compare-grid, .method-band, .dark-footer {
            font-family: 'Satoshi', 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5 {
            color: var(--ink-dark);
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
        }

        .stApp {
            background-color: var(--surface-bg);
            color: var(--ink-dark);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.05rem !important;
            padding-bottom: 2.5rem;
        }

        /* Minimalist Top Navbar (Warm Ivory Pill + Acid Lime Marker + Charcoal CTA) */
        .top-nav {
            background: var(--surface-card);
            border: 1px solid var(--border-hairline);
            border-radius: 9999px;
            padding: 0.45rem 0.55rem 0.45rem 1.15rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .top-nav-brand {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 700;
            font-size: 0.92rem;
            letter-spacing: -0.02em;
            color: var(--ink-dark);
            white-space: nowrap;
        }
        .top-nav-dot {
            width: 10px;
            height: 10px;
            border-radius: 9999px;
            background: var(--accent-lime);
            border: 1.5px solid var(--ink-dark);
            display: inline-block;
            flex-shrink: 0;
        }
        .top-nav-meta {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--ink-muted);
            white-space: nowrap;
        }
        .top-nav-actions {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            flex-shrink: 0;
        }
        .top-nav-link {
            display: inline-block;
            text-decoration: none !important;
            font-size: 0.79rem;
            font-weight: 600;
            color: var(--ink-dark) !important;
            background: var(--surface-bg);
            border: 1px solid var(--border-hairline);
            padding: 0.38rem 0.85rem;
            border-radius: 9999px;
            white-space: nowrap;
            transition: background-color 0.15s ease;
        }
        .top-nav-link:hover {
            background: var(--surface-warm);
        }
        .top-nav-link.primary {
            background: var(--ink-dark);
            color: #FFFFFF !important;
            border-color: var(--ink-dark);
        }
        .top-nav-link.primary:hover {
            background: var(--accent-violet);
            border-color: var(--accent-violet);
        }

        /* Deep Charcoal Architectural Hero Card + 4 Accent Bento Tiles */
        .hero-bento {
            background: var(--ink-dark);
            border-radius: 22px;
            padding: 1.8rem 2.05rem;
            color: #FAF9F5;
            margin-bottom: 1.15rem;
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 1.65rem;
            align-items: center;
        }
        .hero-title {
            font-size: 1.98rem;
            font-weight: 700;
            line-height: 1.14;
            letter-spacing: -0.032em;
            margin: 0 0 0.7rem 0;
            color: #FFFFFF !important;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            font-weight: 400;
            line-height: 1.58;
            color: rgba(250, 249, 245, 0.78);
            margin: 0;
            max-width: 54ch;
        }
        .hero-widget {
            background: var(--surface-card);
            border-radius: 18px;
            padding: 1rem;
            color: var(--ink-dark);
        }
        .hero-widget-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--ink-muted);
            margin-bottom: 0.68rem;
        }
        .hero-tiles {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.55rem;
        }
        .hero-tile {
            background: var(--surface-warm);
            border-radius: 14px;
            padding: 0.82rem 0.9rem;
            color: var(--ink-dark);
        }
        .hero-tile.lime {
            background: var(--accent-lime);
            color: var(--ink-dark);
        }
        .hero-tile.violet {
            background: var(--accent-violet);
            color: #FFFFFF;
        }
        .hero-tile.orange {
            background: var(--accent-orange);
            color: var(--ink-dark);
        }
        .hero-tile-val {
            font-size: 1.32rem;
            font-weight: 700;
            line-height: 1.12;
            letter-spacing: -0.025em;
            font-variant-numeric: tabular-nums;
        }
        .hero-tile-cap {
            font-size: 0.75rem;
            margin-top: 0.24rem;
            font-weight: 600;
            color: var(--ink-muted);
        }
        .hero-tile.lime .hero-tile-cap,
        .hero-tile.orange .hero-tile-cap {
            color: rgba(21, 22, 24, 0.82);
        }
        .hero-tile.violet .hero-tile-cap {
            color: rgba(255, 255, 255, 0.88);
        }

        /* Connected Horizontal Workflow Steps (01 -> 02 -> 03 dengan Triad Aksen) */
        .flow-strip {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 44px minmax(0, 1fr) 44px minmax(0, 1fr);
            align-items: start;
            gap: 0.65rem;
            background: var(--surface-card);
            border: 1px solid var(--border-hairline);
            border-radius: 18px;
            padding: 1.05rem 1.35rem;
            margin-bottom: 1rem;
        }
        .flow-item {
            min-width: 0;
        }
        .flow-num {
            display: inline-block;
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.1;
            letter-spacing: -0.02em;
            padding: 0.26rem 0.62rem;
            border-radius: 9px;
            font-variant-numeric: tabular-nums;
            margin-bottom: 0.42rem;
        }
        .flow-num.violet {
            background: var(--accent-violet);
            color: #FFFFFF;
        }
        .flow-num.orange {
            background: var(--accent-orange);
            color: var(--ink-dark);
        }
        .flow-num.lime {
            background: var(--accent-lime);
            color: var(--ink-dark);
        }
        .flow-title {
            font-size: 0.93rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--ink-dark);
            line-height: 1.25;
            margin-bottom: 0.16rem;
        }
        .flow-desc {
            font-size: 0.8rem;
            font-weight: 400;
            line-height: 1.42;
            color: var(--ink-muted);
        }
        .flow-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            padding-top: 0.55rem;
        }

        /* Pill Tabs (cocok dengan div[role="tab"] Streamlit 1.50) */
        .stTabs [role="tablist"] {
            background-color: var(--surface-card) !important;
            padding: 6px 8px !important;
            border-radius: 9999px !important;
            border: 1px solid var(--border-hairline) !important;
            box-shadow: none !important;
            gap: 4px !important;
            margin-bottom: 1.05rem !important;
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
            padding: 0.5rem 0.88rem !important;
            min-height: 38px !important;
            height: auto !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            color: var(--ink-muted) !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .stTabs [role="tablist"] [role="tab"] p {
            margin: 0 !important;
            font-weight: 600 !important;
        }
        .stTabs [role="tablist"] [role="tab"][aria-selected="true"] {
            background-color: var(--ink-dark) !important;
            color: #FFFFFF !important;
            border-radius: 9999px !important;
        }
        .stTabs [role="tablist"] [role="tab"][aria-selected="true"] p {
            color: #FFFFFF !important;
        }

        /* Unified Bento Prediction Cards */
        .pred-card {
            background: var(--surface-card);
            border: 1px solid var(--border-hairline);
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            margin-top: 0.35rem;
            margin-bottom: 1.1rem;
        }
        .pred-card.frame {
            border-color: #D7C6FF;
        }
        .pred-card.medan {
            border-color: #FFD0AD;
        }
        .pred-card.uncat {
            background: #FEF9E7;
            border-color: #F4D03F;
        }
        .pred-card-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            margin-bottom: 0.55rem;
        }
        .pred-scheme-tag {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.24rem 0.68rem;
            border-radius: 9999px;
        }
        .pred-card.frame .pred-scheme-tag {
            background: var(--accent-violet);
            color: #FFFFFF;
        }
        .pred-card.medan .pred-scheme-tag {
            background: var(--accent-orange);
            color: var(--ink-dark);
        }
        .pred-card.uncat .pred-scheme-tag {
            background: var(--accent-lime);
            color: var(--ink-dark);
        }
        .pred-cat-ar {
            font-family: var(--font-arabic) !important;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--ink-dark);
            direction: rtl;
        }
        .pred-cat-main {
            font-size: 1.32rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--ink-dark);
            margin-bottom: 0.85rem;
            line-height: 1.25;
        }
        .pred-cat-en {
            font-weight: 500;
            font-size: 0.95rem;
            color: var(--ink-muted);
        }
        .pred-stats-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            padding-top: 0.78rem;
            border-top: 1px solid var(--border-hairline);
        }
        .pred-stat-label {
            font-size: 0.74rem;
            font-weight: 500;
            color: var(--ink-muted);
            margin-bottom: 0.16rem;
            white-space: nowrap;
        }
        .pred-stat-val {
            font-size: 1.16rem;
            font-weight: 700;
            color: var(--ink-dark);
            line-height: 1.2;
            letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums;
        }

        /* Clean Streamlit Input Container */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--surface-card) !important;
            border: 1px solid var(--border-hairline) !important;
            border-radius: 18px !important;
            box-shadow: none !important;
        }

        /* Elevated Pill CTA Buttons */
        .stButton > button {
            border-radius: 9999px !important;
            font-weight: 600 !important;
            padding: 0.58rem 1.35rem !important;
            transition: background-color 0.15s ease, transform 0.12s ease !important;
        }
        .stButton > button[kind="primary"] {
            background-color: var(--ink-dark) !important;
            border-color: var(--ink-dark) !important;
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"] p {
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: var(--accent-violet) !important;
            border-color: var(--accent-violet) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background-color: var(--surface-card) !important;
            border: 1px solid var(--border-hairline) !important;
            color: var(--ink-dark) !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--surface-warm) !important;
        }

        /* Flat Hairline Metric Cards */
        [data-testid="stMetric"] {
            background-color: var(--surface-card);
            border: 1px solid var(--border-hairline);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            box-shadow: none;
        }
        [data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
        }

        /* Side-by-Side Comparison Bento Cards */
        .compare-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 0.6rem 0 1.35rem 0;
        }
        .compare-card-muted {
            background: var(--surface-card);
            border: 1px solid var(--border-hairline);
            border-radius: 18px;
            padding: 1.35rem 1.5rem;
            color: var(--ink-dark);
        }
        .compare-card-highlight {
            background: var(--accent-violet);
            border-radius: 18px;
            padding: 1.35rem 1.5rem;
            color: #FFFFFF;
        }
        .compare-tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.24rem 0.68rem;
            border-radius: 9999px;
            margin-bottom: 0.55rem;
        }
        .compare-card-muted .compare-tag {
            background: var(--accent-orange);
            color: var(--ink-dark);
        }
        .compare-card-highlight .compare-tag {
            background: var(--accent-lime);
            color: var(--ink-dark);
        }
        .compare-card-highlight .compare-title,
        [data-testid="stMarkdownContainer"] .compare-card-highlight h4 {
            color: #FFFFFF !important;
            margin: 0 0 0.2rem 0 !important;
            font-size: 1.12rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
        }
        .compare-card-muted .compare-title {
            color: var(--ink-dark) !important;
            margin: 0 0 0.2rem 0 !important;
            font-size: 1.12rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
        }
        .compare-list {
            list-style: none;
            padding: 0;
            margin: 0.75rem 0 0 0;
        }
        .compare-list li {
            padding: 0.44rem 0;
            font-size: 0.9rem;
            border-bottom: 1px dashed rgba(21, 22, 24, 0.14);
        }
        .compare-card-highlight .compare-list li {
            border-bottom: 1px dashed rgba(255, 255, 255, 0.24);
        }
        .compare-list li:last-child {
            border-bottom: none;
        }

        /* 2-Col Methodology Cards (Dark Charcoal Showcase Band + Warm Ivory Cards) */
        .method-band {
            background: var(--ink-dark);
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
            background: var(--surface-card);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            color: var(--ink-dark);
            position: relative;
        }
        .method-badge {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: var(--ink-dark);
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.24rem 0.58rem;
            border-radius: 8px;
            font-variant-numeric: tabular-nums;
        }
        .method-badge.violet {
            background: var(--accent-violet);
            color: #FFFFFF;
        }
        .method-badge.orange {
            background: var(--accent-orange);
            color: var(--ink-dark);
        }
        .method-badge.lime {
            background: var(--accent-lime);
            color: var(--ink-dark);
        }
        .method-card h4 {
            margin: 0 2.8rem 0.45rem 0;
            font-size: 0.98rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--ink-dark);
        }
        .method-card p {
            margin: 0;
            font-size: 0.86rem;
            line-height: 1.55;
            color: var(--ink-muted);
        }

        /* Arabic RTL Containers (dilindungi dari override font Latin) */
        .arabic-rtl-box, .arabic-rtl-box * {
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif !important;
        }
        .arabic-rtl-box {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-size: 1.18rem;
            line-height: 2.05;
            background-color: var(--surface-card);
            border: 1px solid var(--border-hairline);
            border-radius: 16px;
            padding: 1.15rem 1.4rem;
            color: var(--ink-dark);
        }
        .arabic-clean-box, .arabic-clean-box * {
            font-family: 'Amiri', 'Traditional Arabic', 'Geeza Pro', 'Arial', sans-serif !important;
        }
        .arabic-clean-box {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-size: 1.04rem;
            line-height: 1.85;
            background-color: var(--surface-warm);
            border: 1px solid var(--border-hairline);
            border-radius: 12px;
            padding: 0.85rem 1.05rem;
            color: var(--ink-dark);
        }
        .ar-pos {
            background-color: #DCFCE7;
            color: #166534;
            padding: 0.08rem 0.38rem;
            border-radius: 6px;
            font-weight: 700;
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
            font-size: 1.12rem !important;
            line-height: 1.85 !important;
            background-color: var(--surface-card) !important;
            border-radius: 14px !important;
        }

        /* Deep Charcoal Footer + Acid Lime Primary Action */
        .dark-footer {
            background: var(--ink-dark);
            color: #FAF9F5;
            border-radius: 22px;
            padding: 1.75rem 2rem;
            margin-top: 2rem;
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 1.5rem;
            align-items: center;
        }
        .dark-footer h3 {
            margin: 0 0 0.45rem 0;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #FFFFFF !important;
        }
        .dark-footer p {
            margin: 0;
            font-size: 0.88rem;
            color: rgba(250, 249, 245, 0.72);
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
            background: var(--accent-lime);
            color: var(--ink-dark) !important;
            text-decoration: none !important;
            font-weight: 700;
            font-size: 0.82rem;
            padding: 0.52rem 1.05rem;
            border-radius: 9999px;
        }
        .footer-btn.outline {
            background: rgba(250, 249, 245, 0.10);
            color: #FFFFFF !important;
            border: 1px solid rgba(250, 249, 245, 0.22);
        }

        @media (max-width: 720px) {
            .top-nav {
                border-radius: 18px;
                flex-wrap: wrap;
                padding: 0.75rem 1rem;
            }
            .top-nav-meta {
                display: none;
            }
            .hero-bento, .compare-grid, .method-grid, .dark-footer, .flow-strip {
                grid-template-columns: 1fr;
            }
            .flow-arrow {
                display: none;
            }
            .footer-links {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_top_navbar() -> None:
    nav_logo_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="76 95 360 360" '
        'width="22" height="22" aria-hidden="true" style="flex-shrink:0; display:block;">'
        '<path d="M138 262 C138 342 190 382 256 382 C322 382 374 336 374 236 L374 196" '
        'fill="none" stroke="#7B2CFF" stroke-width="46" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="252" cy="176" r="30" fill="#FF7A1A"/></svg>'
    )
    st.markdown(
        f"""
        <div class="top-nav">
            <div class="top-nav-brand">
                {nav_logo_svg}
                <span>Arabic News Semantics</span>
            </div>
            <div class="top-nav-meta">11 Categories &middot; 6 Arabic News Portals</div>
            <div class="top-nav-actions">
                <a class="top-nav-link" href="{JOURNAL_DOI_URL}" target="_blank" rel="noopener">SiELE Journal</a>
                <a class="top-nav-link" href="{THESIS_REPO_URL}" target="_blank" rel="noopener">Thesis</a>
                <a class="top-nav-link primary" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener">GitHub</a>
            </div>
        </div>
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
                <h1 class="hero-title">Arabic News Classification via Semantic Fields &amp; Frame Semantics</h1>
                <p class="hero-subtitle">
                    Classify Arabic news articles across 11 thematic categories by extracting core
                    Semantic Field keywords and contextual Frame Semantics phrases.
                </p>
            </div>
            <div class="hero-widget">
                <div class="hero-widget-label">System Performance Summary</div>
                <div class="hero-tiles">
                    <div class="hero-tile lime">
                        <div class="hero-tile-val">{frame_txt}</div>
                        <div class="hero-tile-cap">Frame Semantics Accuracy</div>
                    </div>
                    <div class="hero-tile violet">
                        <div class="hero-tile-val">{medan_txt}</div>
                        <div class="hero-tile-cap">Semantic Field Accuracy</div>
                    </div>
                    <div class="hero-tile orange">
                        <div class="hero-tile-val">{valid_txt}</div>
                        <div class="hero-tile-cap">Valid Corpus Articles</div>
                    </div>
                    <div class="hero-tile">
                        <div class="hero-tile-val">6 Portals</div>
                        <div class="hero-tile-cap">Middle East &amp; North Africa</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    footer_logo_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="76 95 360 360" '
        'width="22" height="22" aria-hidden="true" style="flex-shrink:0;">'
        '<path d="M138 262 C138 342 190 382 256 382 C322 382 374 336 374 236 L374 196" '
        'fill="none" stroke="#FFFFFF" stroke-width="46" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="252" cy="176" r="30" fill="#A6F40D"/></svg>'
    )
    st.markdown(
        f"""
        <div class="dark-footer">
            <div>
                <h3 style="display:flex; align-items:center; gap:0.55rem;">
                    {footer_logo_svg}
                    <span>Arabic News Semantics</span>
                </h3>
                <p>
                    Developed by <strong>Maulana Ihsan Ahmad</strong> — Master's Program in Arabic Language and Literature,
                    UIN Sunan Kalijaga Yogyakarta.
                </p>
            </div>
            <div class="footer-links">
                <a class="footer-btn" href="{JOURNAL_DOI_URL}" target="_blank" rel="noopener">SiELE Journal</a>
                <a class="footer-btn outline" href="{THESIS_REPO_URL}" target="_blank" rel="noopener">Thesis</a>
                <a class="footer-btn outline" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener">GitHub</a>
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
    arrow_svg = (
        '<div class="flow-arrow" aria-hidden="true">'
        '<svg width="34" height="12" viewBox="0 0 34 12" fill="none">'
        '<line x1="0" y1="6" x2="31" y2="6" stroke="#C7C1B2" stroke-width="1.6"/>'
        '<path d="M26.5 1.5L32 6L26.5 10.5" stroke="#151618" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
    )
    st.markdown(
        f"""
        <div class="flow-strip">
            <div class="flow-item">
                <div class="flow-num violet">01</div>
                <div class="flow-title">Input Arabic Article</div>
                <div class="flow-desc">Write, paste, or select a sample across 11 news categories.</div>
            </div>
            {arrow_svg}
            <div class="flow-item">
                <div class="flow-num orange">02</div>
                <div class="flow-title">Dual Extraction</div>
                <div class="flow-desc">Extract core Semantic Field terms &amp; Frame Semantics phrases.</div>
            </div>
            {arrow_svg}
            <div class="flow-item">
                <div class="flow-num lime">03</div>
                <div class="flow-title">Scores &amp; Weights</div>
                <div class="flow-desc">Inspect 11-class Sigmoid probabilities &amp; feature contributions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "medan_makna" not in bundles or "frame_semantik" not in bundles:
        st.warning(
            "Trained models are not yet available in `models/`. "
            "Run `notebooks/04_model_training_and_evaluation.ipynb` first to train and save "
            "`svm_medan_makna.joblib` and `svm_frame_semantik.joblib`."
        )
        return

    CUSTOM_INPUT_LABEL = "Write / Paste Arabic Text"
    sample_options: List[str] = [CUSTOM_INPUT_LABEL]
    label_by_option: Dict[str, int] = {}
    if not samples_df.empty:
        for _, r in samples_df.sort_values("label").iterrows():
            cid = int(r["label"])
            opt_lbl = f"Sample {cid}: {LABELS_EN[cid]} ({LABELS[cid]})"
            sample_options.append(opt_lbl)
            label_by_option[opt_lbl] = cid
    else:
        st.caption(
            "Representative test samples have not been generated (`results/stats/sample_11_test_articles.csv`). "
            "You can still paste an Arabic news article directly in the input box below."
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
                "Article Source:",
                sample_options,
                key="sample_selector",
            )
        with col_sel2:
            top_n_features = st.slider("Top Features Displayed", min_value=5, max_value=30, value=15)

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
            "Arabic news article text (Right-to-Left support):",
            key="article_input",
            height=190,
            placeholder="الصق أو اكتب نص مقال إخباري باللغة العربية هنا... (Paste or type an Arabic news article here)",
        )

        btn_col1, btn_col2, _ = st.columns([1.6, 1, 2.4])
        with btn_col1:
            classify_clicked = st.button("Classify & Extract Features", type="primary", width="stretch")
        with btn_col2:
            clear_clicked = st.button("Clear Text", width="stretch")

    if clear_clicked:
        st.session_state["pending_clear"] = True
        st.rerun()

    if classify_clicked:
        if not st.session_state["article_input"].strip():
            st.warning("Please enter or paste an Arabic news article before clicking Classify.")
            return
        st.session_state["submitted_text"] = st.session_state["article_input"]

    active_text = st.session_state.get("submitted_text", "").strip()
    if not active_text:
        st.info(
            "Enter or paste an Arabic news article and click **Classify & Extract Features**, "
            "or pick a sample category from **Article Source** above."
        )
        return

    with st.spinner("Extracting Semantic Field terms, Frame Semantics phrases, and model weights..."):
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
            f"Reference Test Label: **{LABELS[true_label_info]} ({LABELS_EN[true_label_info]})**"
        )

    both_uncat = frame_res.get("is_uncategorized") and medan_res.get("is_uncategorized")
    if both_uncat:
        st.warning(
            "**Uncategorized** — No recognized Arabic news terms or phrases were matched by either model "
            "(0 active features)."
        )

    def _render_prediction_card(variant: str, scheme_title: str, count_label: str, res_obj: Dict[str, Any]) -> None:
        if res_obj.get("is_uncategorized"):
            card_html = (
                f'<div class="pred-card uncat">'
                f'<div class="pred-card-head"><span class="pred-scheme-tag">{html.escape(scheme_title)}</span>'
                f'<span class="pred-cat-ar">غير مصنف</span></div>'
                f'<div class="pred-cat-main">Uncategorized</div>'
                f'<div class="pred-stats-row">'
                f'<div><div class="pred-stat-label">Sigmoid Score</div><div class="pred-stat-val">—</div></div>'
                f'<div><div class="pred-stat-label">{html.escape(count_label)}</div><div class="pred-stat-val">0</div></div>'
                f'<div><div class="pred-stat-label">TF-IDF Weight</div><div class="pred-stat-val">0.0000</div></div>'
                f"</div></div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)
            return

        top_prob = res_obj["category_scores"][0]["sigmoid_percent"]
        card_html = (
            f'<div class="pred-card {variant}">'
            f'<div class="pred-card-head"><span class="pred-scheme-tag">{html.escape(scheme_title)}</span>'
            f'<span class="pred-cat-ar">{html.escape(str(res_obj["predicted_category"]))}</span></div>'
            f'<div class="pred-cat-main">{html.escape(str(res_obj["predicted_category_en"]))}</div>'
            f'<div class="pred-stats-row">'
            f'<div><div class="pred-stat-label">Sigmoid Score</div><div class="pred-stat-val">{top_prob:.2f}%</div></div>'
            f'<div><div class="pred-stat-label">{html.escape(count_label)}</div><div class="pred-stat-val">{res_obj["n_active_features"]}</div></div>'
            f'<div><div class="pred-stat-label">TF-IDF Weight</div><div class="pred-stat-val">{res_obj["total_tfidf_weight"]:.4f}</div></div>'
            f"</div></div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        _render_prediction_card("frame", "Frame Semantics", "Active Phrases", frame_res)
    with c2:
        _render_prediction_card("medan", "Semantic Field", "Active Words", medan_res)

    if both_uncat:
        return

    st.markdown("#### Arabic Article Text (RTL) &amp; Semantic Field Highlights")
    st.caption(
        "Green = positive Semantic Field coefficient (+) for the predicted category; "
        "Red = negative coefficient (-)."
    )
    pos_terms = [d["term"] for d in medan_res["positive_coefs"]]
    neg_terms = [d["term"] for d in medan_res["negative_coefs"]]
    highlighted_html = highlight_arabic_terms_html(active_text, pos_terms, neg_terms)
    st.markdown(f'<div class="arabic-rtl-box">{highlighted_html}</div>', unsafe_allow_html=True)

    with st.expander("View Text Preprocessing Results (Semantic Field vs Frame Semantics)"):
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**1. Frame Semantics Path (With Function Words &amp; Prepositions Retained):**")
            st.markdown(
                f'<div class="arabic-clean-box">{html.escape(frame_res["cleaned_text"] or "—")}</div>',
                unsafe_allow_html=True,
            )
        with e2:
            st.markdown("**2. Semantic Field Path (After Stopword Removal):**")
            st.markdown(
                f'<div class="arabic-clean-box">{html.escape(medan_res["cleaned_text"] or "—")}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Sigmoid Probability Distribution Across 11 Categories")
    score_col1, score_col2 = st.columns(2)
    for col_box, title_txt, res_obj, bar_color in [
        (score_col1, "Frame Semantics", frame_res, VIOLET_PRIMARY),
        (score_col2, "Semantic Field", medan_res, ORANGE_SECONDARY),
    ]:
        with col_box:
            if res_obj.get("is_uncategorized"):
                st.info(f"{title_txt}: **Uncategorized** due to 0 active features.")
                continue
            scores_df = pd.DataFrame(res_obj["category_scores"])
            scores_sorted = scores_df.sort_values("sigmoid_percent", ascending=True)
            fig, ax = plt.subplots(figsize=(7.6, 5.0))
            ylabels = [reshape_arabic_for_plot(r["category"]) for _, r in scores_sorted.iterrows()]
            bars = ax.barh(ylabels, scores_sorted["sigmoid_percent"], color=bar_color, height=0.66)
            for bar, val in zip(bars, scores_sorted["sigmoid_percent"]):
                ax.text(
                    min(float(val) + 1.5, 102.0),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=8.5,
                    color=INK_DARK,
                )
            ax.set_xlim(0, 118)
            ax.set_xlabel("Sigmoid Score (%)", fontsize=10)
            ax.set_title(title_txt, fontsize=11, pad=10)
            ax.tick_params(axis="y", labelsize=9.5)
            _apply_chart_theme(ax, grid_axis="x")
            if font_prop:
                for lbl in ax.get_yticklabels():
                    lbl.set_fontproperties(font_prop)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.markdown("#### Semantic Field Terms, Frame Semantics Phrases, TF-IDF Weights &amp; Model Coefficients")
    ext_col1, ext_col2 = st.columns(2)

    with ext_col1:
        st.markdown("##### 1. Semantic Field Extraction")
        if medan_res.get("is_uncategorized"):
            st.info("No matching Semantic Field words found in the input text (**Uncategorized**).")
        else:
            st.write(
                f"Positive coefs (+) : `{len(medan_res['positive_coefs'])}` words "
                f"(sum **`+{medan_res['total_positive_coef']:.4f}`**) | "
                f"Negative coefs (-) : `{len(medan_res['negative_coefs'])}` words "
                f"(sum **`{medan_res['total_negative_coef']:.4f}`**) | "
                f"Intercept (`bias`): `{medan_res['intercept']:.4f}`"
            )
            medan_df = pd.DataFrame(medan_res["features_by_tfidf"]).rename(
                columns={
                    "term": "Semantic Field Term",
                    "tfidf_weight": "TF-IDF Weight",
                    "svm_coef": "Model Coef",
                    "contribution": "Contribution",
                }
            )
            if not medan_df.empty:
                medan_top = medan_df.sort_values(
                    by="Model Coef", key=lambda s: s.abs(), ascending=False
                ).head(top_n_features)
                fig_h_m = max(4.6, 0.34 * len(medan_top) + 1.4)
                fig_m, ax_m = plt.subplots(figsize=(7.6, fig_h_m))
                plot_m = medan_top.sort_values("Model Coef", ascending=True)
                ax_m.barh(
                    [reshape_arabic_for_plot(t) for t in plot_m["Semantic Field Term"]],
                    plot_m["Model Coef"],
                    color=["#16A34A" if v > 0 else "#DC2626" for v in plot_m["Model Coef"]],
                    height=0.66,
                )
                ax_m.axvline(0, color=INK_DARK, linewidth=0.9)
                ax_m.set_title(f"Top {len(plot_m)} Semantic Field Coefficients (+/-)", fontsize=11, pad=10)
                ax_m.set_xlabel("Hyperplane Coefficient", fontsize=10)
                ax_m.tick_params(axis="y", labelsize=9.5)
                _apply_chart_theme(ax_m, grid_axis="x")
                if font_prop:
                    for lbl in ax_m.get_yticklabels():
                        lbl.set_fontproperties(font_prop)
                fig_m.tight_layout()
                st.pyplot(fig_m)
                plt.close(fig_m)

                st.dataframe(
                    medan_df[["Semantic Field Term", "TF-IDF Weight", "Model Coef", "Contribution"]].head(
                        top_n_features
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with ext_col2:
        st.markdown("##### 2. Frame Semantics Extraction")
        if frame_res.get("is_uncategorized"):
            st.info("No active Frame Semantics phrases found in the input text (**Uncategorized**).")
        else:
            st.write(
                f"Positive coefs (+) : `{len(frame_res['positive_coefs'])}` phrases "
                f"(sum **`+{frame_res['total_positive_coef']:.4f}`**) | "
                f"Negative coefs (-) : `{len(frame_res['negative_coefs'])}` phrases "
                f"(sum **`{frame_res['total_negative_coef']:.4f}`**) | "
                f"Intercept (`bias`): `{frame_res['intercept']:.4f}`"
            )
            frame_df = pd.DataFrame(frame_res["features_by_tfidf"]).rename(
                columns={
                    "term": "Frame Semantics Phrase",
                    "n_tokens": "Words",
                    "tfidf_weight": "TF-IDF Weight",
                    "svm_coef": "Model Coef",
                    "contribution": "Contribution",
                }
            )
            if not frame_df.empty:
                frame_top = frame_df.head(top_n_features).iloc[::-1]
                fig_h_f = max(4.6, 0.34 * len(frame_top) + 1.4)
                fig_f, ax_f = plt.subplots(figsize=(7.6, fig_h_f))
                ax_f.barh(
                    [reshape_arabic_for_plot(t) for t in frame_top["Frame Semantics Phrase"]],
                    frame_top["TF-IDF Weight"],
                    color=VIOLET_PRIMARY,
                    height=0.66,
                )
                max_w = float(frame_top["TF-IDF Weight"].max()) if len(frame_top) else 1.0
                ax_f.set_xlim(0, max_w * 1.15)
                ax_f.set_title(f"Top {len(frame_top)} Frame Semantics Phrases (TF-IDF Weight)", fontsize=11, pad=10)
                ax_f.set_xlabel("TF-IDF Weight", fontsize=10)
                ax_f.tick_params(axis="y", labelsize=9.5)
                _apply_chart_theme(ax_f, grid_axis="x")
                if font_prop:
                    for lbl in ax_f.get_yticklabels():
                        lbl.set_fontproperties(font_prop)
                fig_f.tight_layout()
                st.pyplot(fig_f)
                plt.close(fig_f)

                st.dataframe(
                    frame_df[
                        ["Frame Semantics Phrase", "Words", "TF-IDF Weight", "Model Coef", "Contribution"]
                    ].head(top_n_features),
                    width="stretch",
                    hide_index=True,
                )


def _render_corpus_and_lexicons_tab(
    corpus_stats: Dict[str, Any],
    font_prop: Optional[fm.FontProperties],
) -> None:
    st.subheader("Arabic News Corpus Profile &amp; 11-Category Semantic Lexicons")

    source_mode = corpus_stats.get("source_mode", "not_generated_yet")
    if source_mode == "not_generated_yet":
        st.info(
            "Corpus statistics in `results/stats/` have not been generated yet. "
            "Run `notebooks/01_data_collection_and_compilation.ipynb` and "
            "`notebooks/03_corpus_trends_and_linguistic_stats.ipynb` to compute them from `data/basic_articles_dataset.csv`."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valid Articles", f"{corpus_stats['total_valid_articles']:,}")
        total_tok = corpus_stats.get("total_tokens")
        ar_types = corpus_stats.get("arabic_word_types")
        nonar_types = corpus_stats.get("nonarabic_word_types")
        m2.metric("Total Corpus Tokens", f"{total_tok:,}" if total_tok is not None else "Run NB 03")
        m3.metric("Unique Arabic Word Types", f"{ar_types:,}" if ar_types is not None else "Run NB 03")
        m4.metric("Non-Arabic Word Types", f"{nonar_types:,}" if nonar_types is not None else "Run NB 03")

        raw_cat_df: pd.DataFrame = corpus_stats["category_distribution"].copy()
        cat_en_map = {LABELS[cid]: LABELS_EN[cid] for cid in LABELS}
        raw_cat_df["Translation"] = raw_cat_df["Kategori"].map(cat_en_map).fillna(raw_cat_df["Terjemahan"])
        if corpus_stats.get("tokens_per_category"):
            raw_cat_df["Total Tokens"] = raw_cat_df["Kategori"].map(corpus_stats["tokens_per_category"])

        display_cat_df = raw_cat_df.rename(
            columns={
                "Kategori": "Category (Arabic)",
                "Artikel Valid": "Valid Articles",
                "Persentase (%)": "Share (%)",
            }
        )
        ordered_cat_cols = [
            c
            for c in ["Category (Arabic)", "Translation", "Valid Articles", "Share (%)", "Total Tokens"]
            if c in display_cat_df.columns
        ]
        display_cat_df = display_cat_df[ordered_cat_cols]

        src_df: pd.DataFrame = corpus_stats["source_distribution"].copy()
        src_df["Country"] = src_df["Asal Negara"].map(COUNTRY_EN_MAP).fillna(src_df["Asal Negara"])
        display_src_df = src_df.rename(
            columns={
                "Portal Berita": "News Portal",
                "Jumlah Artikel": "Articles",
                "Persentase (%)": "Share (%)",
            }
        )
        ordered_src_cols = [
            c for c in ["News Portal", "Country", "Articles", "Share (%)"] if c in display_src_df.columns
        ]
        display_src_df = display_src_df[ordered_src_cols]

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_cat, ax_cat = plt.subplots(figsize=(8.6, 5.4))
            labels_cat = [reshape_arabic_for_plot(c) for c in raw_cat_df["Kategori"].tolist()[::-1]]
            trans_cat = raw_cat_df["Translation"].tolist()[::-1]
            vals_cat = raw_cat_df["Artikel Valid"].tolist()[::-1]
            bars_cat = ax_cat.barh(labels_cat, vals_cat, color=VIOLET_PRIMARY, height=0.66)
            max_cat = max(vals_cat) if vals_cat else 1
            for bar, val, trans in zip(bars_cat, vals_cat, trans_cat):
                ax_cat.text(
                    val + max_cat * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{trans} ({val:,})",
                    va="center",
                    fontsize=8.5,
                    color=INK_DARK,
                )
            ax_cat.set_xlim(0, max_cat * 1.48)
            ax_cat.set_title("Article Distribution Across 11 Categories", fontsize=11.5, pad=10)
            ax_cat.set_xlabel("Valid Articles", fontsize=10)
            ax_cat.tick_params(axis="y", labelsize=10)
            _apply_chart_theme(ax_cat, grid_axis="x")
            if font_prop:
                for lbl in ax_cat.get_yticklabels():
                    lbl.set_fontproperties(font_prop)
            fig_cat.tight_layout()
            st.pyplot(fig_cat)
            plt.close(fig_cat)

        with col_c2:
            fig_src, ax_src = plt.subplots(figsize=(8.2, 5.4))
            fig_src.patch.set_facecolor(CHART_SURFACE)
            ax_src.set_facecolor(CHART_SURFACE)
            total_src = float(src_df["Jumlah Artikel"].sum()) or 1.0
            palette = ["#7B2CFF", "#FF7A1A", "#A6F40D", "#151618", "#A688FF", "#FDBA74"]
            wedges, _, _ = ax_src.pie(
                src_df["Jumlah Artikel"],
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 4.0 else "",
                pctdistance=0.76,
                startangle=140,
                colors=palette[: len(src_df)],
                wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.8},
                textprops={"color": "#FFFFFF", "fontweight": "bold", "fontsize": 8.5},
            )
            legend_labels = [
                f"{r['Portal Berita']} ({r['Country']}) — {r['Jumlah Artikel'] / total_src * 100:.1f}%"
                for _, r in src_df.iterrows()
            ]
            ax_src.legend(
                wedges,
                legend_labels,
                title="Arabic News Portals",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=2,
                fontsize=8.5,
                frameon=False,
            )
            ax_src.set_title("Share of 6 Arabic News Portals", fontsize=11.5, pad=10)
            fig_src.tight_layout()
            st.pyplot(fig_src)
            plt.close(fig_src)

        with st.expander("View Category & Portal Breakdown Tables"):
            t1, t2 = st.columns(2)
            with t1:
                st.dataframe(display_cat_df, width="stretch", hide_index=True)
            with t2:
                st.dataframe(display_src_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Semantic Field &amp; Frame Semantics Lexicon Explorer")
    cat_choices = [f"{cid}: {cname} ({LABELS_EN[cid]})" for cid, cname in LABELS.items()]
    chosen = st.selectbox("Select a Category to Inspect Lexicons:", cat_choices)
    chosen_cid = int(chosen.split(":")[0].strip())
    chosen_name = LABELS[chosen_cid]

    lex = load_category_lexicon_snapshot(chosen_name, top_k=15)
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"**Top-15 Frame Semantics TF-IDF Lexicon (`{chosen_name}`)**")
        if lex["frame_tfidf"].empty:
            st.info("Not available yet. Run Notebook 05 to generate Frame Semantics TF-IDF lexicons.")
        else:
            st.dataframe(lex["frame_tfidf"], width="stretch", hide_index=True)

        st.markdown(f"**Top-15 Core Terms Without Stopwords (`{chosen_name}`)**")
        if lex["core_words"].empty:
            st.info("Not available yet. Run Notebook 03 to generate top word counts.")
        else:
            st.dataframe(lex["core_words"], width="stretch", hide_index=True)

    with k2:
        st.markdown(f"**Top-15 Highest-Frequency 2-Word Phrases (`{chosen_name}`)**")
        if lex["phrase_2"].empty:
            st.info("Not available yet. Run Notebook 03 to generate phrase counts.")
        else:
            st.dataframe(lex["phrase_2"], width="stretch", hide_index=True)

        st.markdown(f"**Top-15 Highest-Frequency 3-Word Phrases (`{chosen_name}`)**")
        if lex["phrase_3"].empty:
            st.info("Not available yet. Run Notebook 03 to generate phrase counts.")
        else:
            st.dataframe(lex["phrase_3"], width="stretch", hide_index=True)


def _render_evaluation_tab(
    eval_tables: Dict[str, Any],
    font_prop: Optional[fm.FontProperties],
) -> None:
    st.subheader("Effectiveness &amp; Efficiency Benchmark of 5 Classification Models")

    eff_df: pd.DataFrame = eval_tables["efficiency"]
    perf_df: pd.DataFrame = eval_tables["performance"]
    svm_frame = eval_tables.get("svm_frame_detail")
    svm_medan = eval_tables.get("svm_medan_detail")

    if perf_df.empty or svm_frame is None or svm_medan is None:
        st.info(
            "Evaluation metric reports for the 5 models were not found in `results/metrics/`. "
            "Run `notebooks/04_model_training_and_evaluation.ipynb` first to train and evaluate all five algorithms."
        )
        return

    st.markdown(
        f"""
        <div class="compare-grid">
            <div class="compare-card-muted">
                <span class="compare-tag">Lexical Baseline</span>
                <div class="compare-title">Semantic Field (Without Frame Semantics)</div>
                <ul class="compare-list">
                    <li>&bull; Core lexical token representation (stopword removal applied)</li>
                    <li>&bull; Data split: 80% Train / 15% Validation (N={svm_medan['support']:,}) / 5% Test</li>
                    <li>&bull; SVM Accuracy: <strong>{svm_medan['accuracy'] * 100:.2f}%</strong></li>
                    <li>&bull; Lightweight memory footprint and fast inference</li>
                </ul>
            </div>
            <div class="compare-card-highlight">
                <span class="compare-tag">Proposed Approach</span>
                <div class="compare-title">With Frame Semantics</div>
                <ul class="compare-list">
                    <li>&#10003; Contextual 2&ndash;3 word phrases retaining Arabic prepositions &amp; particles</li>
                    <li>&#10003; Data split: 85% Train / 10% Validation (N={svm_frame['support']:,}) / 5% Test</li>
                    <li>&#10003; Highest SVM Accuracy: <strong>{svm_frame['accuracy'] * 100:.2f}%</strong> (F1 = {svm_frame['f1_score']:.4f})</li>
                    <li>&#10003; Captures syntactic-semantic relations across news concepts</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_eff, tab_perf = st.tabs(
        ["Classification Effectiveness", "Computational Efficiency (Time, RAM, CPU)"]
    )
    with tab_eff:
        st.dataframe(perf_df, width="stretch", hide_index=True)
    with tab_perf:
        st.dataframe(eff_df, width="stretch", hide_index=True)

    x = np.arange(len(perf_df))
    w = 0.36
    short_names = eval_tables["short_names"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    min_acc = float(min(perf_df["Frame Accuracy"].min(), perf_df["Field Accuracy"].min()) * 100)
    b1 = axes[0].bar(x - w / 2, perf_df["Frame Accuracy"] * 100, w, label="Frame Semantics", color=VIOLET_PRIMARY)
    b2 = axes[0].bar(x + w / 2, perf_df["Field Accuracy"] * 100, w, label="Semantic Field", color=ORANGE_SECONDARY)
    axes[0].set_ylim(max(0.0, min_acc - 5.0), 96.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[0].set_title("Accuracy (%)", fontsize=11, pad=10)
    _apply_chart_theme(axes[0], grid_axis="y")

    axes[1].bar(x - w / 2, eff_df["Frame Time (s)"], w, color=VIOLET_PRIMARY)
    axes[1].bar(x + w / 2, eff_df["Field Time (s)"], w, color=ORANGE_SECONDARY)
    max_t = float(max(eff_df["Frame Time (s)"].max(), eff_df["Field Time (s)"].max()))
    axes[1].set_ylim(0, max_t * 1.15 if max_t else 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[1].set_title("Evaluation Time (seconds)", fontsize=11, pad=10)
    _apply_chart_theme(axes[1], grid_axis="y")

    axes[2].bar(x - w / 2, eff_df["Frame RAM (MB)"], w, color=VIOLET_PRIMARY)
    axes[2].bar(x + w / 2, eff_df["Field RAM (MB)"], w, color=ORANGE_SECONDARY)
    max_r = float(max(eff_df["Frame RAM (MB)"].max(), eff_df["Field RAM (MB)"].max()))
    axes[2].set_ylim(0, max_r * 1.15 if max_r else 1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=24, ha="right", fontsize=9.5)
    axes[2].set_title("Peak RAM Usage (MB)", fontsize=11, pad=10)
    _apply_chart_theme(axes[2], grid_axis="y")

    fig.legend(
        [b1[0], b2[0]],
        ["Frame Semantics", "Semantic Field"],
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
    st.subheader("Per-Category Performance &amp; SVM Confusion Matrix")

    scheme_choice = st.radio(
        "Select SVM Evaluation Scheme:",
        [
            f"With Frame Semantics — Accuracy {svm_frame['accuracy'] * 100:.2f}%",
            f"Without Frame Semantics (Semantic Field) — Accuracy {svm_medan['accuracy'] * 100:.2f}%",
        ],
        horizontal=True,
    )
    active_svm = svm_frame if scheme_choice.startswith("With Frame Semantics") else svm_medan

    per_class_df = pd.DataFrame(
        [
            {
                "ID": cid,
                "Category": LABELS_EN[cid],
                "Arabic": LABELS[cid],
                "Precision": active_svm["per_class"][cid]["precision"],
                "Recall": active_svm["per_class"][cid]["recall"],
                "F1-Score": active_svm["per_class"][cid]["f1-score"],
                "Support": active_svm["per_class"][cid]["support"],
            }
            for cid in range(1, 12)
        ]
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("SVM Accuracy", f"{active_svm['accuracy'] * 100:.2f}%")
    s2.metric("Total Samples", f"{active_svm['support']:,}")
    s3.metric("Precision", f"{active_svm['precision']:.4f}")
    s4.metric("F1-Score", f"{active_svm['f1_score']:.4f}")

    cm_col1, cm_col2 = st.columns([1.5, 1])
    with cm_col1:
        labels_ar = [reshape_arabic_for_plot(LABELS[i]) for i in range(1, 12)]
        fig_cm, ax_cm = plt.subplots(figsize=(9.8, 8.4))
        fig_cm.patch.set_facecolor(CHART_SURFACE)
        cm_cmap = sns.light_palette(VIOLET_PRIMARY, as_cmap=True)
        sns.heatmap(
            active_svm["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap=cm_cmap,
            xticklabels=labels_ar,
            yticklabels=labels_ar,
            cbar_kws={"shrink": 0.82},
            ax=ax_cm,
        )
        ax_cm.set_title(
            f"SVM Confusion Matrix — Accuracy: {active_svm['accuracy'] * 100:.2f}% (N = {active_svm['support']:,})",
            fontsize=11.5,
            pad=14,
            color=INK_DARK,
            fontweight="bold",
        )
        ax_cm.set_xlabel("Predicted", fontsize=10.5, labelpad=8, color="#575860")
        ax_cm.set_ylabel("Actual", fontsize=10.5, labelpad=8, color="#575860")
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
        st.markdown("##### Category Performance Summary")
        st.markdown(
            f"""
            <div class="pred-card frame" style="margin-bottom:0.75rem;">
                <div class="pred-card-head">
                    <span class="pred-scheme-tag">Highest F1</span>
                    <span class="pred-cat-ar">{html.escape(str(best_row['Arabic']))}</span>
                </div>
                <div class="pred-cat-main" style="margin-bottom:0.2rem; font-size:1.15rem;">
                    {html.escape(str(best_row['Category']))} &mdash; <code>{best_row['F1-Score']:.4f}</code>
                </div>
            </div>
            <div class="pred-card medan" style="margin-bottom:0.75rem;">
                <div class="pred-card-head">
                    <span class="pred-scheme-tag">Lowest F1</span>
                    <span class="pred-cat-ar">{html.escape(str(low_row['Arabic']))}</span>
                </div>
                <div class="pred-cat-main" style="margin-bottom:0.2rem; font-size:1.15rem;">
                    {html.escape(str(low_row['Category']))} &mdash; <code>{low_row['F1-Score']:.4f}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Diagonal cells represent correctly predicted articles per category, "
            "while off-diagonal counts reflect cross-category misclassifications."
        )

    st.markdown("##### Detailed Per-Category Metrics")
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
            <div style="margin:0 0 0.35rem 0; font-size:1.38rem; font-weight:700; letter-spacing:-0.025em; color:#FFFFFF;">
                Research Methodology &amp; System Architecture
            </div>
            <p style="margin:0; font-size:0.9rem; color:rgba(255,255,255,0.88);">
                All experiments are reproducibly derived from the single corpus source <code>data/basic_articles_dataset.csv</code> and <code>data/stopwords.txt</code>.
            </p>
            <div class="method-grid">
                <div class="method-card">
                    <span class="method-badge violet">01</span>
                    <h4>Data Collection (Web Scraping)</h4>
                    <p>Automated acquisition across 6 Arabic online news portals (<code>alsumaria.tv</code>, <code>al-sharq.com</code>, <code>assahraa.ma</code>, <code>akhbarelsyaha.com</code>, <code>sarayanews.com</code>, <code>addustour.com</code>) covering 11 thematic categories.</p>
                </div>
                <div class="method-card">
                    <span class="method-badge orange">02</span>
                    <h4>Preprocessing &amp; Dual-Path Splitting</h4>
                    <p>Removal of numerals, Arabic/ASCII punctuation, and casing normalization, branching into the <strong>Semantic Field</strong> path (stopword removal, 80/15/5) and <strong>Frame Semantics</strong> path (function words &amp; prepositions retained, 85/10/5).</p>
                </div>
                <div class="method-card">
                    <span class="method-badge lime">03</span>
                    <h4>Corpus Statistics &amp; Lexicon Extraction</h4>
                    <p>Computes token totals before and after stopword removal, Arabic vs. Non-Arabic vocabulary sizes, and extracts Top-1000 core words and TF-IDF phrase lexicons per category.</p>
                </div>
                <div class="method-card">
                    <span class="method-badge">04</span>
                    <h4>5-Model Benchmark &amp; Calibration</h4>
                    <p>Evaluates classification effectiveness and compute efficiency via <code>psutil</code> across <em>SVM, Logistic Regression, Naïve Bayes, Random Forest, and Gradient Boosting</em>, paired with <strong>Sigmoid</strong> confidence mapping for SVM.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    favicon_path = ASSETS_BRANDING_DIR / "png" / "logo-icon-badge-64.png"
    st.set_page_config(
        page_title="Arabic News Semantics — Semantic Fields & Frame Semantics",
        page_icon=str(favicon_path) if favicon_path.exists() else "📰",
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

    _render_top_navbar()
    _render_hero_banner(corpus_stats, eval_tables)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Classification & Extraction",
            "Corpus & Lexicons",
            "5-Model Benchmark",
            "Methodology",
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
