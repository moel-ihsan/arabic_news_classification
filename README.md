<p align="left">
  <a href="https://arabic-news-semantics.streamlit.app/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/branding/png/logo-horizontal-dark-color-2100w.png">
      <source media="(prefers-color-scheme: light)" srcset="assets/branding/png/logo-horizontal-color-2100w.png">
      <img alt="Arabic News Semantics logo" src="assets/branding/png/logo-horizontal-color-2100w.png" width="320">
    </picture>
  </a>
</p>

# Arabic News Classification via Semantic Fields &amp; Frame Semantics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arabic-news-semantics.streamlit.app/)

Official computational linguistics and Natural Language Processing (NLP) research repository for multi-class Arabic news classification (**11 thematic categories**) across **170,934 valid Arabic online news articles** scraped from **6 Middle Eastern and North African news portals**. This project systematically benchmarks two semantic feature representations — **Semantic Field** and **Frame Semantics** — paired with an interactive **Streamlit** demo ([Live App](https://arabic-news-semantics.streamlit.app/)) featuring native Right-to-Left (RTL) Arabic support.

---

## Research Background

Arabic text processing presents rich morphological inflection and lexical ambiguity when words are modeled in isolation without syntactic context. This study evaluates both classification effectiveness and computational efficiency across five supervised learning algorithms (*Support Vector Machine*, *Logistic Regression*, *Naïve Bayes*, *Random Forest*, and *Gradient Boosting*) under two TF-IDF representation schemes:

1. **Semantic Field Approach**:
   - Models core content words as single lexical tokens after applying Arabic stopword removal to isolate topic-bearing vocabulary (`80%` train / `15%` validation / `5%` test).
   - **SVM** achieves **`90.26%`** accuracy on the full corpus (`N = 25,641`).
2. **Frame Semantics Approach**:
   - Models contextual 2–3 word phrases while **retaining Arabic prepositions and function words** (`في`, `من`, `على`, `إلى`, `أن`), preserving the relational syntactic-semantic frame connecting entities and events (`85%` train / `10%` validation / `5%` test).
   - **SVM (`LinearSVC C=10`)** achieves **`90.61%`** accuracy (`Precision = 0.9088`, `F1-Score = 0.9068`, `N = 17,094`), reaching up to `0.9879` F1 on Sports (`رياضة`).

---

## Corpus Profile

All articles were collected via structured web scraping (`requests` and `BeautifulSoup`) from six Arabic news portals and compiled into `basic_articles_dataset.csv` (`171,175` raw files; **`170,934` non-null valid articles**; **`42,661,503` raw tokens**; **`30,559,650` tokens after stopword removal**; **`754,584` unique Arabic word types**; and **`74,727` Non-Arabic word types**).

### Distribution Across 6 News Portals

| Key | News Portal | Country | Articles | Share |
|---|---|---|---:|---:|
| `alsumaria` | `alsumaria.tv` | Iraq | `73,196` | `42.8%` |
| `alsharq` | `al-sharq.com` | Qatar | `41,159` | `24.1%` |
| `assahraa` | `assahraa.ma` | Morocco | `28,601` | `16.7%` |
| `akhbarelsyaha` | `akhbarelsyaha.com` | Egypt | `15,002` | `8.8%` |
| `saraya` | `sarayanews.com` | Jordan | `7,805` | `4.6%` |
| `addustour` | `addustour.com` | Jordan | `5,171` | `3.0%` |
| **Total** | — | — | **`170,934`** | **`100.0%`** |

### Distribution Across 11 Categories

| ID | Category (Arabic) | English Label | Valid Articles | Share | Total Tokens |
|---:|---|---|---:|---:|---:|
| `1` | `أمن` | Security | `17,341` | `10.1%` | `4,102,495` |
| `2` | `اقتصاد` | Economy | `15,899` | `9.3%` | `3,636,123` |
| `3` | `تكنولوجيا` | Technology | `11,036` | `6.5%` | `2,751,310` |
| `4` | `ثقافة` | Culture | `10,844` | `6.3%` | `2,086,216` |
| `5` | `دوليات` | International | `17,964` | `10.5%` | `5,608,498` |
| `6` | `رياضة` | Sports | `18,612` | `10.9%` | `5,476,537` |
| `7` | `سياحة` | Travel | `18,250` | `10.7%` | `5,608,691` |
| `8` | `سياسة` | Politics | `18,295` | `10.7%` | `3,895,209` |
| `9` | `صحة` | Health | `13,260` | `7.8%` | `3,004,546` |
| `10` | `علم` | Science | `14,199` | `8.3%` | `3,438,063` |
| `11` | `مجتمع` | Society | `15,234` | `8.9%` | `3,053,815` |

---

## Repository Structure

```text
arabic_news_classification/
├── app/
│   └── streamlit_app.py                      # Interactive Streamlit web app (RTL Arabic support)
├── assets/
│   └── branding/
│       ├── preview.png                       # Light & dark brand identity preview sheet
│       ├── svg/                              # Palette-aligned vector icons & horizontal lockups
│       └── png/                              # Exported PNG icons (64–1024px) & 2100w headers
├── data/
│   ├── stopwords.txt                         # Arabic stopword list (used by Semantic Field path)
│   └── basic_articles_dataset.csv            # Compiled corpus dataset (local, gitignored)
├── notebooks/
│   ├── 01_data_collection_and_compilation.ipynb
│   ├── 02_preprocessing_and_dataset_splitting.ipynb
│   ├── 03_corpus_trends_and_linguistic_stats.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_semantic_field_and_frame_extraction.ipynb
├── src/arabic_news/
│   ├── __init__.py
│   ├── acquisition.py                        # News scrapers, corpus compilation, & distribution stats
│   ├── config.py                             # Relative paths, 11-class maps, portal metadata, & seed
│   ├── evaluation.py                         # 80/15/5 & 85/10/5 splits, metrics, psutil profiling
│   ├── features.py                           # TF-IDF vectorization & per-category phrase weighting
│   ├── models.py                             # 5 classification models, dual-SVM inference, Sigmoid score
│   ├── preprocessing.py                      # Arabic/ASCII cleaning, stopwords, & RTL shaping
│   └── visualization.py                      # Cross-platform Arabic font lookup for plots & WordClouds
├── scripts/
│   └── train_all_models.py                   # Memory-efficient resumable CLI training runner
├── tests/
│   └── test_reproducibility.py               # Automated tests for pipeline, notebooks, CLI, & UI
├── .gitignore
├── .streamlit/config.toml                    # Custom light theme configuration
├── CITATION.cff                              # Academic citation metadata
├── LICENSE
├── packages.txt                              # Linux Arabic fonts for Streamlit Cloud
├── pyproject.toml
└── requirements.txt
```

*(Large raw files such as `data/basic_articles_dataset.csv` and the full local 5-model artifacts under `models/all_models/` are excluded via `.gitignore` to respect repository size limits and original news copyright. All outputs in `results/` and `models/` are generated from `data/basic_articles_dataset.csv` and `data/stopwords.txt`. With the precomputed `results/` files (~2.5 MB) and deployable SVM bundles `models/svm_medan_makna.joblib` (17 MB) and `models/svm_frame_semantik.joblib` (34 MB) committed, the Streamlit demo runs out of the box without retraining.)*

---

## Installation

Python `3.9`–`3.11` in a fresh virtual environment is recommended:

```bash
git clone https://github.com/moel-ihsan/arabic_news_classification.git
cd arabic_news_classification

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

Place `basic_articles_dataset.csv` inside `data/` before executing the notebooks (`data/stopwords.txt` is included in the repository).

---

## Reproducing Notebooks & Tests

The five notebooks follow the research pipeline sequentially using `data/basic_articles_dataset.csv` and `data/stopwords.txt`:

1. `notebooks/01_data_collection_and_compilation.ipynb` — Scraper specifications for the 6 Arabic news portals, HTML-to-text compilation, and initial corpus distributions (`results/stats/` & `results/figures/`).
2. `notebooks/02_preprocessing_and_dataset_splitting.ipynb` — Arabic normalization, dual branching (stopword removal vs. preposition retention), deterministic stratified splits (`random_state=42`), and export of 11 sample test articles to `results/stats/sample_11_test_articles.csv`.
3. `notebooks/03_corpus_trends_and_linguistic_stats.ipynb` — Pre/post-stopword token counts, Arabic vs. Non-Arabic unique vocabulary counts (`results/stats/corpus_summary.json`), and Top-1000 unigram, bigram, and trigram tables (`results/lexicons/`).
4. `notebooks/04_model_training_and_evaluation.ipynb` — Training, `psutil` CPU/RAM profiling, and evaluation of all 5 algorithms across both representation schemes (`results/metrics/`), saving reusable models/vectorizers to `models/all_models/*.joblib` and compact SVM bundles (`models/svm_*.joblib`) for the Streamlit app.
5. `notebooks/05_semantic_field_and_frame_extraction.ipynb` — Frame Semantics TF-IDF lexicon extraction (`results/lexicons/tfidf_top1000/`), dual-scheme SVM feature & coefficient inspection on 11 test articles, Sigmoid probability analysis, and WordCloud visualizations.

*(Memory-efficient CLI runner for Stage 4 with automatic per-model checkpoint resume:)*

```bash
# Train fast linear & probabilistic models first
python scripts/train_all_models.py --models SVC LogisticRegression MultinomialNB

# Resume and train the remaining tree ensemble models
python scripts/train_all_models.py
```

Run the automated test suite to verify pipeline integrity, notebooks, CLI resumption, and the Streamlit UI:

```bash
pytest -v
```

---

## Running the Streamlit Demo

- **Live Deployment (Streamlit Community Cloud)**: https://arabic-news-semantics.streamlit.app/
- **Run Locally** from the repository root:

```bash
streamlit run app/streamlit_app.py
```

Key capabilities in the web interface:
- **Dual Classification & Extraction (RTL)**: Paste any Arabic article or select one of 11 representative samples to inspect simultaneous predictions from **Frame Semantics** and **Semantic Field**, 11-class Sigmoid probability charts, and RTL Arabic text with highlighted positive (`+`) and negative (`-`) SVM hyperplane terms.
- **TF-IDF Weights & SVM Coefficients**: Inspect active Semantic Field words and Frame Semantics phrases ranked by TF-IDF weight, hyperplane coefficient, and contribution score.
- **Corpus & 11-Category Lexicon Explorer**: Interactive category/portal distributions alongside Top-15 TF-IDF phrases, core words, 2-word phrases, and 3-word phrases per category.
- **5-Model Benchmark Dashboard**: Side-by-side accuracy, precision, recall, F1-score, runtime, peak RAM, CPU tables, bar charts, and 11×11 SVM confusion matrices.

---

## Evaluation Summary

### 1. Effectiveness & Efficiency Across 5 Models (Full Corpus)

| Model | Frame Accuracy | Frame F1 | Field Accuracy | Field F1 | Frame / Field Time (s) | Frame / Field RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| **SVM (`LinearSVC C=10`)** | **`0.9061`** | **`0.9068`** | `0.9026` | `0.9027` | `2.0609` / `0.1099` | `2982.23` / `1107.53` |
| **Logistic Regression** | `0.9027` | `0.9035` | **`0.9039`** | **`0.9041`** | `3.7081` / `0.1100` | `4013.31` / `946.28` |
| **Naïve Bayes** | `0.8834` | `0.8835` | `0.8478` | `0.8487` | `2.6702` / `0.0852` | `3829.72` / `390.92` |
| **Random Forest** | `0.8675` | `0.8686` | `0.8684` | `0.8688` | `142.1841` / `2.8103` | `3263.75` / `2184.48` |
| **Gradient Boosting** | `0.8587` | `0.8623` | `0.8810` | `0.8814` | `4.7601` / `4.5862` | `2266.72` / `1701.52` |

### 2. Per-Category SVM Performance (`LinearSVC C=10`, Frame Semantics)

| Category | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `أمن` (Security) | `0.9563` | `0.9475` | `0.9519` | `1,734` |
| `اقتصاد` (Economy) | `0.9297` | `0.9396` | `0.9346` | `1,590` |
| `تكنولوجيا` (Technology) | `0.8950` | `0.9665` | `0.9294` | `1,103` |
| `ثقافة` (Culture) | `0.9697` | `0.8848` | `0.9253` | `1,085` |
| `دوليات` (International) | `0.9075` | `0.8898` | `0.8986` | `1,797` |
| `رياضة` (Sports) | `0.9861` | `0.9898` | **`0.9879`** | `1,861` |
| `سياحة` (Travel) | `0.9259` | `0.9304` | `0.9282` | `1,826` |
| `سياسة` (Politics) | `0.9644` | `0.9328` | `0.9483` | `1,829` |
| `صحة` (Health) | `0.7791` | `0.7421` | `0.7601` | `1,326` |
| `علم` (Science) | `0.7040` | `0.8092` | `0.7529` | `1,420` |
| `مجتمع` (Society) | `0.9232` | `0.8844` | `0.9034` | `1,523` |

---

## Citation

If you use the code, methodology, or findings from this repository in your research, please cite the following publications:

1. **Journal Article (*Studies in English Language and Education*, 2024)**:
   Ahmad, M. I., &amp; Anwari, M. K. (2024). Computational linguistics and natural language processing techniques for semantic field extraction in Arabic online news. *Studies in English Language and Education*, 11(3), 1686–1709. https://doi.org/10.24815/siele.v11i3.38090

2. **Master's Thesis (UIN Sunan Kalijaga Yogyakarta, 2024)**:
   Ahmad, M. I. (2024). *Komputasi Klasifikasi pada Berita Online Berbahasa Arab dengan Pendekatan Semantik Berbasis Frame* [Master's thesis, Universitas Islam Negeri Sunan Kalijaga Yogyakarta]. https://digilib.uin-suka.ac.id/id/eprint/75147/

```bibtex
@article{ahmad2024computational,
  author  = {Ahmad, Maulana Ihsan and Anwari, Moh. Kanif},
  title   = {Computational Linguistics and Natural Language Processing Techniques for Semantic Field Extraction in Arabic Online News},
  journal = {Studies in English Language and Education},
  year    = {2024},
  volume  = {11},
  number  = {3},
  pages   = {1686--1709},
  doi     = {10.24815/siele.v11i3.38090},
  url     = {https://doi.org/10.24815/siele.v11i3.38090}
}

@mastersthesis{ahmad2024komputasi,
  author = {Ahmad, Maulana Ihsan},
  title  = {Komputasi Klasifikasi pada Berita Online Berbahasa Arab dengan Pendekatan Semantik Berbasis Frame},
  school = {Universitas Islam Negeri Sunan Kalijaga Yogyakarta},
  year   = {2024},
  url    = {https://digilib.uin-suka.ac.id/id/eprint/75147/}
}
```

---

## License &amp; Contact

- **License**: Released under the [MIT License](LICENSE).
- **Author**: **Maulana Ihsan Ahmad** — Master's Program in Arabic Language and Literature, Faculty of Adab and Cultural Sciences, UIN Sunan Kalijaga Yogyakarta.
- **Email**: `23201011009@student.uin-suka.ac.id`
