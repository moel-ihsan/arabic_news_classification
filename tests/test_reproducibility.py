"""Pengujian otomatis pipeline end-to-end: data tunggal, pra-pemrosesan, ekstraksi, pelatihan SVM, notebook, & Streamlit."""

import ast
import json
import math
from pathlib import Path
import sys
import unicodedata

import nbformat as nbf
import pandas as pd
import pytest

from arabic_news.acquisition import (
    PORTAL_SCRAPER_CONFIG,
    compile_raw_articles_dataset,
    compute_corpus_distributions,
    compute_token_counts_per_category,
    compute_word_type_counts,
    load_or_compute_corpus_stats,
    parse_category_urls_html,
    save_corpus_summary_json,
    scrape_article_html,
)
from arabic_news.config import (
    DATA_DIR,
    LABELS,
    RANDOM_SEED,
    REPO_ROOT,
    SOURCES_INFO,
    resolve_data_file,
)
from arabic_news.evaluation import (
    evaluate_model,
    parse_evaluation_metrics_txt,
    parse_resource_log_txt,
    prepare_medan_and_frame_splits,
    save_evaluation_metrics_txt,
    save_resource_log_txt,
    select_one_sample_per_category,
    split_dataset_80_15_5,
    split_dataset_85_10_5,
    train_classifier_with_monitoring,
)
from arabic_news.features import (
    count_top_ngrams_by_category,
    create_tfidf_vectorizer,
    extract_ngrams_with_weights_by_label,
)
from arabic_news.models import (
    analyze_article_dual_svm,
    build_deployable_svm_bundle,
    load_compact_svm_bundles,
    load_reusable_model,
    predict_with_reusable_model,
    save_deployable_svm_bundle,
    save_reusable_model,
    save_vectorizer,
    sigmoid,
)
from arabic_news.preprocessing import load_stopwords, preprocess_text


def _build_synthetic_11_category_df(per_category: int = 20) -> pd.DataFrame:
    """Membuat DataFrame sintetis 11 kategori dengan frasa domain Arab untuk uji pipeline end-to-end."""
    domain_phrases = {
        1: ("الأمن العام وقوات الشرطة في المحافظة", "وزارة الداخلية أعلنت خطة أمنية جديدة"),
        2: ("البنك المركزي وسعر صرف الدولار في السوق", "النمو الاقتصادي وارتفاع أسعار النفط اليوم"),
        3: ("شركة التكنولوجيا تطلق الهواتف الذكية الجديدة", "تطبيقات الذكاء الاصطناعي والبرمجيات الحديثة"),
        4: ("المهرجان الثقافي الدولي في المعرض الأدبي", "الكاتب العربي ينشر روايته الثقافية الجديدة"),
        5: ("اجتماع مجلس الأمن الدولي لوقف إطلاق النار", "العلاقات الدبلوماسية بين الدول في القمة"),
        6: ("المنتخب الوطني يفوز في بطولة كأس العالم", "المباراة النهائية في الدوري الرياضي الممتاز"),
        7: ("السياحة في المواقع الأثرية والمنتجعات الفندقية", "الرحلات السياحية نحو المدن الساحلية"),
        8: ("جلسة البرلمان ومجلس الوزراء لمناقشة القانون", "الحكومة والانتخابات التشريعية في البلاد"),
        9: ("وزارة الصحة تحذر من انتشار الفيروسات الطبية", "الأطباء والمستشفيات في حملة التطعيم الصحي"),
        10: ("اكتشاف علمي جديد حول الفضاء والكواكب البعيدة", "الباحثون في المختبر العلمي والفيزياء"),
        11: ("المبادرة المجتمعية لدعم الأسرة والشباب", "الجمعية الخيرية تنظم ملتقى المجتمع المحلي"),
    }
    portals = ["alsumaria", "alsharq", "assahraa", "akhbarelsyaha", "saraya", "addustour"]
    rows = []
    for cid, cname in LABELS.items():
        p1, p2 = domain_phrases[cid]
        for idx in range(per_category):
            text = f"{p1} {p2} في تقرير رقم {idx + 1} حول {cname}"
            rows.append(
                {
                    "label": cid,
                    "category": cname,
                    "filename": f"{idx + 1}.txt",
                    "source": portals[(cid + idx) % len(portals)],
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def test_data_folder_layout_and_stopwords():
    """Pastikan data/ hanya menyimpan data/stopwords.txt (dan opsional basic_articles_dataset.csv) dengan skema valid."""
    stopwords_path = resolve_data_file("stopwords.txt")
    assert stopwords_path.exists()

    stops = load_stopwords(stopwords_path=stopwords_path, include_library_stopwords=True)
    assert len(stops) > 100
    assert "في" in stops and "من" in stops

    allowed_files = {"stopwords.txt", "basic_articles_dataset.csv"}
    present_files = {p.name for p in DATA_DIR.iterdir()}
    assert present_files.issubset(allowed_files)

    basic_csv = resolve_data_file("basic_articles_dataset.csv")
    if basic_csv.exists():
        sample_head = pd.read_csv(basic_csv, nrows=25, low_memory=False)
        assert {"label", "category", "filename", "source", "text"}.issubset(sample_head.columns)


def test_stage1_scraper_and_dynamic_corpus_summary(tmp_path):
    """Verifikasi scraper 6 portal, kompilasi teks -> DataFrame, serta compute & cache corpus_summary.json."""
    assert set(PORTAL_SCRAPER_CONFIG.keys()) == {
        "alsumaria",
        "alsharq",
        "assahraa",
        "akhbarelsyaha",
        "saraya",
        "addustour",
    }
    assert len(SOURCES_INFO) == 6
    assert all("count" not in item and "percent" not in item for item in SOURCES_INFO)

    sample_index_html = '<div class="ec-arbic"><a class="title" href="/article/123">خبر</a></div>'
    urls = parse_category_urls_html(sample_index_html, portal="alsharq")
    assert urls == ["https://al-sharq.com/article/123"]

    sample_article_html = (
        '<div class="minit--1"><h1 class="title">فوز المنتخب</h1>'
        "<p>فقرة أولى في المباراة.</p><p>فقرة ثانية مهمة.</p></div>"
    )
    title, body = scrape_article_html(sample_article_html, portal="alsharq")
    assert title == "فوز المنتخب"
    assert "فقرة أولى" in body and "فقرة ثانية" in body

    raw_folder = tmp_path / "sources" / "رياضة" / "alsharq"
    raw_folder.mkdir(parents=True)
    (raw_folder / "1.txt").write_text(f"{title}\n\n{body}\n", encoding="utf-8")
    compiled_df = compile_raw_articles_dataset(tmp_path / "sources")
    assert len(compiled_df) == 1
    assert int(compiled_df.iloc[0]["label"]) == 6
    assert compiled_df.iloc[0]["source"] == "alsharq"

    empty_stats = load_or_compute_corpus_stats(
        dataset_path=tmp_path / "missing.csv",
        stats_dir=tmp_path / "empty_stats",
    )
    assert empty_stats["source_mode"] == "not_generated_yet"

    df_syn = _build_synthetic_11_category_df(per_category=10)
    dists = compute_corpus_distributions(df_syn)
    assert dists["total_valid_articles"] == 110
    tok_cat = compute_token_counts_per_category(df_syn)
    word_types = compute_word_type_counts(df_syn)
    payload = {**dists, "tokens_per_category": tok_cat, **word_types}
    summary_path = save_corpus_summary_json(payload, stats_dir=tmp_path / "stats")
    assert summary_path.exists()

    reloaded = load_or_compute_corpus_stats(stats_dir=tmp_path / "stats", prefer_summary=True)
    assert reloaded["source_mode"] == "loaded_from_results_summary"
    assert reloaded["total_valid_articles"] == 110
    assert isinstance(reloaded["category_distribution"], pd.DataFrame)


def test_stage2_preprocessing_and_splitting():
    """Verifikasi pembersihan teks Arab, percabangan stopwords, split 80/15/5 & 85/10/5, dan 11 sampel uji."""
    stops = set(load_stopwords())
    raw = "فوز المنتخب الوطني في 2024، البطولة!"
    frame_clean = preprocess_text(raw, stopwords=None)
    medan_clean = preprocess_text(raw, stopwords=stops)
    assert "2024" not in frame_clean and "!" not in frame_clean and "،" not in frame_clean
    assert "في" in frame_clean.split()
    assert "في" not in medan_clean.split()

    df_syn = _build_synthetic_11_category_df(per_category=20)
    tr_m, val_m, te_m, y_tr_m, y_val_m, y_te_m = split_dataset_80_15_5(
        df_syn, random_state=RANDOM_SEED, shuffle_first=True
    )
    assert len(tr_m) + len(val_m) + len(te_m) == len(df_syn)
    assert len(y_tr_m) + len(y_val_m) + len(y_te_m) == len(df_syn)

    tr_f, val_f, te_f = split_dataset_85_10_5(df_syn, random_state=RANDOM_SEED)
    assert len(tr_f) + len(val_f) + len(te_f) == len(df_syn)

    samples_11 = select_one_sample_per_category(te_f, random_state=RANDOM_SEED)
    assert len(samples_11) == 11
    assert set(samples_11["label"].astype(int)) == set(range(1, 12))

    dual_splits = prepare_medan_and_frame_splits(df_syn, stopwords=stops, random_state=RANDOM_SEED)
    assert set(dual_splits.keys()) == {"medan_makna", "frame_semantik"}
    assert "في" in " ".join(dual_splits["frame_semantik"]["train_df"]["text"]).split()


def test_stage3_stage4_stage5_end_to_end_pipeline(tmp_path):
    """Verifikasi ekstraksi n-gram, TF-IDF per kategori, pelatihan SVM, log metrik/CPU, & analisis ganda."""
    df_syn = _build_synthetic_11_category_df(per_category=24)
    stops = load_stopwords()
    stop_set = set(stops)

    df_syn["medan_text"] = df_syn["text"].apply(lambda t: preprocess_text(t, stopwords=stop_set))
    df_syn["frame_text"] = df_syn["text"].apply(lambda t: preprocess_text(t, stopwords=None))

    top_bi = count_top_ngrams_by_category(df_syn, n=2, text_column="text", top_n=5)
    assert len(top_bi) == 11
    assert all(len(v) > 0 for v in top_bi.values())

    sport_tfidf = extract_ngrams_with_weights_by_label(
        df_syn, category_id=6, text_column="frame_text", ngram_range=(2, 3), top_n=10
    )
    assert not sport_tfidf.empty and "TF-IDF Weight" in sport_tfidf.columns

    splits = prepare_medan_and_frame_splits(df_syn, stopwords=stops, random_state=RANDOM_SEED)
    vec_f = create_tfidf_vectorizer(ngram_range=(2, 3))
    X_tr_f = vec_f.fit_transform(splits["frame_semantik"]["train_df"]["text"])
    X_val_f = vec_f.transform(splits["frame_semantik"]["val_df"]["text"])
    y_tr_f = splits["frame_semantik"]["train_df"]["label"]
    y_val_f = splits["frame_semantik"]["val_df"]["label"]

    vec_m = create_tfidf_vectorizer(ngram_range=(1, 1))
    X_tr_m = vec_m.fit_transform(splits["medan_makna"]["train_text"])
    X_val_m = vec_m.transform(splits["medan_makna"]["val_text"])
    y_tr_m = splits["medan_makna"]["train_labels"]
    y_val_m = splits["medan_makna"]["val_labels"]

    metrics_dir = tmp_path / "results" / "metrics"
    models_dir = tmp_path / "models"
    save_vectorizer(vec_f, "frame_semantik", models_dir=models_dir)
    save_vectorizer(vec_m, "medan_makna", models_dir=models_dir)

    all_keys = ["SVC", "LogisticRegression", "MultinomialNB", "RandomForest", "GradientBoosting"]
    for key, rfile in [
        ("SVC", "svm.txt"),
        ("LogisticRegression", "logistic.txt"),
        ("MultinomialNB", "naive.txt"),
        ("RandomForest", "forest.txt"),
        ("GradientBoosting", "gradient.txt"),
    ]:
        if key in {"SVC", "LogisticRegression", "MultinomialNB"}:
            clf_f, res_f = train_classifier_with_monitoring(key, X_tr_f, y_tr_f, mode="frame", interval=0.05)
            ev_f = evaluate_model(clf_f, X_val_f, y_val_f)
            clf_m, res_m = train_classifier_with_monitoring(key, X_tr_m, y_tr_m, mode="frame", interval=0.05)
            ev_m = evaluate_model(clf_m, X_val_m, y_val_m)
        else:
            # Simulasi cepat metrik/resource file untuk RandomForest & GradientBoosting pada dataset uji sintetis
            ev_f = evaluate_model(clf_f, X_val_f, y_val_f)
            ev_m = evaluate_model(clf_m, X_val_m, y_val_m)
            res_f = res_m = {"avg_cpu_percent": 50.0, "max_memory_mb": 200.0}

        save_reusable_model(clf_f, model_key=key, scheme="frame_semantik", ngram_range=(2, 3), accuracy=ev_f["accuracy"], models_dir=models_dir)
        save_reusable_model(clf_m, model_key=key, scheme="medan_makna", ngram_range=(1, 1), accuracy=ev_m["accuracy"], models_dir=models_dir)

        mf_txt = save_evaluation_metrics_txt(ev_f, key, metrics_dir / "frame_semantik" / f"no_lemma_evaluation_metrics_{key}.txt")
        mn_txt = save_evaluation_metrics_txt(ev_m, key, metrics_dir / "medan_semantik" / f"no_lemma_evaluation_metrics_{key}.txt")
        save_resource_log_txt(res_f, key, metrics_dir / "cpu_and_memory" / "frame_semantik" / rfile)
        save_resource_log_txt(res_m, key, metrics_dir / "cpu_and_memory" / "medan_semantik" / rfile)

        parsed_f = parse_evaluation_metrics_txt(mf_txt)
        parsed_m = parse_evaluation_metrics_txt(mn_txt)
        parsed_res = parse_resource_log_txt(metrics_dir / "cpu_and_memory" / "frame_semantik" / rfile)
        assert parsed_f["accuracy"] == pytest.approx(ev_f["accuracy"], abs=1e-4)
        assert parsed_m["support"] == len(y_val_m)
        assert parsed_res["max_memory_mb"] is not None

        if key == "SVC":
            assert ev_f["accuracy"] >= 0.90
            assert ev_m["accuracy"] >= 0.90
            b_f = build_deployable_svm_bundle(vec_f, clf_f, mode="frame_semantik", ngram_range=(2, 3), max_features=5000)
            b_m = build_deployable_svm_bundle(vec_m, clf_m, mode="medan_makna", ngram_range=(1, 1), max_features=5000)
            save_deployable_svm_bundle(b_f, models_dir / "svm_frame_semantik.joblib")
            save_deployable_svm_bundle(b_m, models_dir / "svm_medan_makna.joblib")

    sport_article = "المنتخب الوطني يفوز في بطولة كأس العالم في المباراة النهائية في الدوري الرياضي"

    # Pastikan semua 5 model (.joblib) di models/all_models/ dapat dimuat ulang dan dipakai prediksi
    assert len(list((models_dir / "all_models").glob("*.joblib"))) == 12
    for key in all_keys:
        for scheme in ("frame_semantik", "medan_makna"):
            reloaded = load_reusable_model(key, scheme=scheme, models_dir=models_dir)
            assert reloaded["model_key"] == key and reloaded["scheme"] == scheme
            pred = predict_with_reusable_model(sport_article, model_key=key, scheme=scheme, models_dir=models_dir, stopwords=stops)
            assert pred["predicted_label"] == 6

    # Pastikan bundel Streamlit khusus memuat model terbaik SVM
    bundles = load_compact_svm_bundles(models_dir=models_dir)
    assert set(bundles.keys()) == {"frame_semantik", "medan_makna"}
    assert bundles["frame_semantik"]["classifier"].__class__.__name__ == "LinearSVC"
    assert bundles["medan_makna"]["classifier"].__class__.__name__ == "LinearSVC"

    dual = analyze_article_dual_svm(sport_article, bundles=bundles, stopwords=stops)
    assert dual["medan_makna"]["predicted_label"] == 6
    assert dual["frame_semantik"]["predicted_label"] == 6
    assert len(dual["medan_makna"]["features_by_tfidf"]) > 0
    assert len(dual["frame_semantik"]["features_by_tfidf"]) > 0
    assert math.isclose(float(sigmoid(0.0) * 100.0), 50.0, rel_tol=1e-9)

    sys.path.insert(0, str(REPO_ROOT))
    from app.streamlit_app import (
        highlight_arabic_terms_html,
        load_category_lexicon_snapshot,
        load_dashboard_corpus_stats,
        load_evaluation_summary_tables,
        load_sample_test_articles,
    )

    eval_loaded = load_evaluation_summary_tables(metrics_dir=metrics_dir)
    assert len(eval_loaded["performance"]) == 5
    assert eval_loaded["svm_frame_detail"] is not None

    empty_eval = load_evaluation_summary_tables(metrics_dir=tmp_path / "missing_metrics")
    assert empty_eval["performance"].empty and empty_eval["svm_frame_detail"] is None

    empty_lex = load_category_lexicon_snapshot("رياضة", lexicons_dir=tmp_path / "no_lex", stats_dir=tmp_path / "no_stats")
    assert all(df.empty for df in empty_lex.values())

    assert load_sample_test_articles(stats_dir=tmp_path / "no_stats").empty
    assert load_dashboard_corpus_stats(stats_dir=tmp_path / "no_stats")["source_mode"] == "not_generated_yet"

    hl_html = highlight_arabic_terms_html(
        sport_article,
        [d["term"] for d in dual["medan_makna"]["positive_coefs"][:3]],
        [d["term"] for d in dual["medan_makna"]["negative_coefs"][:3]],
    )
    assert "ar-pos" in hl_html


def test_streamlit_app_renders_cleanly_before_notebook_execution():
    """Pastikan app/streamlit_app.py berjalan tanpa exception meski results/ dan models/ belum dibuat."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=20)
    at.run()
    assert not at.exception


def test_notebooks_and_repo_cleanliness():
    """Verifikasi 5 notebook berurutan, sintaks Python valid, dan bebas teks meta-audit / path absolut."""
    expected_notebooks = [
        "01_data_collection_and_compilation.ipynb",
        "02_preprocessing_and_dataset_splitting.ipynb",
        "03_corpus_trends_and_linguistic_stats.ipynb",
        "04_model_training_and_evaluation.ipynb",
        "05_semantic_field_and_frame_extraction.ipynb",
    ]
    nb_dir = REPO_ROOT / "notebooks"
    actual_notebooks = sorted(p.name for p in nb_dir.glob("*.ipynb"))
    assert actual_notebooks == expected_notebooks

    forbidden_snippets = [
        "/Users/",
        "/Volumes/",
        "Source of Truth",
        "Opsi A",
        "sample_sport_article",
        "journal_unigram_benchmark",
        "sport_article_case_study",
    ]

    for nb_name in expected_notebooks:
        nb_path = nb_dir / nb_name
        raw_txt = nb_path.read_text(encoding="utf-8")
        for bad in forbidden_snippets:
            assert bad not in raw_txt, f"Teks terlarang '{bad}' ditemukan di {nb_name}"

        nb = nbf.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)

    for py_path in (
        list((REPO_ROOT / "src").rglob("*.py"))
        + list((REPO_ROOT / "scripts").glob("*.py"))
        + [REPO_ROOT / "app" / "streamlit_app.py"]
    ):
        py_txt = py_path.read_text(encoding="utf-8")
        for bad in forbidden_snippets:
            assert bad not in py_txt, f"Teks terlarang '{bad}' ditemukan di {py_path.name}"


def test_cli_train_all_models_resumable(tmp_path):
    """Verifikasi skrip CLI hemat RAM scripts/train_all_models.py mendukung subset model & auto-resume checkpoint."""
    syn_csv = tmp_path / "syn_corpus.csv"
    _build_synthetic_11_category_df(per_category=24).to_csv(syn_csv, index=False, encoding="utf-8")

    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "figures"

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.train_all_models import main as cli_main

    # Tahap 1: latih model ringan dulu (SVC + LogisticRegression)
    cli_main([
        "--dataset", str(syn_csv),
        "--models", "SVC", "LogisticRegression",
        "--metrics-dir", str(metrics_dir),
        "--models-dir", str(models_dir),
        "--figures-dir", str(figures_dir),
    ])
    assert (models_dir / "svm_frame_semantik.joblib").exists()
    assert (models_dir / "svm_medan_makna.joblib").exists()
    assert (models_dir / "all_models" / "logistic_regression_frame_semantik.joblib").exists()
    assert len(pd.read_csv(metrics_dir / "effectiveness_comparison.csv")) == 2
    assert (figures_dir / "model_comparison.png").exists()
    assert (figures_dir / "confusion_matrix_svm_frame.png").exists()

    # Tahap 2: resume otomatis untuk menambah MultinomialNB tanpa melatih ulang SVC/LogReg
    svc_mtime_before = (models_dir / "svm_frame_semantik.joblib").stat().st_mtime
    cli_main([
        "--dataset", str(syn_csv),
        "--models", "SVC", "LogisticRegression", "MultinomialNB",
        "--metrics-dir", str(metrics_dir),
        "--models-dir", str(models_dir),
        "--figures-dir", str(figures_dir),
    ])
    assert (models_dir / "svm_frame_semantik.joblib").stat().st_mtime == svc_mtime_before
    assert (models_dir / "all_models" / "multinomial_nb_medan_makna.joblib").exists()
    assert len(pd.read_csv(metrics_dir / "effectiveness_comparison.csv")) == 3
