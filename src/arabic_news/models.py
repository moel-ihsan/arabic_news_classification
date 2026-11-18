"""Definisi lima algoritma klasifikasi, pembuatan bundel SVM deployable, dan ekstraksi fitur."""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC

from .config import LABELS, LABELS_EN, LABELS_ID, MODELS_DIR, RANDOM_SEED
from .preprocessing import load_stopwords, preprocess_text

MODEL_FILE_SLUGS: Dict[str, str] = {
    "SVC": "svc",
    "LogisticRegression": "logistic_regression",
    "MultinomialNB": "multinomial_nb",
    "RandomForest": "random_forest",
    "GradientBoosting": "gradient_boosting",
}


def get_classifier_unigram(classify_with: str, random_state: int = RANDOM_SEED):
    """Mengembalikan lima model klasifikasi untuk konfigurasi Medan Makna (unigram)."""
    if classify_with == "MultinomialNB":
        return MultinomialNB()
    if classify_with == "SVC":
        return SVC(kernel="linear")
    if classify_with in ("Logistics", "LogisticRegression"):
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if classify_with in ("RandomـForest", "RandomForest"):
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    if classify_with in ("GradientـBoosting", "GradientBoosting"):
        return GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    raise ValueError(f"Tipe model unigram tidak dikenal: {classify_with}")


def get_classifier_frame(classify_with: str, random_state: int = RANDOM_SEED):
    """Mengembalikan lima model klasifikasi untuk evaluasi komparatif Medan Makna dan Semantik Frame."""
    if classify_with == "SVC":
        return LinearSVC(C=10, random_state=random_state)
    if classify_with == "MultinomialNB":
        return MultinomialNB(alpha=0, fit_prior=False)
    if classify_with == "LogisticRegression":
        return LogisticRegression(
            C=1000, solver="liblinear", max_iter=9999999, random_state=random_state
        )
    if classify_with == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            max_samples=0.95,
            random_state=random_state,
            n_jobs=-1,
        )
    if classify_with == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=2,
            max_features="sqrt",
            verbose=0,
            random_state=random_state,
        )
    raise ValueError(f"Tipe model tidak dikenal: {classify_with}")


def train_classifier(
    classify_with: str,
    train_tfidf,
    train_labels,
    mode: str = "frame",
    random_state: int = RANDOM_SEED,
):
    """Melatih satu model klasifikasi sesuai konfigurasi ('frame' atau 'unigram')."""
    if mode in ("unigram", "medan"):
        classifier = get_classifier_unigram(classify_with, random_state=random_state)
    elif mode == "frame":
        classifier = get_classifier_frame(classify_with, random_state=random_state)
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}. Gunakan 'unigram' atau 'frame'.")

    classifier.fit(train_tfidf, train_labels)
    return classifier


def sigmoid(x):
    """Fungsi sigmoid per elemen untuk memetakan decision_function ke skor persentase kelas."""
    return 1.0 / (1.0 + np.exp(-x))


def build_deployable_svm_bundle(
    vectorizer: TfidfVectorizer,
    classifier: LinearSVC,
    mode: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    full_accuracy: Optional[float] = None,
    compact_accuracy: Optional[float] = None,
) -> Dict[str, Any]:
    """Membuat bundel model LinearSVC kompak (<50 MB) dengan fitur berkoefisien SVM terbesar."""
    coef = np.asarray(classifier.coef_)
    max_abs = np.max(np.abs(coef), axis=0)
    valid_mask = np.isfinite(max_abs) & (max_abs <= 50.0)
    valid_indices = np.where(valid_mask)[0]

    k = min(max_features, len(valid_indices))
    top_local = np.argpartition(max_abs[valid_indices], -k)[-k:]
    selected_idx = np.sort(valid_indices[top_local])

    old_vocab = vectorizer.vocabulary_
    inv_vocab = {idx: term for term, idx in old_vocab.items()}
    new_vocab = {inv_vocab[old_i]: new_i for new_i, old_i in enumerate(selected_idx)}

    sub_vec = TfidfVectorizer(
        ngram_range=ngram_range,
        vocabulary=new_vocab,
        dtype=np.float32,
    )
    sub_vec._tfidf = TfidfTransformer(
        norm=vectorizer.norm,
        use_idf=vectorizer.use_idf,
        smooth_idf=vectorizer.smooth_idf,
        sublinear_tf=vectorizer.sublinear_tf,
    )
    sub_idf = np.asarray(vectorizer.idf_[selected_idx], dtype=np.float32)
    sub_vec._tfidf.idf_ = sub_idf
    sub_vec.fixed_vocabulary_ = True

    sub_clf = LinearSVC(C=classifier.C, random_state=classifier.random_state)
    sub_clf.classes_ = np.array(classifier.classes_, copy=True)
    sub_clf.coef_ = np.asarray(coef[:, selected_idx], dtype=np.float32)
    sub_clf.intercept_ = np.asarray(classifier.intercept_, dtype=np.float32)
    sub_clf.n_features_in_ = len(selected_idx)

    return {
        "vectorizer": sub_vec,
        "classifier": sub_clf,
        "mode": mode,
        "ngram_range": ngram_range,
        "full_accuracy": full_accuracy,
        "compact_accuracy": compact_accuracy,
    }


def save_deployable_svm_bundle(bundle: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Menyimpan bundel model SVM ke berkas joblib terkompresi."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    return out


def load_compact_svm_bundles(
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Memuat model SVM terlatih (LinearSVC C=10) untuk Medan Makna dan Semantik Frame."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    bundles: Dict[str, Dict[str, Any]] = {}
    for key, filename in (
        ("medan_makna", "svm_medan_makna.joblib"),
        ("frame_semantik", "svm_frame_semantik.joblib"),
    ):
        path = base / filename
        if path.exists():
            bundles[key] = joblib.load(path)
    return bundles


def _model_slug(model_key: str) -> str:
    return MODEL_FILE_SLUGS.get(model_key, model_key.lower())


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    scheme: str,
    models_dir: Optional[Union[str, Path]] = None,
    compress: int = 3,
) -> Path:
    """Menyimpan TfidfVectorizer terlatih ke models/all_models/ untuk inferensi ulang ke-5 model."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    out_path = base / "all_models" / f"vectorizer_{scheme}.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_path, compress=compress)
    return out_path


def save_reusable_model(
    classifier: Any,
    model_key: str,
    scheme: str,
    ngram_range: Tuple[int, int],
    accuracy: Optional[float] = None,
    models_dir: Optional[Union[str, Path]] = None,
    compress: int = 3,
) -> Path:
    """Menyimpan model klasifikasi terlatih (semua 5 algoritma) ke models/all_models/ sebagai .joblib."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    slug = _model_slug(model_key)
    out_path = base / "all_models" / f"{slug}_{scheme}.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": model_key,
        "scheme": scheme,
        "ngram_range": ngram_range,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "classifier": classifier,
    }
    joblib.dump(payload, out_path, compress=compress)
    return out_path


def load_reusable_model(
    model_key: str,
    scheme: str = "frame_semantik",
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Memuat model terlatih (.joblib) beserta TfidfVectorizer pasangannya dari models/all_models/."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    all_dir = base / "all_models"
    slug = _model_slug(model_key)
    model_path = all_dir / f"{slug}_{scheme}.joblib"
    vec_path = all_dir / f"vectorizer_{scheme}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Berkas model belum ditemukan: {model_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"Berkas vectorizer belum ditemukan: {vec_path}")

    payload = dict(joblib.load(model_path))
    payload["vectorizer"] = joblib.load(vec_path)
    return payload


def predict_with_reusable_model(
    text: str,
    model_key: str,
    scheme: str = "frame_semantik",
    models_dir: Optional[Union[str, Path]] = None,
    stopwords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Melakukan pra-pemrosesan dan prediksi kategori menggunakan salah satu dari 5 model .joblib tersimpan."""
    bundle = load_reusable_model(model_key=model_key, scheme=scheme, models_dir=models_dir)
    if scheme == "medan_makna":
        stops = stopwords if stopwords is not None else load_stopwords()
        cleaned = preprocess_text(text, stopwords=stops)
    elif scheme == "frame_semantik":
        cleaned = preprocess_text(text, stopwords=None)
    else:
        raise ValueError(f"Skema tidak dikenal: {scheme}. Gunakan 'frame_semantik' atau 'medan_makna'.")

    clf = bundle["classifier"]
    vec = bundle["vectorizer"]
    X = vec.transform([cleaned])
    pred_label = int(clf.predict(X)[0])

    return {
        "model_key": bundle["model_key"],
        "scheme": scheme,
        "cleaned_text": cleaned,
        "predicted_label": pred_label,
        "predicted_category": LABELS.get(pred_label, str(pred_label)),
        "predicted_category_id": LABELS_ID.get(pred_label, str(pred_label)),
        "predicted_category_en": LABELS_EN.get(pred_label, str(pred_label)),
        "validation_accuracy": bundle.get("accuracy"),
    }


def extract_features_and_svm_weights(
    text: str,
    vectorizer,
    classifier,
    cleaned_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Memprediksi kategori, skor sigmoid 11 kelas, bobot TF-IDF, dan koefisien SVM tiap fitur."""
    input_text = cleaned_text if cleaned_text is not None else str(text)
    tfidf_mat = vectorizer.transform([input_text])
    row_vec = tfidf_mat[0]
    active_indices = row_vec.indices
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", str(text)))
    is_uncategorized = (len(active_indices) == 0) or (not has_arabic)

    raw_scores = classifier.decision_function(tfidf_mat)[0]
    sigmoid_probs = sigmoid(raw_scores) * 100.0
    classes = [int(c) for c in classifier.classes_]

    category_scores: List[Dict[str, Any]] = []
    for idx, cid in enumerate(classes):
        category_scores.append(
            {
                "label": cid,
                "category": LABELS.get(cid, str(cid)),
                "category_id": LABELS_ID.get(cid, str(cid)),
                "category_en": LABELS_EN.get(cid, str(cid)),
                "decision_score": float(raw_scores[idx]),
                "sigmoid_percent": float(sigmoid_probs[idx]),
            }
        )
    category_scores.sort(key=lambda item: item["sigmoid_percent"], reverse=True)

    if is_uncategorized:
        return {
            "is_uncategorized": True,
            "cleaned_text": input_text,
            "predicted_label": None,
            "predicted_category": "غير مصنف",
            "predicted_category_id": "Tidak Terkategori",
            "predicted_category_en": "Uncategorized",
            "intercept": 0.0,
            "total_tfidf_weight": 0.0,
            "total_positive_coef": 0.0,
            "total_negative_coef": 0.0,
            "n_active_features": 0,
            "category_scores": category_scores,
            "features_by_tfidf": [],
            "positive_coefs": [],
            "negative_coefs": [],
        }

    pred_label = int(classifier.predict(tfidf_mat)[0])
    pred_class_idx = classes.index(pred_label)
    feature_names = vectorizer.get_feature_names_out()
    coef_row = classifier.coef_[pred_class_idx]

    features_detail: List[Dict[str, Any]] = []
    for idx in active_indices:
        tfidf_w = float(row_vec[0, idx])
        svm_w = float(coef_row[idx])
        features_detail.append(
            {
                "term": str(feature_names[idx]),
                "n_tokens": len(str(feature_names[idx]).split()),
                "tfidf_weight": tfidf_w,
                "svm_coef": svm_w,
                "contribution": tfidf_w * svm_w,
            }
        )

    features_by_tfidf = sorted(features_detail, key=lambda x: x["tfidf_weight"], reverse=True)
    positive_coefs = sorted(
        [f for f in features_detail if f["svm_coef"] > 0],
        key=lambda x: x["svm_coef"],
        reverse=True,
    )
    negative_coefs = sorted(
        [f for f in features_detail if f["svm_coef"] < 0],
        key=lambda x: x["svm_coef"],
    )

    return {
        "is_uncategorized": False,
        "cleaned_text": input_text,
        "predicted_label": pred_label,
        "predicted_category": LABELS.get(pred_label, str(pred_label)),
        "predicted_category_id": LABELS_ID.get(pred_label, str(pred_label)),
        "predicted_category_en": LABELS_EN.get(pred_label, str(pred_label)),
        "intercept": float(classifier.intercept_[pred_class_idx]),
        "total_tfidf_weight": float(np.sum(row_vec.data)),
        "total_positive_coef": float(sum(f["svm_coef"] for f in positive_coefs)),
        "total_negative_coef": float(sum(f["svm_coef"] for f in negative_coefs)),
        "n_active_features": len(features_detail),
        "category_scores": category_scores,
        "features_by_tfidf": features_by_tfidf,
        "positive_coefs": positive_coefs,
        "negative_coefs": negative_coefs,
    }


def analyze_article_dual_svm(
    text: str,
    bundles: Optional[Dict[str, Dict[str, Any]]] = None,
    stopwords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Menjalankan analisis ganda (Medan Makna + Semantik Frame) pada satu artikel berbahasa Arab."""
    active_bundles = bundles if bundles is not None else load_compact_svm_bundles()
    stops = stopwords if stopwords is not None else load_stopwords()

    output: Dict[str, Any] = {}
    if "medan_makna" in active_bundles:
        b_m = active_bundles["medan_makna"]
        cleaned_medan = preprocess_text(text, stopwords=stops)
        output["medan_makna"] = extract_features_and_svm_weights(
            text=text,
            vectorizer=b_m["vectorizer"],
            classifier=b_m["classifier"],
            cleaned_text=cleaned_medan,
        )

    if "frame_semantik" in active_bundles:
        b_f = active_bundles["frame_semantik"]
        cleaned_frame = preprocess_text(text, stopwords=None)
        output["frame_semantik"] = extract_features_and_svm_weights(
            text=text,
            vectorizer=b_f["vectorizer"],
            classifier=b_f["classifier"],
            cleaned_text=cleaned_frame,
        )

    return output
