"""Skrip CLI hemat RAM dan mendukung resume (--resume) untuk melatih & menyimpan seluruh 5 model."""

import argparse
from datetime import datetime
import gc
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arabic_news.config import (
    FIGURES_DIR,
    LABELS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    resolve_data_file,
)
from arabic_news.evaluation import (
    evaluate_model,
    parse_evaluation_metrics_txt,
    parse_resource_log_txt,
    save_evaluation_metrics_txt,
    save_resource_log_txt,
    split_dataset_80_15_5,
    split_dataset_85_10_5,
    train_classifier_with_monitoring,
)
from arabic_news.features import create_tfidf_vectorizer
from arabic_news.models import (
    MODEL_FILE_SLUGS,
    build_deployable_svm_bundle,
    predict_with_reusable_model,
    save_deployable_svm_bundle,
    save_reusable_model,
    save_vectorizer,
)
from arabic_news.preprocessing import load_stopwords, preprocess_text, reshape_arabic_for_plot
from arabic_news.visualization import find_arabic_font_path

MODELS_ORDER: List[Tuple[str, str, str, str]] = [
    ("SVC", "SVM (LinearSVC C=10)", "svm.txt", "SVM"),
    ("LogisticRegression", "Logistic Regression", "logistic.txt", "LogReg"),
    ("MultinomialNB", "Naïve Bayes", "naive.txt", "NaiveBayes"),
    ("RandomForest", "Random Forest", "forest.txt", "RandomForest"),
    ("GradientBoosting", "Gradient Boosting", "gradient.txt", "GradBoost"),
]

SCHEMES_CONFIG: Dict[str, Dict[str, Any]] = {
    "medan_makna": {
        "metrics_subdir": "medan_semantik",
        "ngram_range": (1, 1),
        "compact_max_features": 500_000,
        "compact_filename": "svm_medan_makna.joblib",
        "label": "Medan Makna (1,1 — 80/15/5 — hapus stopwords)",
    },
    "frame_semantik": {
        "metrics_subdir": "frame_semantik",
        "ngram_range": (2, 3),
        "compact_max_features": 700_000,
        "compact_filename": "svm_frame_semantik.joblib",
        "label": "Semantik Frame (2,3 — 85/10/5 — retensi preposisi)",
    },
}


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _checkpoint_paths(
    model_key: str,
    res_file: str,
    scheme: str,
    metrics_dir: Path,
    models_dir: Path,
) -> List[Path]:
    subdir = SCHEMES_CONFIG[scheme]["metrics_subdir"]
    slug = MODEL_FILE_SLUGS[model_key]
    paths = [
        metrics_dir / subdir / f"no_lemma_evaluation_metrics_{model_key}.txt",
        metrics_dir / "cpu_and_memory" / subdir / res_file,
        models_dir / "all_models" / f"{slug}_{scheme}.joblib",
        models_dir / "all_models" / f"vectorizer_{scheme}.joblib",
    ]
    if model_key == "SVC":
        paths.append(models_dir / SCHEMES_CONFIG[scheme]["compact_filename"])
    return paths


def _checkpoint_exists(
    model_key: str,
    res_file: str,
    scheme: str,
    metrics_dir: Path,
    models_dir: Path,
) -> bool:
    return all(p.exists() for p in _checkpoint_paths(model_key, res_file, scheme, metrics_dir, models_dir))


def _load_split_and_vectorize_scheme(
    dataset_path: Path,
    scheme: str,
    models_dir: Path,
    compress: int = 1,
) -> Tuple[Any, Any, Any, Optional[List[str]]]:
    """Memuat satu skema secara terpisah lalu langsung membebaskan DataFrame mentah dari RAM."""
    t0 = time.perf_counter()
    _log(f"Membaca kolom [label, text] dari {dataset_path.name} untuk {scheme}...")
    df_raw = pd.read_csv(dataset_path, usecols=["label", "text"], low_memory=False)

    if scheme == "medan_makna":
        stops = set(load_stopwords())
        tr_raw, val_raw, _te_raw, y_tr, y_val, _y_te = split_dataset_80_15_5(
            df_raw, random_state=RANDOM_SEED, shuffle_first=True
        )
        del df_raw, _te_raw, _y_te
        gc.collect()

        _log("Pra-pemrosesan Medan Makna (penghapusan stopwords)...")
        tr_texts = [preprocess_text(t, stopwords=stops) for t in tr_raw]
        val_texts = [preprocess_text(t, stopwords=stops) for t in val_raw]
        del tr_raw, val_raw
        gc.collect()

        ngram_range = (1, 1)
    elif scheme == "frame_semantik":
        tr_df, val_df, _te_df = split_dataset_85_10_5(df_raw, random_state=RANDOM_SEED)
        del df_raw, _te_df
        gc.collect()

        _log("Pra-pemrosesan Semantik Frame (retensi preposisi/stopwords)...")
        tr_texts = [preprocess_text(t, stopwords=None) for t in tr_df["text"]]
        y_tr = tr_df["label"].reset_index(drop=True)
        val_texts = [preprocess_text(t, stopwords=None) for t in val_df["text"]]
        y_val = val_df["label"].reset_index(drop=True)
        del tr_df, val_df
        gc.collect()

        ngram_range = (2, 3)
    else:
        raise ValueError(f"Skema tidak dikenal: {scheme}")

    _log(f"Vektorisasi TF-IDF {ngram_range} (train={len(tr_texts):,}, val={len(val_texts):,})...")
    vec = create_tfidf_vectorizer(ngram_range=ngram_range)
    X_tr = vec.fit_transform(tr_texts)
    del tr_texts
    gc.collect()

    X_val = vec.transform(val_texts)
    save_vectorizer(vec, scheme, models_dir=models_dir, compress=compress)
    _log(f"Siap dalam {time.perf_counter() - t0:.1f}s | Dimensi latih={X_tr.shape}")
    return vec, X_tr, y_tr, X_val, y_val, val_texts


def run_scheme(
    dataset_path: Path,
    scheme: str,
    selected_models: Sequence[str],
    force: bool,
    metrics_dir: Path,
    models_dir: Path,
    compress: int = 1,
    monitor_interval: float = 0.5,
) -> None:
    """Melatih & menyimpan model-model terpilih untuk satu skema representasi fitur."""
    cfg = SCHEMES_CONFIG[scheme]
    models_to_run = [m for m in MODELS_ORDER if m[0] in selected_models]
    pending = [
        m for m in models_to_run
        if force or not _checkpoint_exists(m[0], m[2], scheme, metrics_dir, models_dir)
    ]

    if not pending:
        _log(f"[{scheme}] Semua model terpilih sudah tersedia (lewati berkat checkpoint).")
        return

    _log(f"=== Mulai skema: {cfg['label']} | Target: {[m[0] for m in pending]} ===")
    vec, X_tr, y_tr, X_val, y_val, val_texts = _load_split_and_vectorize_scheme(
        dataset_path=dataset_path,
        scheme=scheme,
        models_dir=models_dir,
        compress=compress,
    )

    try:
        for key, display_name, res_file, _short in pending:
            t_m = time.perf_counter()
            _log(f"[{scheme}] Melatih {display_name}...")
            clf, res_stats = train_classifier_with_monitoring(
                key,
                X_tr,
                y_tr,
                mode="frame",
                interval=monitor_interval,
                random_state=RANDOM_SEED,
            )
            ev = evaluate_model(clf, X_val, y_val)

            subdir = cfg["metrics_subdir"]
            save_evaluation_metrics_txt(
                ev, key, metrics_dir / subdir / f"no_lemma_evaluation_metrics_{key}.txt"
            )
            save_resource_log_txt(
                res_stats, key, metrics_dir / "cpu_and_memory" / subdir / res_file
            )
            model_out = save_reusable_model(
                clf,
                model_key=key,
                scheme=scheme,
                ngram_range=cfg["ngram_range"],
                accuracy=ev["accuracy"],
                models_dir=models_dir,
                compress=compress,
            )
            _log(
                f"[{scheme}] Selesai {display_name} dalam {time.perf_counter() - t_m:.1f}s | "
                f"Acc={ev['accuracy']:.4f} | RAM={res_stats['max_memory_mb']:.1f} MB | File={model_out.name}"
            )

            if key == "SVC":
                bundle = build_deployable_svm_bundle(
                    vec,
                    clf,
                    mode=scheme,
                    ngram_range=cfg["ngram_range"],
                    max_features=cfg["compact_max_features"],
                    full_accuracy=ev["accuracy"],
                )
                ev_compact = evaluate_model(
                    bundle["classifier"],
                    bundle["vectorizer"].transform(val_texts),
                    y_val,
                )
                bundle["compact_accuracy"] = ev_compact["accuracy"]
                compact_out = save_deployable_svm_bundle(bundle, models_dir / cfg["compact_filename"])
                _log(
                    f"[{scheme}] Bundel Streamlit {compact_out.name} tersimpan "
                    f"(compact acc={ev_compact['accuracy']:.4f}, {compact_out.stat().st_size / (1024 * 1024):.1f} MB)"
                )
                del val_texts
                val_texts = []
                del bundle

            del clf
            gc.collect()
    finally:
        del vec, X_tr, y_tr, X_val, y_val, val_texts
        gc.collect()


def rebuild_summary_and_figures(
    metrics_dir: Path,
    figures_dir: Path,
) -> Dict[str, Any]:
    """Menggabungkan hasil evaluasi seluruh model yang telah selesai ke CSV perbandingan dan grafik."""
    eff_rows: List[Dict[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []
    short_names: List[str] = []
    svc_frame_detail = None

    for key, display_name, res_file, short_name in MODELS_ORDER:
        mf_path = metrics_dir / "frame_semantik" / f"no_lemma_evaluation_metrics_{key}.txt"
        mn_path = metrics_dir / "medan_semantik" / f"no_lemma_evaluation_metrics_{key}.txt"
        if not mf_path.exists() or not mn_path.exists():
            continue

        mf = parse_evaluation_metrics_txt(mf_path)
        mn = parse_evaluation_metrics_txt(mn_path)
        rf_path = metrics_dir / "cpu_and_memory" / "frame_semantik" / res_file
        rn_path = metrics_dir / "cpu_and_memory" / "medan_semantik" / res_file
        rf = parse_resource_log_txt(rf_path) if rf_path.exists() else {"max_memory_mb": 0.0, "avg_cpu_percent": 0.0}
        rn = parse_resource_log_txt(rn_path) if rn_path.exists() else {"max_memory_mb": 0.0, "avg_cpu_percent": 0.0}

        mem_inc = (
            round((rf["max_memory_mb"] - rn["max_memory_mb"]) / rn["max_memory_mb"] * 100, 2)
            if rn["max_memory_mb"]
            else 0.0
        )
        eff_rows.append(
            {
                "Model": display_name,
                "Waktu Frame (s)": round(mf["execution_time_seconds"], 4),
                "Waktu Medan (s)": round(mn["execution_time_seconds"], 4),
                "RAM Frame (MB)": round(rf["max_memory_mb"], 2),
                "RAM Medan (MB)": round(rn["max_memory_mb"], 2),
                "Kenaikan RAM (%)": mem_inc,
                "CPU Frame (%)": round(rf["avg_cpu_percent"], 2),
                "CPU Medan (%)": round(rn["avg_cpu_percent"], 2),
            }
        )
        perf_rows.append(
            {
                "Model": display_name,
                "Acc Frame (2,3)": round(mf["accuracy"], 4),
                "Prec Frame": round(mf["precision"], 4),
                "Rec Frame": round(mf["recall"], 4),
                "F1 Frame": round(mf["f1_score"], 4),
                "Support Frame": int(mf["support"]),
                "Acc Medan (1,1)": round(mn["accuracy"], 4),
                "Prec Medan": round(mn["precision"], 4),
                "Rec Medan": round(mn["recall"], 4),
                "F1 Medan": round(mn["f1_score"], 4),
                "Support Medan": int(mn["support"]),
            }
        )
        short_names.append(short_name)
        if key == "SVC":
            svc_frame_detail = mf

    eff_df = pd.DataFrame(eff_rows)
    perf_df = pd.DataFrame(perf_rows)
    if not perf_df.empty:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        eff_df.to_csv(metrics_dir / "efficiency_comparison.csv", index=False)
        perf_df.to_csv(metrics_dir / "effectiveness_comparison.csv", index=False)

        x = np.arange(len(perf_df))
        w = 0.35
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        min_acc = float(min(perf_df["Acc Frame (2,3)"].min(), perf_df["Acc Medan (1,1)"].min()) * 100)
        axes[0].bar(x - w / 2, perf_df["Acc Frame (2,3)"] * 100, w, label="Dengan Frame (2,3)")
        axes[0].bar(x + w / 2, perf_df["Acc Medan (1,1)"] * 100, w, label="Tanpa Frame / Medan (1,1)")
        axes[0].set_title("Perbandingan Akurasi (%)")
        axes[0].set_ylim(max(0.0, min_acc - 5.0), 100.0)
        axes[0].set_xticks(x, short_names, rotation=20)
        axes[0].legend()

        axes[1].bar(x - w / 2, eff_df["Waktu Frame (s)"], w, label="Dengan Frame (2,3)")
        axes[1].bar(x + w / 2, eff_df["Waktu Medan (s)"], w, label="Tanpa Frame / Medan (1,1)")
        axes[1].set_title("Waktu Evaluasi (detik)")
        axes[1].set_xticks(x, short_names, rotation=20)
        axes[1].legend()

        axes[2].bar(x - w / 2, eff_df["RAM Frame (MB)"], w, label="Dengan Frame (2,3)")
        axes[2].bar(x + w / 2, eff_df["RAM Medan (MB)"], w, label="Tanpa Frame / Medan (1,1)")
        axes[2].set_title("Konsumsi RAM Puncak (MB)")
        axes[2].set_xticks(x, short_names, rotation=20)
        axes[2].legend()
        plt.tight_layout()
        fig.savefig(figures_dir / "model_comparison.png", dpi=150)
        plt.close(fig)

    if svc_frame_detail is not None and svc_frame_detail.get("confusion_matrix") is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        labels_ar = [reshape_arabic_for_plot(LABELS[i]) for i in range(1, 12)]
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8.5))
        sns.heatmap(
            svc_frame_detail["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_ar,
            yticklabels=labels_ar,
            ax=ax_cm,
        )
        ax_cm.set_title(
            f"Confusion Matrix SVM Dengan Semantik Frame (Akurasi: {svc_frame_detail['accuracy'] * 100:.2f}%)"
        )
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        font_path = find_arabic_font_path()
        if font_path:
            from matplotlib import font_manager as fm

            prop = fm.FontProperties(fname=font_path)
            for lbl in ax_cm.get_xticklabels() + ax_cm.get_yticklabels():
                lbl.set_fontproperties(prop)
        plt.tight_layout()
        fig_cm.savefig(figures_dir / "confusion_matrix_svm_frame.png", dpi=150)
        plt.close(fig_cm)

    return {"efficiency": eff_df, "performance": perf_df}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pelatihan hemat RAM & resumable untuk ke-5 algoritma klasifikasi berita Arab."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m[0] for m in MODELS_ORDER],
        default=[m[0] for m in MODELS_ORDER],
        help="Daftar algoritma yang dilatih (default: semua 5 algoritma).",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["medan_makna", "frame_semantik"],
        default=["medan_makna", "frame_semantik"],
        help="Urutan skema representasi (default: medan_makna lalu frame_semantik agar RAM optimal).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Latih ulang meski berkas metrik & .joblib checkpoint sudah ada.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path kustom ke basic_articles_dataset.csv (opsional).",
    )
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--compress",
        type=int,
        default=1,
        help="Level kompresi joblib untuk models/all_models/ (default: 1 untuk kecepatan I/O M1).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_path = args.dataset if args.dataset else resolve_data_file("basic_articles_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {dataset_path}. Letakkan basic_articles_dataset.csv di folder data/."
        )

    t_start = time.perf_counter()
    for scheme in args.schemes:
        run_scheme(
            dataset_path=dataset_path,
            scheme=scheme,
            selected_models=args.models,
            force=args.force,
            metrics_dir=args.metrics_dir,
            models_dir=args.models_dir,
            compress=args.compress,
        )

    summary = rebuild_summary_and_figures(metrics_dir=args.metrics_dir, figures_dir=args.figures_dir)
    if not summary["performance"].empty:
        _log("Ringkasan Efektivitas Model Tersedia:")
        print(summary["performance"].to_string(index=False), flush=True)
        _log("Ringkasan Efisiensi Komputasi Tersedia:")
        print(summary["efficiency"].to_string(index=False), flush=True)
    else:
        _log("Belum ada pasangan metrik lengkap frame_semantik + medan_semantik untuk diringkas.")

    _log(f"Seluruh proses selesai dalam {time.perf_counter() - t_start:.1f} detik.")


if __name__ == "__main__":
    main()
