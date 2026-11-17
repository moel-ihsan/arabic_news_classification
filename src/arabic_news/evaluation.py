"""Utilitas pembagian dataset, evaluasi model, pemantauan CPU/RAM, serta penyimpanan metrik."""

from datetime import datetime
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .config import RANDOM_SEED
from .models import train_classifier
from .preprocessing import load_stopwords, preprocess_text


def _clean_corpus_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Membersihkan baris teks kosong dan memvalidasi label numerik dari basic_articles_dataset.csv."""
    clean_df = df.dropna(subset=[text_col]).copy()
    clean_df[text_col] = clean_df[text_col].astype(str)
    clean_df[label_col] = pd.to_numeric(clean_df[label_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[label_col])
    clean_df[label_col] = clean_df[label_col].astype(int)
    return clean_df


def split_dataset_80_15_5(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    random_state: int = RANDOM_SEED,
    shuffle_first: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Membagi dataset menjadi 80% latih, 15% validasi, dan 5% uji untuk pendekatan Medan Makna."""
    clean_df = _clean_corpus_dataframe(df, text_col=text_col, label_col=label_col)

    if shuffle_first:
        clean_df = shuffle(clean_df, random_state=random_state)

    train_data, test_data, train_labels, test_labels = train_test_split(
        clean_df[text_col],
        clean_df[label_col],
        test_size=0.05,
        random_state=random_state,
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data,
        train_labels,
        test_size=0.15 / 0.95,
        random_state=random_state,
    )
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def split_dataset_85_10_5(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membagi setiap kategori dari basic_articles_dataset.csv menjadi 85% latih, 10% validasi, dan 5% uji."""
    clean_df = _clean_corpus_dataframe(df, text_col=text_col, label_col=label_col)

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    for cid in sorted(clean_df[label_col].unique()):
        sub = clean_df[clean_df[label_col] == cid]
        n = len(sub)
        n_train = round(n * 0.85)
        n_test = round(n * 0.05)
        train_parts.append(sub.iloc[:n_train])
        test_parts.append(sub.iloc[n_train : n_train + n_test])
        val_parts.append(sub.iloc[n_train + n_test :])

    train_df = shuffle(pd.concat(train_parts, ignore_index=True), random_state=random_state)
    val_df = shuffle(pd.concat(val_parts, ignore_index=True), random_state=random_state)
    test_df = shuffle(pd.concat(test_parts, ignore_index=True), random_state=random_state)
    return train_df, val_df, test_df


def select_one_sample_per_category(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Mengambil tepat satu artikel uji representatif untuk masing-masing kategori (1..11)."""
    samples: List[pd.DataFrame] = []
    for cid in sorted(df[label_col].dropna().astype(int).unique()):
        sub = df[df[label_col].astype(int) == cid]
        samples.append(sub.sample(n=1, random_state=random_state + cid))
    return pd.concat(samples, ignore_index=True).sort_values(label_col).reset_index(drop=True)


def prepare_medan_and_frame_splits(
    df: pd.DataFrame,
    stopwords: Optional[Iterable[str]] = None,
    text_col: str = "text",
    label_col: str = "label",
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Menyiapkan split Medan Makna (80/15/5 + hapus stopwords) dan Semantik Frame (85/10/5 tanpa hapus stopwords) dari satu korpus."""
    stop_set = set(stopwords) if stopwords is not None else set(load_stopwords())

    tr_m_raw, val_m_raw, te_m_raw, y_tr_m, y_val_m, y_te_m = split_dataset_80_15_5(
        df, text_col=text_col, label_col=label_col, random_state=random_state, shuffle_first=True
    )
    tr_f_df, val_f_df, te_f_df = split_dataset_85_10_5(
        df, text_col=text_col, label_col=label_col, random_state=random_state
    )

    return {
        "medan_makna": {
            "train_text": tr_m_raw.apply(lambda x: preprocess_text(x, stopwords=stop_set)),
            "val_text": val_m_raw.apply(lambda x: preprocess_text(x, stopwords=stop_set)),
            "test_text": te_m_raw.apply(lambda x: preprocess_text(x, stopwords=stop_set)),
            "train_labels": y_tr_m,
            "val_labels": y_val_m,
            "test_labels": y_te_m,
        },
        "frame_semantik": {
            "train_df": tr_f_df.assign(
                **{text_col: tr_f_df[text_col].apply(lambda x: preprocess_text(x, stopwords=None))}
            ),
            "val_df": val_f_df.assign(
                **{text_col: val_f_df[text_col].apply(lambda x: preprocess_text(x, stopwords=None))}
            ),
            "test_df": te_f_df.assign(
                **{text_col: te_f_df[text_col].apply(lambda x: preprocess_text(x, stopwords=None))}
            ),
        },
    }


def evaluate_model(classifier, val_tfidf, val_labels) -> Dict[str, Any]:
    """Mengevaluasi model pada data validasi dan mengembalikan metrik serta laporan teks."""
    start_time = datetime.now()
    predictions = classifier.predict(val_tfidf)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    accuracy = float(accuracy_score(val_labels, predictions))
    cm = confusion_matrix(val_labels, predictions)
    report_dict = classification_report(
        val_labels, predictions, zero_division=1, digits=4, output_dict=True
    )
    report_text = classification_report(
        val_labels, predictions, zero_division=1, digits=4
    )
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_text": report_text,
        "execution_time_seconds": execution_time,
    }


def save_evaluation_metrics_txt(
    eval_res: Dict[str, Any],
    model_key: str,
    output_path: Union[str, Path],
) -> Path:
    """Menyimpan metrik akurasi, confusion matrix, dan classification report ke berkas teks."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cm_str = np.array2string(eval_res["confusion_matrix"], max_line_width=1000)
    content = (
        f"{model_key}\n"
        f"{model_key} Accuracy: {eval_res['accuracy']:.4f}\n\n"
        f"Confusion Matrix:\n{cm_str}\n\n"
        f"Classification Report:\n{eval_res['report_text']}\n"
        f"Execution Time: {eval_res['execution_time_seconds']:.6f} seconds\n"
    )
    out.write_text(content, encoding="utf-8")
    return out


def save_resource_log_txt(
    res_stats: Dict[str, float],
    model_key: str,
    output_path: Union[str, Path],
) -> Path:
    """Menyimpan log penggunaan CPU rata-rata dan puncak memori RAM ke berkas teks."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"Model: {model_key}\n"
        f"{model_key} Total CPU Usage: {res_stats['avg_cpu_percent']:.2f}% | "
        f"Max Memory Usage: {res_stats['max_memory_mb']:.2f} MB\n"
    )
    out.write_text(content, encoding="utf-8")
    return out


def _monitor_resource_usage(
    classifier_name: str,
    interval: float,
    stop_event: threading.Event,
    stats_out: Dict[str, float],
) -> None:
    """Thread pemantau persentase CPU dan memori RSS (MB) secara berkala menggunakan psutil."""
    process = psutil.Process()
    total_cpu_usage = 0.0
    max_memory_usage = 0.0
    count = 0

    while not stop_event.is_set():
        memory_usage = process.memory_info().rss / 1024 / 1024
        cpu_usage = process.cpu_percent(interval=interval)
        total_cpu_usage += cpu_usage
        max_memory_usage = max(max_memory_usage, memory_usage)
        time.sleep(interval)
        count += 1

    # Catat sampel akhir jika pelatihan selesai lebih cepat dari interval
    if count == 0:
        max_memory_usage = process.memory_info().rss / 1024 / 1024
        total_cpu_usage = process.cpu_percent(interval=None)
        count = 1

    stats_out["avg_cpu_percent"] = float(total_cpu_usage / count)
    stats_out["max_memory_mb"] = float(max_memory_usage)


def train_classifier_with_monitoring(
    classify_with: str,
    train_tfidf,
    train_labels,
    mode: str = "frame",
    interval: float = 0.5,
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, Dict[str, float]]:
    """Melatih model sambil mencatat rata-rata penggunaan CPU (%) dan puncak memori RSS (MB)."""
    stop_event = threading.Event()
    resource_stats: Dict[str, float] = {"avg_cpu_percent": 0.0, "max_memory_mb": 0.0}
    monitor_thread = threading.Thread(
        target=_monitor_resource_usage,
        args=(classify_with, interval, stop_event, resource_stats),
        daemon=True,
    )
    monitor_thread.start()
    try:
        classifier = train_classifier(
            classify_with, train_tfidf, train_labels, mode=mode, random_state=random_state
        )
    finally:
        stop_event.set()
        monitor_thread.join()

    return classifier, resource_stats


def parse_evaluation_metrics_txt(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Mem-parsing berkas laporan evaluasi model menjadi dictionary terstruktur."""
    content = Path(filepath).read_text(encoding="utf-8")
    lines = [line.rstrip() for line in content.splitlines()]
    model_name = lines[0].strip() if lines else ""

    acc_match = re.search(r"Accuracy:\s*([0-9.]+)", content)
    accuracy = float(acc_match.group(1)) if acc_match else None

    time_match = re.search(r"Execution Time:\s*([0-9.]+)\s*seconds", content)
    exec_time = float(time_match.group(1)) if time_match else None

    cm_rows: List[List[int]] = []
    in_cm = False
    for line in lines:
        if "Confusion Matrix:" in line:
            in_cm = True
            continue
        if in_cm:
            if "Classification Report:" in line:
                break
            nums = re.findall(r"\d+", line)
            if nums:
                cm_rows.append([int(n) for n in nums])
    cm = np.array(cm_rows, dtype=int) if cm_rows else None

    per_class: Dict[int, Dict[str, float]] = {}
    weighted_avg: Dict[str, float] = {}
    macro_avg: Dict[str, float] = {}
    total_support = None

    for line in lines:
        cls_m = re.match(
            r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line
        )
        if cls_m:
            cid = int(cls_m.group(1))
            per_class[cid] = {
                "precision": float(cls_m.group(2)),
                "recall": float(cls_m.group(3)),
                "f1-score": float(cls_m.group(4)),
                "support": int(cls_m.group(5)),
            }
            continue
        macro_m = re.match(
            r"^\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line
        )
        if macro_m:
            macro_avg = {
                "precision": float(macro_m.group(1)),
                "recall": float(macro_m.group(2)),
                "f1-score": float(macro_m.group(3)),
                "support": int(macro_m.group(4)),
            }
            total_support = int(macro_m.group(4))
            continue
        weighted_m = re.match(
            r"^\s*weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line
        )
        if weighted_m:
            weighted_avg = {
                "precision": float(weighted_m.group(1)),
                "recall": float(weighted_m.group(2)),
                "f1-score": float(weighted_m.group(3)),
                "support": int(weighted_m.group(4)),
            }

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": weighted_avg.get("precision"),
        "recall": weighted_avg.get("recall"),
        "f1_score": weighted_avg.get("f1-score"),
        "support": total_support,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "per_class": per_class,
        "confusion_matrix": cm,
        "execution_time_seconds": exec_time,
    }


def parse_resource_log_txt(filepath: Union[str, Path]) -> Dict[str, Optional[float]]:
    """Mem-parsing persentase CPU rata-rata dan penggunaan RAM puncak (MB) dari log komputasi."""
    content = Path(filepath).read_text(encoding="utf-8")
    matches = re.findall(
        r"Total CPU Usage:\s*([0-9.]+)%\s*\|\s*Max Memory Usage:\s*([0-9.]+)\s*MB",
        content,
    )
    if not matches:
        return {"avg_cpu_percent": None, "max_memory_mb": None}
    last_cpu, last_mem = matches[-1]
    return {
        "avg_cpu_percent": float(last_cpu),
        "max_memory_mb": float(last_mem),
    }
