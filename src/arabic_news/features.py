"""Ekstraksi fitur TF-IDF unigram (Medan Makna), bigram/trigram (Semantik Frame), dan statistik n-gram."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import LABELS
from .preprocessing import preprocess_ngram_tokenizer


def create_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (2, 3),
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> TfidfVectorizer:
    """Membuat TfidfVectorizer untuk representasi Medan Makna (unigram) atau Semantik Frame (bigram–trigram)."""
    return TfidfVectorizer(ngram_range=ngram_range, tokenizer=tokenizer)


def extract_all_keywords(
    tfidf_matrix,
    feature_names: np.ndarray,
) -> List[Tuple[List[Tuple[str, float]], float]]:
    """Mengekstrak pasangan (frasa, bobot TF-IDF) beserta total bobot setiap dokumen."""
    keywords_list: List[Tuple[List[Tuple[str, float]], float]] = []
    for i in range(tfidf_matrix.shape[0]):
        doc = tfidf_matrix[i, :]
        keyword_indices = doc.indices
        keywords = [(feature_names[idx], float(doc[0, idx])) for idx in keyword_indices]
        total_weight = float(np.sum(doc.data))
        keywords_list.append((keywords, total_weight))
    return keywords_list


def extract_ngrams_with_weights_by_label(
    df: pd.DataFrame,
    category_id: int,
    category_column: str = "label",
    text_column: str = "text",
    ngram_range: Tuple[int, int] = (2, 3),
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Menghitung rata-rata bobot TF-IDF bigram dan trigram untuk membentuk kamus satu kategori."""
    category_data = df[df[category_column] == category_id]
    if category_data.empty:
        return pd.DataFrame(columns=["N-gram", "TF-IDF Weight"])

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        tokenizer=preprocess_ngram_tokenizer,
    )
    tfidf_matrix = vectorizer.fit_transform(category_data[text_column].astype(str))
    ngrams = vectorizer.get_feature_names_out()
    tfidf_weights = tfidf_matrix.mean(axis=0).A1

    ngrams_df = pd.DataFrame({"N-gram": ngrams, "TF-IDF Weight": tfidf_weights})
    ngrams_df = ngrams_df.sort_values(by="TF-IDF Weight", ascending=False).reset_index(drop=True)
    if top_n is not None:
        ngrams_df = ngrams_df.head(top_n)
    return ngrams_df


def count_top_ngrams_by_category(
    df: pd.DataFrame,
    n: int = 2,
    category_column: str = "label",
    text_column: str = "text",
    top_n: int = 1000,
    use_regex_tokenizer: bool = True,
) -> Dict[str, List[Tuple[str, int]]]:
    """Menghitung frekuensi top-N unigram/bigram/trigram per kategori secara tervektorisasi."""
    tokenizer_fn = preprocess_ngram_tokenizer if use_regex_tokenizer else str.split
    result: Dict[str, List[Tuple[str, int]]] = {}
    for cid in sorted(df[category_column].dropna().astype(int).unique()):
        sub = df[df[category_column].astype(int) == cid]
        if sub.empty:
            continue
        vec = CountVectorizer(ngram_range=(n, n), tokenizer=tokenizer_fn, lowercase=False)
        mat = vec.fit_transform(sub[text_column].fillna("").astype(str))
        freqs = np.asarray(mat.sum(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        k = min(top_n, len(freqs))
        top_idx = np.argpartition(freqs, -k)[-k:]
        top_idx = top_idx[np.argsort(freqs[top_idx])[::-1]]
        category_name = LABELS.get(cid, str(cid))
        result[category_name] = [(str(terms[i]), int(freqs[i])) for i in top_idx]
    return result


def count_ngrams_by_category(
    df: pd.DataFrame,
    n: int = 2,
    category_column: str = "label",
    text_column: str = "text",
    top_n: Optional[int] = None,
) -> Dict[str, List[Tuple[str, int]]]:
    """Menghitung frekuensi kemunculan bigram atau trigram di setiap kategori berita."""
    freq_dict: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        category = row[category_column]
        words = preprocess_ngram_tokenizer(row[text_column])
        for gram in nltk.ngrams(words, n):
            freq_dict[category][" ".join(gram)] += 1

    result: Dict[str, List[Tuple[str, int]]] = {}
    for category_id, counts in freq_dict.items():
        category_name = LABELS.get(category_id, str(category_id))
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            sorted_counts = sorted_counts[:top_n]
        result[category_name] = sorted_counts
    return result
