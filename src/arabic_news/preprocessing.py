"""Utilitas pra-pemrosesan teks berita berbahasa Arab untuk skema Medan Makna dan Semantik Frame."""

import re
from pathlib import Path
from string import punctuation as ASCII_PUNCTUATION
from typing import Iterable, List, Optional, Set, Union

import arabic_reshaper
from bidi.algorithm import get_display

from .config import resolve_data_file

ARABIC_PUNCTUATION: str = "،؛؟”0123456789“" + ASCII_PUNCTUATION


def load_stopwords(
    stopwords_path: Optional[Union[str, Path]] = None,
    include_library_stopwords: bool = True,
) -> List[str]:
    """Memuat daftar kata henti Arab gabungan dari data/stopwords.txt dan pustaka Arabic-Stopwords."""
    path = Path(stopwords_path) if stopwords_path else resolve_data_file("stopwords.txt")
    stopwords: List[str] = []
    if path.exists():
        stopwords = path.read_text(encoding="utf-8").splitlines()

    if include_library_stopwords:
        try:
            import arabicstopwords.arabicstopwords as stp

            stopwords = list(stp.stopwords_list()) + stopwords
        except ImportError:
            pass

    return list(dict.fromkeys(stopwords))


def preprocess_text(
    text: str,
    stopwords: Optional[Union[Iterable[str], Set[str]]] = None,
) -> str:
    """Membersihkan teks (lowercase, angka, tanda baca Arab/ASCII) serta opsional menghapus stopwords."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(ARABIC_PUNCTUATION)}]", "", text)
    words = text.split()

    if stopwords is not None:
        stop_set = stopwords if isinstance(stopwords, (set, frozenset)) else set(stopwords)
        words = [word for word in words if word not in stop_set]

    return " ".join(words)


def preprocess_ngram_tokenizer(text: str) -> List[str]:
    """Tokenizer berbasis ekspresi reguler untuk pembentukan kamus frasa bigram/trigram."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    cleaned = re.sub(r"\W+", " ", text)
    return cleaned.lower().split()


def reshape_arabic_for_plot(text: str) -> str:
    """Membentuk ulang teks Arab RTL agar tampil benar pada grafik Matplotlib dan WordCloud."""
    return get_display(arabic_reshaper.reshape(text))
