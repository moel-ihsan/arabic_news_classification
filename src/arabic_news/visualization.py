"""Utilitas visualisasi grafik dan WordCloud berbahasa Arab dengan path relatif lintas-platform."""

from pathlib import Path
from typing import List, Optional

from .config import RAW_DATA_DIR
from .preprocessing import reshape_arabic_for_plot


def find_arabic_font_path(custom_font_path: Optional[str] = None) -> Optional[str]:
    """Mencari berkas font pendukung huruf Arab: prioritas font relatif repo, font Arab Linux/macOS, lalu fallback."""
    candidates: List[Path] = []
    if custom_font_path:
        candidates.append(Path(custom_font_path))

    # Font opsional di dalam direktori relatif proyek
    fonts_dir = RAW_DATA_DIR / "fonts"
    if fonts_dir.exists():
        candidates.extend(sorted(fonts_dir.glob("*.ttf")))
        candidates.extend(sorted(fonts_dir.glob("*.otf")))

    # Prioritaskan font Arab lengkap (mendukung huruf Latin & Arabic Presentation Forms) di Linux/macOS
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/hosny-amiri/Amiri-Regular.ttf"),
            Path("/usr/share/fonts/opentype/hosny-amiri/Amiri-Regular.ttf"),
            Path("/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Tahoma.ttf"),
            Path("/System/Library/Fonts/GeezaPro.ttc"),
        ]
    )

    linux_fonts = Path("/usr/share/fonts")
    if linux_fonts.exists():
        for pattern in ("*Amiri-Regular*", "*Amiri*", "*NotoNaskhArabic*", "*NotoSansArabic*"):
            candidates.extend(sorted(linux_fonts.rglob(f"{pattern}.ttf")))
            candidates.extend(sorted(linux_fonts.rglob(f"{pattern}.otf")))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Pencarian dinamis via matplotlib fontManager (hanya jika kandidat langsung belum ditemukan)
    try:
        import matplotlib.font_manager as fm

        for font_name in ("Amiri", "Arial", "Tahoma", "Noto Naskh Arabic", "Noto Sans Arabic", "DejaVu Sans"):
            matched = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
            if matched and Path(matched).exists():
                return str(matched)
    except Exception:
        pass

    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return str(dejavu) if dejavu.exists() else None


__all__ = ["find_arabic_font_path", "reshape_arabic_for_plot"]
