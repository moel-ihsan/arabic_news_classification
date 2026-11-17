"""Utilitas visualisasi grafik dan WordCloud berbahasa Arab dengan path relatif lintas-platform."""

from pathlib import Path
from typing import Optional

from .config import RAW_DATA_DIR
from .preprocessing import reshape_arabic_for_plot


def find_arabic_font_path(custom_font_path: Optional[str] = None) -> Optional[str]:
    """Mencari berkas font pendukung huruf Arab: prioritas font relatif repo lalu font sistem OS."""
    candidates = []
    if custom_font_path:
        candidates.append(Path(custom_font_path))

    # Font opsional di dalam direktori relatif proyek
    fonts_dir = RAW_DATA_DIR / "fonts"
    if fonts_dir.exists():
        candidates.extend(sorted(fonts_dir.glob("*.ttf")))
        candidates.extend(sorted(fonts_dir.glob("*.otf")))

    # Pencarian dinamis via matplotlib fontManager & lokasi standar OS
    try:
        import matplotlib.font_manager as fm

        for font_name in ("Arial", "Tahoma", "Noto Naskh Arabic", "DejaVu Sans"):
            matched = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
            if matched and Path(matched).exists():
                candidates.append(Path(matched))
    except Exception:
        pass

    candidates.extend(
        [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Tahoma.ttf"),
            Path("/System/Library/Fonts/GeezaPro.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


__all__ = ["find_arabic_font_path", "reshape_arabic_for_plot"]
