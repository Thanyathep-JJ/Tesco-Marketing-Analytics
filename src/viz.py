"""Shared plotting style and figure saving."""
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIG_DIR, SEGMENT_COLORS, SEGMENT_ORDER

PRIMARY = "#5b8fd6"
ACCENT = "#2a9d8f"
MUTED = "#c9c9c9"


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        # Tahoma renders Thai on Windows; DejaVu Sans is the fallback elsewhere.
        "font.family": ["Tahoma", "DejaVu Sans"],
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })


def save(fig, name: str):
    """Save a figure to reports/figures/<name>.png for use in slides."""
    fig.savefig(FIG_DIR / f"{name}.png")


def seg_palette():
    return [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]
