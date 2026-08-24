"""Static, portable climate charts for the per-lot report."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "elm-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "Cálido-Húmedo": "#2b8cbe",
    "Cálido-Seco": "#d95f0e",
    "Frío-Húmedo": "#41ab5d",
    "Frío-Seco": "#756bb1",
}


def write_climate_charts(output_dir: Path, climate: dict[str, Any]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [_annual_rainfall(output_dir / "precipitacion_anual.png", climate)]
    for campaign, values in climate["campaigns"].items():
        artifacts.append(_quadrants(output_dir / f"cuadrantes_{campaign}.png", campaign, values))
    return artifacts


def _annual_rainfall(path: Path, climate: dict[str, Any]) -> Path:
    rows = climate["annual_rainfall"]
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.plot(
        [row["year"] for row in rows],
        [row["precipitation_mm"] for row in rows],
        color="#28784a",
        linewidth=1.5,
    )
    axis.set(title="Precipitación anual", xlabel="Año", ylabel="mm")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _quadrants(path: Path, campaign: str, values: dict[str, Any]) -> Path:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for quadrant, color in COLORS.items():
        members = [row for row in values["years"] if row["quadrant"] == quadrant]
        axis.scatter(
            [row["temperature_c"] for row in members],
            [row["precipitation_mm"] for row in members],
            s=30,
            alpha=0.75,
            label=quadrant,
            color=color,
        )
    axis.axvline(values["normal_temperature_c"], color="#555", linewidth=1)
    axis.axhline(values["normal_precipitation_mm"], color="#555", linewidth=1)
    axis.set(
        title=f"Cuadrantes climáticos · {campaign.capitalize()}",
        xlabel="Temperatura media (°C)",
        ylabel="Precipitación acumulada (mm)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
