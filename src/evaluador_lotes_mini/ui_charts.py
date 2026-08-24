"""Interactive Plotly charts built from portable climate JSON outputs."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

QUADRANT_STYLES = {
    "Cálido-Húmedo": ("#0072B2", "circle"),
    "Cálido-Seco": ("#D55E00", "diamond"),
    "Frío-Húmedo": ("#009E73", "square"),
    "Frío-Seco": ("#CC79A7", "cross"),
}


def annual_rainfall_figure(climate: dict[str, Any]) -> go.Figure:
    rows = climate["annual_rainfall"]
    values = [row["precipitation_mm"] for row in rows]
    mean_value = sum(values) / len(values) if values else 0
    figure = go.Figure(
        go.Scatter(
            x=[row["year"] for row in rows],
            y=values,
            mode="lines+markers",
            name="Precipitación",
            line={"color": "#28784a", "width": 2},
            marker={"size": 5},
            hovertemplate="Año %{x}<br>%{y:.0f} mm<extra></extra>",
        )
    )
    figure.add_hline(
        y=mean_value,
        line_dash="dash",
        line_color="#555",
        annotation_text=f"Media {mean_value:.0f} mm",
    )
    figure.update_layout(
        title="Precipitación anual",
        xaxis_title="Año",
        yaxis_title="Precipitación (mm)",
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
    )
    return figure


def quadrant_figure(climate: dict[str, Any], campaign: str) -> go.Figure:
    values = climate["campaigns"][campaign]
    figure = go.Figure()
    for quadrant, (color, symbol) in QUADRANT_STYLES.items():
        members = [row for row in values["years"] if row["quadrant"] == quadrant]
        figure.add_trace(
            go.Scatter(
                x=[row["temperature_c"] for row in members],
                y=[row["precipitation_mm"] for row in members],
                customdata=[row["year"] for row in members],
                mode="markers",
                name=quadrant,
                marker={"color": color, "symbol": symbol, "size": 9, "opacity": 0.8},
                hovertemplate=(
                    "Año %{customdata}<br>Temperatura %{x:.1f} °C"
                    "<br>Precipitación %{y:.0f} mm<extra></extra>"
                ),
            )
        )
    figure.add_vline(
        x=values["normal_temperature_c"],
        line_color="#666",
        line_width=1,
        annotation_text="Temperatura normal",
    )
    figure.add_hline(
        y=values["normal_precipitation_mm"],
        line_color="#666",
        line_width=1,
        annotation_text="Precipitación normal",
    )
    figure.update_layout(
        title=f"Cuadrantes climáticos · {campaign.capitalize()}",
        xaxis_title="Temperatura media (°C)",
        yaxis_title="Precipitación acumulada (mm)",
        template="plotly_white",
        legend_title="Cuadrante",
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
    )
    return figure
