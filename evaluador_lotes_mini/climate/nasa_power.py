"""NASA POWER monthly climatology and T × P quadrant classification."""

from __future__ import annotations

import calendar
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any

import requests
from shapely.geometry.base import BaseGeometry

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
QUADRANTS = ("Cálido-Húmedo", "Cálido-Seco", "Frío-Húmedo", "Frío-Seco")


@dataclass(frozen=True, slots=True)
class MonthlyClimate:
    year: int
    month: int
    precipitation_mm: float | None
    temperature_c: float | None
    evapotranspiration_mm: float | None


def fetch_monthly_climate(
    geometry: BaseGeometry,
    start_year: int = 1985,
    end_year: int = 2025,
    timeout_seconds: int = 90,
) -> list[MonthlyClimate]:
    centroid = geometry.centroid
    response = requests.get(
        POWER_URL,
        params={
            "parameters": "PRECTOTCORR,T2M,EVPTRNS",
            "community": "AG",
            "longitude": round(centroid.x, 5),
            "latitude": round(centroid.y, 5),
            "start": start_year,
            "end": end_year,
            "format": "JSON",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    parameters = response.json()["properties"]["parameter"]
    precip = parameters.get("PRECTOTCORR", {})
    temp = parameters.get("T2M", {})
    etp = parameters.get("EVPTRNS", {})
    rows: list[MonthlyClimate] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            key = f"{year}{month:02d}"
            rows.append(
                MonthlyClimate(
                    year=year,
                    month=month,
                    precipitation_mm=_valid(precip.get(key)),
                    temperature_c=_valid(temp.get(key)),
                    evapotranspiration_mm=_valid(etp.get(key)),
                )
            )
    return rows


def analyze_climate(
    rows: list[MonthlyClimate], start_year: int = 1985, end_year: int = 2025
) -> dict[str, Any]:
    monthly_profile = []
    for month in range(1, 13):
        values = [row for row in rows if row.month == month]
        monthly_profile.append(
            {
                "month": month,
                "precipitation_mm": _mean(row.precipitation_mm for row in values),
                "temperature_c": _mean(row.temperature_c for row in values),
                "evapotranspiration_mm": _mean(row.evapotranspiration_mm for row in values),
            }
        )

    annual: dict[int, float] = defaultdict(float)
    annual_count: dict[int, int] = defaultdict(int)
    for row in rows:
        if row.precipitation_mm is not None:
            annual[row.year] += row.precipitation_mm
            annual_count[row.year] += 1
    annual_rainfall = [
        {"year": year, "precipitation_mm": round(annual[year], 1)}
        for year in range(start_year, end_year + 1)
        if annual_count[year] == 12
    ]

    campaigns = {
        "gruesa": _classify_campaign(rows, [10, 11, 12, 1, 2, 3], True, start_year, end_year),
        "fina": _classify_campaign(rows, [4, 5, 6, 7, 8, 9], False, start_year, end_year),
    }
    return {
        "monthly_profile": monthly_profile,
        "annual_rainfall": annual_rainfall,
        "campaigns": campaigns,
        "source": "NASA POWER monthly point API",
        "methodology": {
            "spatial_sample": "Punto en el centroide del lote",
            "normal_period": "1991-2020",
            "gruesa_months": "octubre-marzo; el año identifica el octubre inicial",
            "fina_months": "abril-septiembre del mismo año",
            "precipitation_aggregation": "suma de 6 meses completos",
            "temperature_aggregation": "media ponderada por días de los 6 meses",
            "quadrants": "cortes en precipitación y temperatura medias del período normal",
            "representative_year": (
                "año Sentinel reciente más cercano al centro T×P de cada cuadrante"
            ),
            "evapotranspiration_role": (
                "se exporta y se calcula el balance P-ETP, pero no define los cuadrantes T×P"
            ),
        },
    }


def representative_years(climate: dict[str, Any], minimum_satellite_year: int = 2017) -> dict:
    result: dict[str, dict[str, int]] = {"gruesa": {}, "fina": {}}
    for campaign_name, campaign in climate["campaigns"].items():
        all_years = campaign["years"]
        if not all_years:
            continue
        temperature_scale = max(pstdev(row["temperature_c"] for row in all_years), 0.1)
        precipitation_scale = max(pstdev(row["precipitation_mm"] for row in all_years), 1.0)
        for quadrant, summary in campaign["quadrant_summary"].items():
            members = [row for row in all_years if row["quadrant"] == quadrant]
            modern = [row for row in members if row["year"] >= minimum_satellite_year]
            pool = modern or [row for row in members if row["year"] >= 2015]
            if pool:
                target_t = summary["average_temperature_c"]
                target_p = summary["average_precipitation_mm"]
                selected = min(
                    pool,
                    key=lambda row: (
                        ((row["temperature_c"] - target_t) / temperature_scale) ** 2
                        + ((row["precipitation_mm"] - target_p) / precipitation_scale) ** 2,
                        -row["year"],
                    ),
                )
                result[campaign_name][quadrant] = int(selected["year"])
    return result


def serialize_monthly(rows: list[MonthlyClimate]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def _classify_campaign(
    rows: list[MonthlyClimate],
    months: list[int],
    crosses_year: bool,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    values: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.month not in months or row.precipitation_mm is None or row.temperature_c is None:
            continue
        campaign_year = row.year
        if crosses_year and row.month <= 6:
            campaign_year -= 1
        if campaign_year < start_year or campaign_year > end_year:
            continue
        item = values.setdefault(
            campaign_year,
            {
                "precip": 0.0,
                "temperature_day_sum": 0.0,
                "temperature_days": 0,
                "etp": 0.0,
                "etp_count": 0,
                "months": set(),
            },
        )
        days = calendar.monthrange(row.year, row.month)[1]
        item["precip"] += row.precipitation_mm
        item["temperature_day_sum"] += row.temperature_c * days
        item["temperature_days"] += days
        item["months"].add((row.year, row.month))
        if row.evapotranspiration_mm is not None:
            item["etp"] += row.evapotranspiration_mm
            item["etp_count"] += 1

    years = []
    for year, value in sorted(values.items()):
        if len(value["months"]) != len(months):
            continue
        campaign_etp = value["etp"] if value["etp_count"] == len(months) else None
        years.append(
            {
                "year": year,
                "precipitation_mm": value["precip"],
                "temperature_c": (
                    value["temperature_day_sum"] / value["temperature_days"]
                ),
                "evapotranspiration_mm": campaign_etp,
                "water_balance_mm": (
                    value["precip"] - campaign_etp if campaign_etp is not None else None
                ),
            }
        )
    if not years:
        return {
            "normal_precipitation_mm": None,
            "normal_temperature_c": None,
            "years": [],
            "quadrant_summary": {
                quadrant: {
                    "count": 0,
                    "years": [],
                    "average_precipitation_mm": None,
                    "average_temperature_c": None,
                }
                for quadrant in QUADRANTS
            },
        }
    normal_pool = [item for item in years if 1991 <= item["year"] <= 2020]
    if len(normal_pool) < 20:
        normal_pool = years
    normal_p = fmean(item["precipitation_mm"] for item in normal_pool)
    normal_t = fmean(item["temperature_c"] for item in normal_pool)

    for item in years:
        warm = item["temperature_c"] >= normal_t
        wet = item["precipitation_mm"] >= normal_p
        item["quadrant"] = (
            "Cálido-Húmedo"
            if warm and wet
            else "Cálido-Seco"
            if warm
            else "Frío-Húmedo"
            if wet
            else "Frío-Seco"
        )
        item["precipitation_mm"] = round(item["precipitation_mm"])
        item["temperature_c"] = round(item["temperature_c"], 1)
        if item["evapotranspiration_mm"] is not None:
            item["evapotranspiration_mm"] = round(item["evapotranspiration_mm"])
            item["water_balance_mm"] = round(item["water_balance_mm"])

    summary: dict[str, dict[str, Any]] = {}
    for quadrant in QUADRANTS:
        members = [item for item in years if item["quadrant"] == quadrant]
        summary[quadrant] = {
            "count": len(members),
            "years": [item["year"] for item in members],
            "average_precipitation_mm": (
                round(fmean(item["precipitation_mm"] for item in members)) if members else None
            ),
            "average_temperature_c": (
                round(fmean(item["temperature_c"] for item in members), 1) if members else None
            ),
        }
    return {
        "normal_precipitation_mm": round(normal_p),
        "normal_temperature_c": round(normal_t, 1),
        "years": years,
        "quadrant_summary": summary,
    }


def _valid(value: Any) -> float | None:
    if value in (None, -999, -999.0):
        return None
    return float(value)


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(fmean(clean), 2) if clean else None
