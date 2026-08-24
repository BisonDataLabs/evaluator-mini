"""Resumable end-to-end processing for one lot or a bulk lot collection."""

from __future__ import annotations

import json
import shutil
import traceback
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import monotonic
from typing import Any

from evaluador_lotes_mini.analysis.stability import calculate_stability
from evaluador_lotes_mini.analysis.zoning import (
    build_zone_alternatives,
    recommended_zone_count,
)
from evaluador_lotes_mini.climate.nasa_power import (
    analyze_climate,
    fetch_monthly_climate,
    representative_years,
    serialize_monthly,
)
from evaluador_lotes_mini.config import Settings
from evaluador_lotes_mini.geometry import (
    area_hectares,
    feature_collection,
    geometry_hash,
    safe_slug,
)
from evaluador_lotes_mini.imagery.planetary import PlanetaryImagery, campaign_dates
from evaluador_lotes_mini.models import Lot, LotResult, ProcessingOptions
from evaluador_lotes_mini.outputs.charts import write_climate_charts
from evaluador_lotes_mini.outputs.gis import (
    write_geopackage,
    write_rows_csv,
    write_stability_csv,
)

ProgressCallback = Callable[[str, float], None]


def process_batch(
    lots: Iterable[Lot],
    settings: Settings,
    options: ProcessingOptions,
    *,
    batch_name: str | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[Path, list[LotResult]]:
    lot_list = list(lots)
    if not lot_list:
        raise ValueError("No hay lotes para procesar")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = settings.output_dir / (safe_slug(batch_name) if batch_name else f"corrida-{stamp}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    results: list[LotResult] = []
    for index, lot in enumerate(lot_list, start=1):
        prefix = f"[{index}/{len(lot_list)}] {lot.name}"
        _notify(progress, f"{prefix}: iniciando", (index - 1) / len(lot_list))

        def lot_progress(message: str, fraction: float, current_index: int = index) -> None:
            batch_fraction = ((current_index - 1) + fraction) / len(lot_list)
            _notify(progress, message, batch_fraction)

        result = process_lot(lot, batch_dir, options, resume=resume, progress=lot_progress)
        results.append(result)
        _write_json(
            batch_dir / "manifest.json",
            {
                "created_at": datetime.now(UTC).isoformat(),
                "options": asdict(options),
                "lots": [item.manifest_dict() for item in results],
            },
        )
        _notify(progress, f"{prefix}: {result.status}", index / len(lot_list))
    _zip_batch_index(batch_dir, results)
    return batch_dir, results


def process_lot(
    lot: Lot,
    batch_dir: Path,
    options: ProcessingOptions,
    *,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> LotResult:
    identity = f"{safe_slug(lot.name)}-{geometry_hash(lot.geometry)[:8]}"
    output_dir = batch_dir / identity
    done_file = output_dir / ".completed.json"
    if resume and done_file.exists():
        payload = json.loads(done_file.read_text(encoding="utf-8"))
        artifacts = [Path(value) for value in payload.get("artifacts", [])]
        if artifacts and all(path.exists() for path in artifacts):
            return LotResult(lot.lot_id, lot.name, output_dir, "skipped", artifacts=artifacts)

    result = LotResult(lot.lot_id, lot.name, output_dir, "running")
    started = monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _notify(progress, f"{lot.name}: clima NASA POWER", 0.05)
        boundary = output_dir / "entrada" / "lote.geojson"
        _write_json(boundary, feature_collection([(lot.geometry, lot.to_feature()["properties"])]))
        result.artifacts.append(boundary)

        monthly = fetch_monthly_climate(lot.geometry, options.start_year, options.end_year)
        climate = analyze_climate(monthly, options.start_year, options.end_year)
        representatives = representative_years(climate)
        climate_dir = output_dir / "clima"
        _write_json(climate_dir / "clima_mensual.json", serialize_monthly(monthly))
        _write_json(climate_dir / "analisis_climatico.json", climate)
        _write_json(climate_dir / "anos_representativos.json", representatives)
        write_rows_csv(climate_dir / "clima_mensual.csv", serialize_monthly(monthly))
        chart_artifacts = write_climate_charts(climate_dir, climate)
        result.artifacts.extend(
            [
                climate_dir / "clima_mensual.csv",
                climate_dir / "analisis_climatico.json",
                climate_dir / "anos_representativos.json",
                *chart_artifacts,
            ]
        )

        imagery = PlanetaryImagery()
        scene_manifest: list[dict[str, Any]] = []
        if options.export_quadrant_imagery:
            expected_quadrants = {
                "Cálido-Húmedo",
                "Cálido-Seco",
                "Frío-Húmedo",
                "Frío-Seco",
            }
            for campaign, quadrants in representatives.items():
                for missing in sorted(expected_quadrants - set(quadrants)):
                    result.warnings.append(
                        f"Sin año Sentinel-2 representativo para {campaign}/{missing}"
                    )
            selections = [
                (campaign, quadrant, year)
                for campaign, quadrants in representatives.items()
                for quadrant, year in quadrants.items()
            ]
            for position, (campaign, quadrant, year) in enumerate(selections, start=1):
                _notify(
                    progress,
                    f"{lot.name}: imagen {position}/{len(selections)} ({campaign}, {quadrant})",
                    0.1 + 0.35 * position / max(1, len(selections)),
                )
                scene_dir = output_dir / "imagenes_cuadrantes" / campaign / safe_slug(quadrant)
                metadata_path = scene_dir / "metadata.json"
                if resume and metadata_path.exists():
                    scene_manifest.append(json.loads(metadata_path.read_text(encoding="utf-8")))
                    continue
                try:
                    start, end = campaign_dates(campaign, int(year))
                    item, selection = imagery.select_peak_scene(
                        lot.geometry,
                        start,
                        end,
                        max_cloud_percent=options.max_cloud_percent,
                    )
                    artifacts, metadata = imagery.export_scene_products(
                        item, lot.geometry, scene_dir, buffer_m=options.buffer_m
                    )
                    metadata.update(
                        {
                            "campaign": campaign,
                            "climate_quadrant": quadrant,
                            "representative_year": year,
                            "selection": asdict(selection),
                        }
                    )
                    _write_json(metadata_path, metadata)
                    result.artifacts.extend(artifacts)
                    scene_manifest.append(metadata)
                except Exception as exc:
                    result.warnings.append(f"Imagen {campaign}/{quadrant}: {exc}")
        _write_json(output_dir / "imagenes_cuadrantes" / "manifest.json", scene_manifest)

        if options.calculate_productivity:
            _notify(progress, f"{lot.name}: estabilidad multitemporal", 0.52)
            stability = calculate_stability(
                imagery,
                lot.geometry,
                output_dir / "estabilidad",
                seasons=options.stability_seasons,
                max_cloud_percent=options.max_cloud_percent,
                progress=lambda completed, total, campaign: _notify(
                    progress,
                    f"{lot.name}: estabilidad {completed}/{total} ({campaign})",
                    0.52 + 0.28 * completed / total,
                ),
            )
            result.artifacts.extend(stability.artifacts)
            _write_json(output_dir / "estabilidad" / "campanas_utilizadas.json", stability.seasons)
            write_stability_csv(output_dir / "estabilidad" / "estadisticas.csv", stability)

            _notify(progress, f"{lot.name}: ambientes productivos", 0.82)
            alternatives = build_zone_alternatives(
                stability, output_dir / "ambientes", options.zone_counts
            )
            geopackage = write_geopackage(
                output_dir / "gis" / f"{safe_slug(lot.name)}.gpkg",
                lot,
                stability,
                alternatives,
            )
            recommendation = recommended_zone_count(alternatives)
            _write_json(
                output_dir / "ambientes" / "alternativas.json",
                {
                    "method": "KMeans",
                    "membership": "hard",
                    "features": [
                        {"name": "productividad_media_relativa", "weight": 1.0},
                        {"name": "variabilidad_temporal", "weight": 1.0},
                    ],
                    "preprocessing": "StandardScaler",
                    "minimum_zone_share_percent": 10,
                    "minimum_silhouette": 0.25,
                    "edge_training_exclusion_m": 20,
                    "recommended_k": recommendation,
                    "alternatives": [
                        {
                            "k": item.k,
                            "silhouette": round(item.silhouette, 4),
                            "zone_percentages": item.zone_percentages,
                            "passes_minimum_zone_share": min(item.zone_percentages.values()) >= 10,
                        }
                        for item in alternatives
                    ],
                },
            )
            result.artifacts.extend([geopackage, *(item.raster_path for item in alternatives)])

        _copy_qgis_assets(output_dir)
        _write_report(output_dir, lot, options, result, scene_manifest)
        result.status = "completed"
        package = _zip_lot(output_dir)
        result.artifacts.append(package)
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        (output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result.elapsed_seconds = round(monotonic() - started, 1)
    if result.status == "completed":
        _write_json(done_file, result.manifest_dict())
    return result


def _copy_qgis_assets(output_dir: Path) -> None:
    assets = Path(__file__).parent / "assets" / "qgis"
    if assets.exists():
        shutil.copytree(
            assets,
            output_dir / "qgis",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _write_report(
    output_dir: Path,
    lot: Lot,
    options: ProcessingOptions,
    result: LotResult,
    scenes: list[dict[str, Any]],
) -> None:
    lot_name = escape(lot.name)
    warnings = (
        "".join(f"<li>{escape(warning)}</li>" for warning in result.warnings) or "<li>Ninguna</li>"
    )
    rows = "".join(
        f"<tr><td>{escape(str(item.get('campaign', '')))}</td>"
        f"<td>{escape(str(item.get('climate_quadrant', '')))}</td>"
        f"<td>{escape(str(item.get('date', '')))}</td>"
        f"<td>{escape(str(item.get('cloud_percent', '')))}</td></tr>"
        for item in scenes
    )
    html = f"""<!doctype html><html lang='es'><meta charset='utf-8'>
<title>Evaluación {lot_name}</title><style>
body{{font:16px system-ui;max-width:980px;margin:40px auto;
padding:0 24px;color:#203027}}
h1,h2{{color:#185c37}} table{{border-collapse:collapse;width:100%}}
td,th{{padding:8px;border:1px solid #ccd8cf}}
.metric{{display:inline-block;padding:14px 22px;background:#edf6f0;
border-radius:8px;margin-right:12px}}
</style><body><h1>{lot_name}</h1>
<div class='metric'><b>{area_hectares(lot.geometry):.1f}</b><br>hectáreas</div>
<div class='metric'><b>{options.buffer_m} m</b><br>contexto satelital</div>
<h2>Clima</h2>
<img src='clima/precipitacion_anual.png' style='max-width:100%'>
<img src='clima/cuadrantes_gruesa.png' style='max-width:49%'>
<img src='clima/cuadrantes_fina.png' style='max-width:49%'>
<h2>Imágenes representativas</h2><table><tr><th>Campaña</th>
<th>Cuadrante</th><th>Fecha</th><th>Nubosidad %</th></tr>{rows}</table>
<h2>Advertencias</h2><ul>{warnings}</ul>
<p>Los archivos GIS están listos para QGIS. Abra el GeoPackage y agregue los
GeoTIFFs.</p></body></html>"""
    (output_dir / "reporte.html").write_text(html, encoding="utf-8")


def _zip_lot(output_dir: Path) -> Path:
    package = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(
        package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name != ".completed.json":
                archive.write(path, path.relative_to(output_dir.parent))
    return package


def _zip_batch_index(batch_dir: Path, results: list[LotResult]) -> Path:
    path = batch_dir / "resultados.csv"
    write_rows_csv(
        path,
        [
            {
                "lot_id": item.lot_id,
                "lot_name": item.lot_name,
                "status": item.status,
                "warnings": " | ".join(item.warnings),
                "error": item.error or "",
            }
            for item in results
        ],
    )
    return path


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _notify(callback: ProgressCallback | None, message: str, fraction: float) -> None:
    if callback:
        callback(message, max(0.0, min(1.0, fraction)))
