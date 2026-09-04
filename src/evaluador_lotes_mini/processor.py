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

import numpy as np
import rasterio
from PIL import Image

from evaluador_lotes_mini.analysis.stability import CampaignType, calculate_stability
from evaluador_lotes_mini.analysis.zoning import (
    build_zone_alternatives,
    recommended_zone_count,
)
from evaluador_lotes_mini.climate.nasa_power import (
    analyze_climate,
    fetch_monthly_climate,
    ranked_representative_years,
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
from evaluador_lotes_mini.ui_preview import render_raster

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
        representative_candidates = ranked_representative_years(climate)
        climate_dir = output_dir / "clima"
        _write_json(climate_dir / "clima_mensual.json", serialize_monthly(monthly))
        _write_json(climate_dir / "analisis_climatico.json", climate)
        _write_json(climate_dir / "anos_representativos.json", representatives)
        _write_json(
            climate_dir / "anos_representativos_candidatos.json",
            representative_candidates,
        )
        write_rows_csv(climate_dir / "clima_mensual.csv", serialize_monthly(monthly))
        chart_artifacts = write_climate_charts(climate_dir, climate)
        result.artifacts.extend(
            [
                climate_dir / "clima_mensual.csv",
                climate_dir / "analisis_climatico.json",
                climate_dir / "anos_representativos.json",
                climate_dir / "anos_representativos_candidatos.json",
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
                (campaign, quadrant, candidate_years[:3])
                for campaign, quadrants in representative_candidates.items()
                for quadrant, candidate_years in quadrants.items()
                if candidate_years
            ]
            for position, (campaign, quadrant, candidate_years) in enumerate(
                selections, start=1
            ):
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
                    artifacts, metadata = _export_first_valid_representative(
                        imagery,
                        lot,
                        options,
                        campaign,
                        quadrant,
                        candidate_years,
                        scene_dir,
                    )
                    result.artifacts.extend(artifacts)
                    scene_manifest.append(metadata)
                except RuntimeError as exc:
                    result.warnings.append(f"Imagen {campaign}/{quadrant}: {exc}")
        _write_json(output_dir / "imagenes_cuadrantes" / "manifest.json", scene_manifest)

        if options.calculate_productivity:
            for index, campaign_type in enumerate(("gruesa", "fina")):
                result.artifacts.extend(
                    _process_productivity_campaign(
                        lot,
                        output_dir,
                        imagery,
                        options,
                        campaign_type,
                        progress=progress,
                        progress_start=0.50 + index * 0.20,
                        progress_end=0.70 + index * 0.20,
                    )
                )

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


def _export_first_valid_representative(
    imagery: PlanetaryImagery,
    lot: Lot,
    options: ProcessingOptions,
    campaign: str,
    quadrant: str,
    candidate_years: list[int],
    scene_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Try ranked years in order and export the first scene that passes unchanged QA."""
    attempt_errors: list[str] = []
    for rank, year in enumerate(candidate_years, start=1):
        try:
            start, end = campaign_dates(campaign, int(year))
            item, selection = imagery.select_peak_scene(
                lot.geometry,
                start,
                end,
                max_cloud_percent=options.max_cloud_percent,
                buffer_m=options.buffer_m,
            )
            artifacts, metadata = imagery.export_scene_products(
                item,
                lot.geometry,
                scene_dir,
                buffer_m=options.buffer_m,
                file_prefix=selection.acquisition_date,
                file_suffix=f"{campaign}_{_qgis_quadrant_token(quadrant)}",
            )
            metadata.update(
                {
                    "campaign": campaign,
                    "climate_quadrant": quadrant,
                    "representative_year": year,
                    "primary_representative_year": candidate_years[0],
                    "representative_rank": rank,
                    "representative_candidates_tried": candidate_years[:rank],
                    "selection": asdict(selection),
                }
            )
            _write_json(scene_dir / "metadata.json", metadata)
            return artifacts, metadata
        except Exception as exc:
            attempt_errors.append(f"{year}: {exc}")

    attempted = ", ".join(str(year) for year in candidate_years)
    last_error = attempt_errors[-1] if attempt_errors else "sin candidatos"
    raise RuntimeError(
        f"no hubo escena válida en los años representativos probados ({attempted}). "
        f"Último intento: {last_error}"
    )


def recalculate_productivity_campaign(
    lot: Lot,
    output_dir: Path,
    options: ProcessingOptions,
    campaign_type: CampaignType,
    excluded_item_ids: set[str],
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Rebuild one productivity branch from cached scenes after human review."""
    started = monotonic()
    started_at = datetime.now(UTC)
    revision_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    before = _productivity_summary(output_dir, campaign_type)
    previous_classes = _read_classes(output_dir, campaign_type)
    before_previews = _snapshot_productivity_previews(
        output_dir, campaign_type, revision_id, "antes"
    )
    imagery = PlanetaryImagery()
    _process_productivity_campaign(
        lot,
        output_dir,
        imagery,
        options,
        campaign_type,
        excluded_item_ids=excluded_item_ids,
        progress=progress,
        progress_start=0.0,
        progress_end=1.0,
    )
    after = _productivity_summary(output_dir, campaign_type)
    current_classes = _read_classes(output_dir, campaign_type)
    after_previews = _snapshot_productivity_previews(
        output_dir, campaign_type, revision_id, "despues"
    )
    completed_at = datetime.now(UTC)
    entry = {
        "revision_id": revision_id,
        "status": "completed",
        "campaign_type": campaign_type,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(monotonic() - started, 1),
        "excluded_item_ids": sorted(excluded_item_ids),
        "before": before,
        "after": after,
        "changed_class_pixels_percent": _changed_class_percent(
            previous_classes, current_classes
        ),
        "previews": {"before": before_previews, "after": after_previews},
    }
    history_file = output_dir / "estabilidad" / campaign_type / "historial_previews.json"
    history = _read_json_list(history_file)
    history.append(entry)
    _write_json(history_file, history)
    _write_json(
        output_dir / "estabilidad" / campaign_type / "revision_humana.json",
        {**entry, "history_count": len(history)},
    )
    return _zip_lot(output_dir)


def _process_productivity_campaign(
    lot: Lot,
    output_dir: Path,
    imagery: PlanetaryImagery,
    options: ProcessingOptions,
    campaign_type: CampaignType,
    *,
    excluded_item_ids: set[str] | None = None,
    progress: ProgressCallback | None = None,
    progress_start: float,
    progress_end: float,
) -> list[Path]:
    span = progress_end - progress_start
    stability_dir = output_dir / "estabilidad" / campaign_type
    environments_dir = output_dir / "ambientes" / campaign_type
    _notify(progress, f"{lot.name}: estabilidad de {campaign_type}", progress_start)
    stability = calculate_stability(
        imagery,
        lot.geometry,
        stability_dir,
        campaign_type=campaign_type,
        seasons=options.stability_seasons,
        max_cloud_percent=options.max_cloud_percent,
        edge_exclusion_m=options.edge_exclusion_m,
        cache_arrays=options.cache_review_arrays,
        excluded_item_ids=excluded_item_ids,
        progress=lambda completed, total, campaign: _notify(
            progress,
            f"{lot.name}: {campaign_type} {completed}/{total} ({campaign})",
            progress_start + span * 0.75 * completed / total,
        ),
    )
    _write_json(stability_dir / "campanas_utilizadas.json", stability.seasons)
    write_stability_csv(stability_dir / "estadisticas.csv", stability)
    _notify(progress, f"{lot.name}: ambientes de {campaign_type}", progress_start + span * 0.8)
    alternatives = build_zone_alternatives(stability, environments_dir, options.zone_counts)
    geopackage = write_geopackage(
        output_dir / "gis" / f"{safe_slug(lot.name)}_{campaign_type}.gpkg",
        lot,
        stability,
        alternatives,
        campaign_type,
    )
    recommendation = recommended_zone_count(alternatives)
    metadata_path = _write_json(
        environments_dir / "alternativas.json",
        {
            "campaign_type": campaign_type,
            "method": "KMeans",
            "membership": "hard",
            "features": [
                {"name": "productividad_media_relativa", "weight": 1.0},
                {"name": "variabilidad_temporal", "weight": 1.0},
            ],
            "preprocessing": "StandardScaler",
            "minimum_zone_share_percent": 10,
            "minimum_silhouette": 0.25,
            "edge_training_exclusion_m": stability.edge_exclusion_m,
            "excluded_item_ids": sorted(excluded_item_ids or set()),
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
    return [
        *stability.artifacts,
        stability_dir / "campanas_utilizadas.json",
        stability_dir / "estadisticas.csv",
        geopackage,
        metadata_path,
        *(item.raster_path for item in alternatives),
    ]


def _copy_qgis_assets(output_dir: Path) -> None:
    assets = Path(__file__).parent / "assets" / "qgis"
    if assets.exists():
        shutil.copytree(
            assets,
            output_dir / "qgis",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _productivity_summary(output_dir: Path, campaign_type: CampaignType) -> dict[str, Any]:
    stability_dir = output_dir / "estabilidad" / campaign_type
    seasons_file = stability_dir / "campanas_utilizadas.json"
    scenes_file = stability_dir / "escenas_utilizadas.json"
    seasons = json.loads(seasons_file.read_text(encoding="utf-8")) if seasons_file.exists() else []
    scenes = json.loads(scenes_file.read_text(encoding="utf-8")) if scenes_file.exists() else []
    return {
        "campaign_count": len(seasons),
        "campaigns": [row.get("campaign") for row in seasons],
        "included_scene_count": sum(bool(row.get("included")) for row in scenes),
        "inventory_scene_count": len(scenes),
    }


def _read_classes(output_dir: Path, campaign_type: CampaignType) -> np.ndarray | None:
    path = output_dir / "estabilidad" / campaign_type / "estabilidad_5_clases.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as source:
        return source.read(1)


def _changed_class_percent(before: np.ndarray | None, after: np.ndarray | None) -> float | None:
    if before is None or after is None or before.shape != after.shape:
        return None
    comparable = (before > 0) & (after > 0)
    total = int(np.count_nonzero(comparable))
    if not total:
        return None
    return round(float(np.count_nonzero(before[comparable] != after[comparable]) / total * 100), 1)


def _snapshot_productivity_previews(
    output_dir: Path,
    campaign_type: CampaignType,
    revision_id: str,
    stage: str,
) -> list[dict[str, str]]:
    stability_dir = output_dir / "estabilidad" / campaign_type
    environment_dir = output_dir / "ambientes" / campaign_type
    boundary = output_dir / "entrada" / "lote.geojson"
    rasters = [
        stability_dir / "estabilidad_5_clases.tif",
        *sorted(environment_dir.glob("ambientes_k*.tif")),
    ]
    target_dir = stability_dir / "historial_previews" / revision_id / stage
    rows: list[dict[str, str]] = []
    for raster in rasters:
        if not raster.exists():
            continue
        target = target_dir / f"{raster.stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(render_raster(raster, boundary)).save(target, format="PNG", optimize=True)
        rows.append(
            {
                "layer": raster.stem,
                "path": str(target.relative_to(output_dir)),
            }
        )
    return rows


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


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
            if (
                path.is_file()
                and path.name != ".completed.json"
                and ".cache_escenas" not in path.parts
                and not (path.parent.name == "qgis" and path.name == "LEEME.txt")
            ):
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


def _qgis_quadrant_token(value: str) -> str:
    return safe_slug(value).replace("-", "_").capitalize()
