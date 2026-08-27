"""Streamlit interface for controlled selection, evaluation and visual review."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluador_lotes_mini.analysis.stability import STABILITY_LABELS
from evaluador_lotes_mini.config import load_settings
from evaluador_lotes_mini.geometry import area_hectares
from evaluador_lotes_mini.ingestion.files import make_lot_ids_unique, read_uploaded_file
from evaluador_lotes_mini.models import Lot, LotResult, ProcessingOptions
from evaluador_lotes_mini.processor import process_batch, recalculate_productivity_campaign
from evaluador_lotes_mini.ui_charts import annual_rainfall_figure, quadrant_figure
from evaluador_lotes_mini.ui_preview import STABILITY_COLORS, render_raster

if os.getenv("ELM_DEPLOYMENT_MODE", "local").lower() != "upload_only":
    from evaluador_lotes_mini.ingestion.snowflake import (
        connection_available,
        fetch_filter_groups,
        fetch_lots,
        search_lot_catalog,
    )

MAX_LOTS_PER_BATCH = 10
REFERENCE_SECONDS_PER_LOT = 14 * 60
SNOWFLAKE_ENABLED = os.getenv("ELM_DEPLOYMENT_MODE", "local").lower() != "upload_only"
st.set_page_config(page_title="Evaluador de lotes mini", page_icon="🌱", layout="wide")


def _load_streamlit_secrets_into_environment() -> None:
    """Bridge deploy-time Streamlit secrets into the framework-agnostic engine."""
    allowed_names = (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
        "SNOWFLAKE_PRIVATE_KEY_PEM",
        "SNOWFLAKE_PRIVATE_KEY_B64",
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "ELM_WORK_DIR",
        "ELM_CACHE_DIR",
        "ELM_OUTPUT_DIR",
        "ELM_MAX_WORKERS",
        "ELM_DEFAULT_BUFFER_M",
    )
    try:
        secrets = st.secrets
        for name in allowed_names:
            if name not in os.environ and name in secrets:
                os.environ[name] = str(secrets[name])
    except FileNotFoundError:
        pass


_load_streamlit_secrets_into_environment()
settings = load_settings()
st.title("Evaluador de lotes mini")
st.caption("Clima · cuadrantes T×P · Sentinel-2 · estabilidad · ambientes productivos")

for key, default in {
    "lots": [],
    "snowflake_groups": [],
    "snowflake_catalog": [],
    "last_results": [],
    "all_results": [],
    "completed_lot_ids": [],
    "last_elapsed": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if not st.session_state.all_results and st.session_state.last_results:
    st.session_state.all_results = st.session_state.last_results


@st.cache_data(show_spinner=False, max_entries=100)
def cached_preview(raster: str, raster_mtime: float, boundary: str, boundary_mtime: float):
    del raster_mtime, boundary_mtime
    return render_raster(Path(raster), Path(boundary))


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculando…"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


with st.sidebar:
    st.header("Configuración")
    buffer_m = st.number_input("Contexto alrededor del lote (m)", 0, 3000, 500, 100)
    max_cloud = st.slider("Nubosidad máxima (%)", 5, 80, 30, 5)
    seasons = st.slider("Campañas para estabilidad", 3, 12, 8)
    st.caption("Alternativas predeterminadas: 2, 3 y 4 ambientes.")
    with st.expander("› Elegir más"):
        zone_counts = st.multiselect(
            "Alternativas de ambientes",
            [2, 3, 4, 5, 6],
            [2, 3, 4],
        )
    quadrant_images = st.toggle("Imágenes por cuadrante climático", value=True)
    productivity = st.toggle("Estabilidad y ambientes", value=True)
    st.caption(
        "Estimación inicial con estabilidad separada de fina y gruesa: "
        "aproximadamente 12–18 minutos por lote."
    )

upload_tab, snowflake_tab = st.tabs(
    ["Subir archivos", "Snowflake" if SNOWFLAKE_ENABLED else "Snowflake · solo local"]
)
with upload_tab:
    files = st.file_uploader(
        "GeoJSON individuales, ZIP con muchos GeoJSON, KMZ, KML, Shapefile ZIP o GeoPackage",
        type=["kmz", "kml", "geojson", "json", "zip", "gpkg"],
        accept_multiple_files=True,
    )
    if st.button("Leer archivos", disabled=not files):
        loaded: list[Lot] = []
        errors: list[str] = []
        for file in files or []:
            try:
                loaded.extend(read_uploaded_file(file.name, file.getvalue()))
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
        st.session_state.lots = make_lot_ids_unique(loaded)
        st.session_state.completed_lot_ids = []
        st.session_state.all_results = []
        if errors:
            st.warning("\n".join(errors))
        if loaded:
            batches = math.ceil(len(st.session_state.lots) / MAX_LOTS_PER_BATCH)
            st.success(
                f"Se cargaron {len(st.session_state.lots)} lotes, organizados en "
                f"{batches} tanda{'s' if batches != 1 else ''} de hasta 10."
            )

with snowflake_tab:
    if not SNOWFLAKE_ENABLED:
        st.markdown("### Snowflake no está habilitado en esta edición")
        st.caption(
            "Por seguridad, la versión publicada no contiene credenciales. "
            "La conexión a Snowflake está disponible únicamente en la edición local."
        )
        st.button("Conectar a Snowflake", disabled=True, key="snowflake-disabled")
    elif not connection_available(settings):
        st.warning("Falta la configuración de Snowflake. Consulte .env.example.")
    else:
        st.write(
            "Filtre primero. En esta etapa solo se consultan nombres y cantidades; "
            "los contornos se descargan después de seleccionar los lotes."
        )
        if st.button("Cargar zonas y campos"):
            with st.spinner("Leyendo catálogo de Snowflake…"):
                st.session_state.snowflake_groups = fetch_filter_groups(settings)

        groups = st.session_state.snowflake_groups
        if groups:
            zones = sorted({row["zona"] for row in groups})
            zone_choice = st.selectbox("Zona", ["Todas"] + zones)
            fields = sorted(
                {
                    row["campo"]
                    for row in groups
                    if zone_choice == "Todas" or row["zona"] == zone_choice
                }
            )
            field_choice = st.selectbox("Campo / establecimiento", ["Todos"] + fields)
            search_text = st.text_input("Buscar por nombre de lote")
            catalog_limit = st.number_input("Máximo de resultados", 1, 2000, 250, 50)
            matching_count = sum(
                row["lotes"]
                for row in groups
                if (zone_choice == "Todas" or row["zona"] == zone_choice)
                and (field_choice == "Todos" or row["campo"] == field_choice)
            )
            st.caption(f"El filtro contiene {matching_count:,} lotes antes de buscar por nombre.")
            if st.button("Buscar lotes"):
                with st.spinner("Buscando lotes…"):
                    st.session_state.snowflake_catalog = search_lot_catalog(
                        settings,
                        zone=None if zone_choice == "Todas" else zone_choice,
                        establishment=None if field_choice == "Todos" else field_choice,
                        name_search=search_text,
                        limit=int(catalog_limit),
                    )

        catalog = st.session_state.snowflake_catalog
        if catalog:
            catalog_table = pd.DataFrame(catalog)
            catalog_table.insert(0, "seleccionar", False)
            selected_catalog = st.data_editor(
                catalog_table,
                hide_index=True,
                disabled=["id", "lote", "campo", "zona", "hectáreas"],
                width="stretch",
                key="snowflake_catalog_editor",
            )
            selected_ids = (
                selected_catalog.loc[selected_catalog["seleccionar"], "id"].astype(str).tolist()
            )
            if st.button(
                f"Traer {len(selected_ids)} contornos seleccionados",
                disabled=not selected_ids,
                type="primary",
            ):
                with st.spinner("Descargando únicamente los contornos elegidos…"):
                    st.session_state.lots = fetch_lots(settings, lot_ids=selected_ids)
                    st.session_state.completed_lot_ids = []
                    st.session_state.all_results = []
                st.success(f"{len(st.session_state.lots)} contornos listos para evaluar.")

lots: list[Lot] = st.session_state.lots
if lots:
    st.subheader("Lotes listos para procesar")
    completed_ids = set(st.session_state.completed_lot_ids)
    pending_lots = [lot for lot in lots if lot.lot_id not in completed_ids]
    if completed_ids:
        st.success(
            f"{len(completed_ids)} lotes completados · "
            f"{len(pending_lots)} pendientes en esta sesión."
        )
    if not pending_lots:
        st.success("Todos los lotes cargados fueron procesados.")
    else:
        st.write("Se recomienda procesar no más de 10 lotes por tanda.")
    table = pd.DataFrame(
        [
            {
                "procesar": index < MAX_LOTS_PER_BATCH,
                "id": lot.lot_id,
                "lote": lot.name.strip(),
                "campo": lot.establishment or "",
                "zona": lot.metadata.get("zona") or "",
                "origen": lot.source,
                "hectáreas": round(area_hectares(lot.geometry), 1),
            }
            for index, lot in enumerate(pending_lots)
        ]
    )
    selected: list[Lot] = []
    if not table.empty:
        editor_key = f"processing-lots-{len(completed_ids)}-{len(pending_lots)}"
        edited = st.data_editor(
            table,
            hide_index=True,
            disabled=["id", "lote", "campo", "zona", "origen", "hectáreas"],
            width="stretch",
            key=editor_key,
        )
        selected_ids = set(edited.loc[edited["procesar"], "id"].astype(str))
        selected = [lot for lot in pending_lots if lot.lot_id in selected_ids]
        st.info(
            f"Tanda actual: {len(selected)} lotes · "
            f"estimación {_duration(len(selected) * REFERENCE_SECONDS_PER_LOT)}."
        )
        if len(selected) > MAX_LOTS_PER_BATCH:
            st.error(
                f"Seleccione como máximo {MAX_LOTS_PER_BATCH} lotes. "
                "Esto evita saturar la memoria y los servicios satelitales."
            )
    if st.button(
        "Ejecutar tanda",
        type="primary",
        disabled=not selected or len(selected) > MAX_LOTS_PER_BATCH,
    ):
        options = ProcessingOptions(
            buffer_m=int(buffer_m),
            stability_seasons=int(seasons),
            max_cloud_percent=float(max_cloud),
            zone_counts=tuple(zone_counts),
            edge_exclusion_m=30,
            cache_review_arrays=SNOWFLAKE_ENABLED,
            export_quadrant_imagery=quadrant_images,
            calculate_productivity=productivity,
        )
        bar = st.progress(0.0)
        message = st.empty()
        clock = st.empty()
        started = time.monotonic()

        def update(label: str, fraction: float) -> None:
            elapsed = time.monotonic() - started
            eta = elapsed * (1 - fraction) / fraction if fraction >= 0.02 else None
            message.write(label)
            bar.progress(fraction)
            eta_text = f" · restante aproximado {_duration(eta)}" if eta else ""
            clock.markdown(f"⏱️ Transcurrido **{_duration(elapsed)}**{eta_text}")

        _, results = process_batch(selected, settings, options, progress=update)
        st.session_state.last_results = results
        result_by_id = {item.lot_id: item for item in st.session_state.all_results}
        result_by_id.update({item.lot_id: item for item in results})
        st.session_state.all_results = list(result_by_id.values())
        completed = set(st.session_state.completed_lot_ids)
        completed.update(item.lot_id for item in results if item.status in {"completed", "skipped"})
        st.session_state.completed_lot_ids = sorted(completed)
        st.session_state.last_elapsed = time.monotonic() - started
        st.success(
            f"Evaluación terminada en {_duration(st.session_state.last_elapsed)}. "
            "Revisar los resultados antes de descargar."
        )
        remaining = len([lot for lot in lots if lot.lot_id not in completed])
        if remaining:
            st.info(
                "Usar “Preparar siguiente tanda” para continuar sin volver a cargar los archivos."
            )
            if st.button("Preparar siguiente tanda"):
                st.rerun()


def _render_result(result: LotResult) -> None:
    elapsed = getattr(result, "elapsed_seconds", 0.0) or _elapsed_from_files(result.output_dir)
    with st.expander(
        f"{result.lot_name.strip()} · {result.status} · {_duration(elapsed)}",
        expanded=True,
    ):
        if result.status == "failed":
            st.error(result.error or "El procesamiento falló.")
            return
        root = result.output_dir
        boundary = root / "entrada" / "lote.geojson"
        if result.warnings:
            st.warning("\n".join(result.warnings))

        st.subheader("Clima")
        climate_file = root / "clima" / "analisis_climatico.json"
        if climate_file.exists():
            climate = json.loads(climate_file.read_text(encoding="utf-8"))
            rain_tab, coarse_tab, fine_tab = st.tabs(
                ["Precipitación anual", "Cuadrantes de gruesa", "Cuadrantes de fina"]
            )
            with rain_tab:
                st.plotly_chart(
                    annual_rainfall_figure(climate),
                    width="stretch",
                    config={"displaylogo": False},
                    key=f"rain-{result.lot_id}",
                )
            with coarse_tab:
                st.plotly_chart(
                    quadrant_figure(climate, "gruesa"),
                    width="stretch",
                    config={"displaylogo": False},
                    key=f"coarse-{result.lot_id}",
                )
            with fine_tab:
                st.plotly_chart(
                    quadrant_figure(climate, "fina"),
                    width="stretch",
                    config={"displaylogo": False},
                    key=f"fine-{result.lot_id}",
                )
            with st.expander("Ver datos climáticos en tabla"):
                st.dataframe(climate["annual_rainfall"], hide_index=True, width="stretch")
            with st.expander("Cómo se definieron fina, gruesa y los cuadrantes"):
                st.markdown(
                    """
- **Gruesa:** octubre a marzo. La campaña 2021, por ejemplo, es octubre de 2021 a
  marzo de 2022.
- **Fina:** abril a septiembre del mismo año.
- Solo se clasifica una campaña cuando están presentes sus seis meses. La precipitación
  se suma y la temperatura se promedia ponderando cada mes por su cantidad de días.
- Los cortes “cálido/frío” y “húmedo/seco” son las medias 1991–2020 de la propia serie.
  No son etiquetas de cultivo ni un modelo entrenado.
- Para las imágenes se elige, entre años Sentinel recientes, el más cercano al centro
  temperatura–precipitación de cada cuadrante.
- NASA POWER se consulta en el centroide del lote. ETP y balance P−ETP quedan en los
  datos exportados, pero los cuadrantes actuales usan únicamente temperatura y lluvia.
                    """
                )

        scene_metadata = list(root.glob("imagenes_cuadrantes/*/*/metadata.json"))
        if scene_metadata:
            st.subheader("Imágenes representativas")
            choices = {}
            scene_rows = [
                (metadata_path, json.loads(metadata_path.read_text(encoding="utf-8")))
                for metadata_path in scene_metadata
            ]
            scene_rows.sort(key=lambda item: str(item[1].get("date") or ""), reverse=True)
            for metadata_path, metadata in scene_rows:
                label = (
                    f"{metadata.get('date', '')} · "
                    f"{metadata.get('campaign', '').capitalize()} · "
                    f"{metadata.get('climate_quadrant', '')}"
                )
                choices[label] = (metadata_path.parent, metadata)
            scene_values = [item[1] for item in choices.values()]
            quality_columns = st.columns(3)
            quality_columns[0].metric("Escenas disponibles", len(scene_values))
            quality_columns[1].metric(
                "Máxima nubosidad",
                f"{max(float(item.get('cloud_percent') or 0) for item in scene_values):.2f}%",
            )
            quality_columns[2].metric(
                "Mínimo de píxeles válidos",
                f"{min(float(item.get('valid_pixel_percent') or 0) for item in scene_values):.1f}%",
            )
            if len(scene_values) < 8:
                st.caption(
                    f"Se generaron {len(scene_values)} de hasta 8 combinaciones. "
                    "Solo se muestran cuadrantes con un año Sentinel-2 representativo disponible."
                )
            selection = st.selectbox(
                "Campaña y cuadrante",
                list(choices),
                key=f"scene-{result.lot_id}",
            )
            scene_dir, metadata = choices[selection]
            st.caption(
                f"Nubosidad de escena: {metadata.get('cloud_percent', 0):.2f}% · "
                f"píxeles válidos: {metadata.get('valid_pixel_percent', 0):.1f}% · "
                f"área enmascarada: {metadata.get('contaminated_percent', 0):.2f}% · "
                f"resolución: {metadata.get('resolution_m', 10):.0f} m · "
                f"buffer: {metadata.get('buffer_m', 0)} m"
            )
            for column, product in zip(st.columns(3), ["NDVI", "IR", "RGB"], strict=True):
                raster = scene_dir / metadata.get("products", {}).get(product, f"{product}.tif")
                column.image(
                    cached_preview(
                        str(raster), raster.stat().st_mtime, str(boundary), boundary.stat().st_mtime
                    ),
                    caption=f"{product} · {metadata.get('date', '')} · borde amarillo = lote",
                    width="stretch",
                )

        stability_branches = [
            branch
            for branch in ("gruesa", "fina")
            if (root / "estabilidad" / branch / "estabilidad_5_clases.tif").exists()
        ]
        if stability_branches:
            st.subheader("Estabilidad y ambientes productivos")
            with st.expander("Cómo se calcularon estas capas"):
                st.markdown(
                    """
- **Fina y gruesa se calculan por separado.** Para cada campaña se construye el máximo
  NDVI libre de nubes. Cada
  campaña se normaliza respecto del propio lote; luego se calcula productividad relativa
  media y variabilidad temporal por píxel.
- Los 30 m interiores al alambrado no entrenan la normalización ni K-Means. Al final se
  asigna a ese borde la clase interior más cercana para entregar capas completas.
- **Ambientes:** se aplica **K-Means** sobre productividad relativa y variabilidad
  estandarizadas.
- La recomendación exige separación estadística y que ninguna zona ocupe menos del 10%.
  Aun así debe validarse con rendimiento, suelo, relieve y conocimiento del productor.
                    """
                )
            branch_tabs = st.tabs([branch.capitalize() for branch in stability_branches])
            for branch_tab, branch in zip(branch_tabs, stability_branches, strict=True):
                with branch_tab:
                    _render_productivity_branch(result, branch, boundary)

        package = result.output_dir.parent / f"{result.output_dir.name}.zip"
        if package.exists():
            st.download_button(
                "Descargar paquete completo para QGIS",
                package.read_bytes(),
                file_name=package.name,
                mime="application/zip",
                key=f"download-{result.lot_id}",
                type="primary",
            )


def _render_revision_history(
    root: Path,
    stability_dir: Path,
    result: LotResult,
    branch: str,
    *,
    expanded: bool = False,
) -> None:
    history_file = stability_dir / "historial_previews.json"
    if not history_file.exists():
        return
    history = json.loads(history_file.read_text(encoding="utf-8"))
    if not history:
        return
    with st.expander("Comparar con previews anteriores", expanded=expanded):
        ordered = sorted(
            history,
            key=lambda row: str(row.get("completed_at") or row.get("started_at") or ""),
            reverse=True,
        )
        labels = {
            (
                f"{row.get('completed_at', row.get('revision_id', ''))} · "
                f"{len(row.get('excluded_item_ids', []))} excluidas"
            ): row
            for row in ordered
        }
        choice = st.selectbox(
            "Recálculo",
            list(labels),
            key=f"revision-history-{result.lot_id}-{branch}",
        )
        revision = labels[choice]
        changed = revision.get("changed_class_pixels_percent")
        changed_text = f"{changed:.1f}%" if changed is not None else "no disponible"
        st.caption(
            f"Duración: {_duration(revision.get('duration_seconds'))} · "
            f"píxeles de estabilidad que cambiaron de clase: {changed_text}."
        )
        before_tab, after_tab = st.tabs(["Antes", "Después"])
        for tab, stage in ((before_tab, "before"), (after_tab, "after")):
            with tab:
                previews = revision.get("previews", {}).get(stage, [])
                for offset in range(0, len(previews), 3):
                    row = previews[offset : offset + 3]
                    for column, preview in zip(st.columns(len(row)), row, strict=True):
                        path = root / preview["path"]
                        if path.exists():
                            column.image(
                                str(path),
                                caption=preview.get("layer", path.stem).replace("_", " "),
                                width="stretch",
                            )


def _scene_can_be_included(row: dict) -> bool:
    reason = str(row.get("reason") or "")
    if reason.startswith("Descartada automáticamente") or reason.startswith("error de lectura"):
        return False
    if row.get("contaminated_percent") is not None:
        return (
            float(row.get("contaminated_percent") or 0) <= 0.5
            and float(row.get("largest_contaminated_patch_m2") or 0) <= 1_000
        )
    return float(row.get("valid_pixel_percent") or 0) >= 25


def _render_productivity_branch(result: LotResult, branch: str, boundary: Path) -> None:
    root = result.output_dir
    stability_dir = root / "estabilidad" / branch
    environments_dir = root / "ambientes" / branch
    stability = stability_dir / "estabilidad_5_clases.tif"
    alternatives = sorted(environments_dir.glob("ambientes_k*.tif"))
    review_file = stability_dir / "revision_humana.json"
    completion_key = f"recalculation-completed-{result.lot_id}-{branch}"
    just_recalculated = bool(st.session_state.pop(completion_key, False))
    if review_file.exists():
        review = json.loads(review_file.read_text(encoding="utf-8"))
        completed_at = review.get("completed_at") or review.get("updated_at") or ""
        st.success(
            f"Resultado revisado · {len(review.get('excluded_item_ids', []))} "
            f"escenas excluidas · recálculo terminado {completed_at} · "
            "el ZIP ya contiene el resultado actual."
        )
        before = review.get("before", {})
        after = review.get("after", {})
        if before and after:
            metrics = st.columns(4)
            metrics[0].metric(
                "Campañas utilizadas",
                after.get("campaign_count", 0),
                after.get("campaign_count", 0) - before.get("campaign_count", 0),
            )
            metrics[1].metric(
                "Escenas utilizadas",
                after.get("included_scene_count", 0),
                after.get("included_scene_count", 0) - before.get("included_scene_count", 0),
            )
            changed = review.get("changed_class_pixels_percent")
            metrics[2].metric(
                "Clases modificadas",
                f"{changed:.1f}%" if changed is not None else "n/d",
            )
            metrics[3].metric(
                "Duración del recálculo",
                _duration(review.get("duration_seconds")),
            )
    else:
        st.caption(
            "Resultado preliminar listo para descargar. La revisión de escenas es opcional."
        )
    _render_revision_history(
        root,
        stability_dir,
        result,
        branch,
        expanded=just_recalculated,
    )
    rasters = [stability, *alternatives]
    for offset in range(0, len(rasters), 3):
        row = rasters[offset : offset + 3]
        for column, raster in zip(st.columns(len(row)), row, strict=True):
            title = (
                f"Estabilidad de {branch} · 5 clases"
                if raster == stability
                else f"{branch.capitalize()} · {raster.stem.replace('_', ' ')}"
            )
            column.image(
                cached_preview(
                    str(raster), raster.stat().st_mtime, str(boundary), boundary.stat().st_mtime
                ),
                caption=title,
                width="stretch",
            )

    st.caption("Leyenda de estabilidad")
    legend_rows = "".join(
        (
            "<div style='display:flex;align-items:center;gap:.65rem;margin:.3rem 0'>"
            f"<span style='width:1.35rem;height:1.35rem;border-radius:.2rem;"
            f"background:rgb({red},{green},{blue});border:1px solid #777'></span>"
            f"<span><b>Clase {class_id}</b> · {STABILITY_LABELS[class_id]}</span></div>"
        )
        for class_id, (red, green, blue) in STABILITY_COLORS.items()
    )
    st.markdown(legend_rows, unsafe_allow_html=True)

    recommendation_file = environments_dir / "alternativas.json"
    if recommendation_file.exists():
        recommendation = json.loads(recommendation_file.read_text(encoding="utf-8"))
        effective_edge = recommendation.get("edge_training_exclusion_m", 30)
        st.caption(f"Borde excluido del ajuste: {effective_edge} m.")
        if recommendation.get("recommended_k"):
            st.info(
                f"Alternativa estadísticamente recomendada para {branch}: "
                f"{recommendation.get('recommended_k')} ambientes. "
                "Debe validarse agronómicamente."
            )
        else:
            st.warning(
                f"No hay una alternativa robusta para recomendar automáticamente en {branch}. "
                "Revise tamaños, formas y evidencia agronómica."
            )
    statistics_file = stability_dir / "estadisticas.csv"
    if statistics_file.exists():
        st.dataframe(pd.read_csv(statistics_file), hide_index=True, width="stretch")

    inventory_file = stability_dir / "escenas_utilizadas.json"
    if not inventory_file.exists():
        return
    with st.expander(f"Revisar o excluir imágenes de {branch}"):
        inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
        campaigns = sorted({str(row["campaign"]) for row in inventory}, reverse=True)
        campaign_filter = st.selectbox(
            "Campaña para inspeccionar",
            ["Todas", *campaigns],
            key=f"campaign-filter-{result.lot_id}-{branch}",
        )
        visible = sorted(
            [
            row
            for row in inventory
            if campaign_filter == "Todas" or row["campaign"] == campaign_filter
            ],
            key=lambda row: str(row.get("date") or ""),
            reverse=True,
        )
        review_table = pd.DataFrame(
            [
                {
                    "excluir": row.get("reason") == "excluida por el usuario",
                    "campaña": row.get("campaign"),
                    "fecha": row.get("date"),
                    "nubosidad_%": row.get("cloud_percent"),
                    "píxeles_válidos_%": row.get("valid_pixel_percent"),
                    "área_enmascarada_%": row.get("contaminated_percent"),
                    "estado": row.get("reason", "incluida"),
                    "item_id": row.get("item_id"),
                }
                for row in visible
            ]
        )
        edited = st.data_editor(
            review_table,
            hide_index=True,
            width="stretch",
            disabled=[
                "campaña",
                "fecha",
                "nubosidad_%",
                "píxeles_válidos_%",
                "área_enmascarada_%",
                "estado",
                "item_id",
            ],
            key=(
                f"scene-review-{result.lot_id}-{branch}-{campaign_filter}-"
                f"{inventory_file.stat().st_mtime_ns}"
            ),
        )
        full_campaigns = st.multiselect(
            "Excluir campañas completas",
            campaigns,
            key=f"excluded-campaigns-{result.lot_id}-{branch}",
        )
        # The false-colour browser always spans the complete inventory. The campaign
        # filter above is intentionally limited to the editable review table.
        preview_inventory = sorted(
            inventory,
            key=lambda row: str(row.get("date") or ""),
            reverse=True,
        )
        preview_choices = {
            (
                f"{row.get('date')} · {row.get('campaign')} · "
                f"{'INCLUIDA' if row.get('included') else 'EXCLUIDA'} · {row.get('item_id')}"
            ): row
            for row in preview_inventory
            if (stability_dir / str(row.get("preview", ""))).exists()
        }
        if preview_choices:
            preview_choice = st.selectbox(
                "Vista falso color · todas las campañas",
                list(preview_choices),
                key=f"scene-preview-{result.lot_id}-{branch}",
            )
            preview = stability_dir / preview_choices[preview_choice]["preview"]
            selected_preview = preview_choices[preview_choice]
            caption = preview_choice
            if selected_preview.get("reason"):
                caption += f" · {selected_preview.get('reason')}"
            st.image(str(preview), caption=caption, width="stretch")

        visible_ids = {str(row["item_id"]) for row in visible}
        excluded_ids = {
            str(row["item_id"])
            for row in inventory
            if row.get("reason") == "excluida por el usuario"
            and str(row["item_id"]) not in visible_ids
        }
        excluded_ids.update(edited.loc[edited["excluir"], "item_id"].astype(str))
        excluded_ids.update(
            str(row["item_id"]) for row in inventory if row["campaign"] in full_campaigns
        )
        matching_lot = next(
            (lot for lot in st.session_state.lots if lot.lot_id == result.lot_id), None
        )
        remaining_campaigns = {
            str(row["campaign"])
            for row in inventory
            if str(row["item_id"]) not in excluded_ids
            and _scene_can_be_included(row)
        }
        insufficient_campaigns = len(remaining_campaigns) < 3
        if insufficient_campaigns:
            st.warning(
                "La selección dejaría menos de tres campañas utilizables. "
                "Conserve imágenes de al menos tres campañas para poder recalcular."
            )
        if st.button(
            f"Recalcular sólo {branch}",
            disabled=matching_lot is None or insufficient_campaigns,
            key=f"recalculate-{result.lot_id}-{branch}",
        ):
            options = ProcessingOptions(
                buffer_m=int(buffer_m),
                stability_seasons=int(seasons),
                max_cloud_percent=float(max_cloud),
                zone_counts=tuple(zone_counts),
                edge_exclusion_m=30,
                cache_review_arrays=SNOWFLAKE_ENABLED,
                export_quadrant_imagery=quadrant_images,
                calculate_productivity=productivity,
            )
            bar = st.progress(0.0)
            message = st.empty()

            def review_progress(label: str, fraction: float) -> None:
                message.write(label)
                bar.progress(fraction)

            try:
                recalculate_productivity_campaign(
                    matching_lot,
                    root,
                    options,
                    branch,
                    excluded_ids,
                    progress=review_progress,
                )
            except Exception as exc:
                st.error(f"No se pudo recalcular {branch}: {exc}")
                return
            cached_preview.clear()
            st.session_state[completion_key] = True
            st.rerun()


def _elapsed_from_files(root: Path) -> float:
    first = root / "entrada" / "lote.geojson"
    last = root / ".completed.json"
    if first.exists() and last.exists():
        return max(0.0, last.stat().st_mtime - first.stat().st_mtime)
    return 0.0


results: list[LotResult] = st.session_state.all_results
if results:
    st.divider()
    st.header("Resultados para revisar")
    result_labels = {
        (
            f"{result.lot_name.strip()} · {result.status} · "
            f"{_duration(getattr(result, 'elapsed_seconds', 0))}"
        ): result
        for result in results
    }
    selected_result = st.selectbox(
        "Elegir lote procesado",
        list(result_labels),
        key="result-review-selector",
    )
    _render_result(result_labels[selected_result])
elif not lots:
    st.info("Suba archivos o use los filtros de Snowflake para comenzar.")
