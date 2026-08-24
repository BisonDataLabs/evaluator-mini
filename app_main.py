"""Streamlit interface for controlled selection, evaluation and visual review."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluador_lotes_mini.config import load_settings
from evaluador_lotes_mini.geometry import area_hectares
from evaluador_lotes_mini.ingestion.files import make_lot_ids_unique, read_uploaded_file
from evaluador_lotes_mini.ingestion.snowflake import (
    connection_available,
    fetch_filter_groups,
    fetch_lots,
    search_lot_catalog,
)
from evaluador_lotes_mini.models import Lot, LotResult, ProcessingOptions
from evaluador_lotes_mini.processor import process_batch
from evaluador_lotes_mini.ui_charts import annual_rainfall_figure, quadrant_figure
from evaluador_lotes_mini.ui_preview import render_raster

MAX_LOTS_PER_BATCH = 10
REFERENCE_SECONDS_PER_LOT = 8 * 60
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
    zone_counts = st.multiselect("Alternativas de ambientes", [2, 3, 4, 5], [2, 3, 4])
    quadrant_images = st.toggle("Imágenes por cuadrante climático", value=True)
    productivity = st.toggle("Estabilidad y ambientes", value=True)
    st.caption("Referencia observada: un lote completo tarda aproximadamente 7–10 minutos.")

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
        st.write(
            f"La herramienta procesa como máximo {MAX_LOTS_PER_BATCH} lotes por tanda. "
            "Al finalizar, la siguiente tanda queda preparada automáticamente."
        )
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
        batches_remaining = math.ceil(len(pending_lots) / MAX_LOTS_PER_BATCH)
        st.info(
            f"Tanda actual: {len(selected)} lotes · "
            f"estimación {_duration(len(selected) * REFERENCE_SECONDS_PER_LOT)} · "
            f"quedan aproximadamente {batches_remaining} tandas incluyendo esta."
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
            "Revise los resultados antes de descargar."
        )
        remaining = len([lot for lot in lots if lot.lot_id not in completed])
        if remaining:
            st.info(
                f"Quedan {remaining} lotes. Use “Preparar siguiente tanda” para continuar "
                "sin volver a cargar los archivos ni buscarlos en Snowflake."
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

        scene_metadata = sorted(root.glob("imagenes_cuadrantes/*/*/metadata.json"))
        if scene_metadata:
            st.subheader("Imágenes representativas")
            choices = {}
            for metadata_path in scene_metadata:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                label = (
                    f"{metadata.get('campaign', '').capitalize()} · "
                    f"{metadata.get('climate_quadrant', '')} · {metadata.get('date', '')}"
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
                f"resolución: {metadata.get('resolution_m', 10):.0f} m · "
                f"buffer: {metadata.get('buffer_m', 0)} m"
            )
            for column, product in zip(st.columns(3), ["RGB", "IR", "NDVI"], strict=True):
                raster = scene_dir / f"{product}.tif"
                column.image(
                    cached_preview(
                        str(raster), raster.stat().st_mtime, str(boundary), boundary.stat().st_mtime
                    ),
                    caption=f"{product} · borde amarillo = lote",
                    width="stretch",
                )

        stability = root / "estabilidad" / "estabilidad_5_clases.tif"
        alternatives = sorted((root / "ambientes").glob("ambientes_k*.tif"))
        if stability.exists():
            st.subheader("Estabilidad y ambientes productivos")
            with st.expander("Cómo se calcularon estas capas"):
                st.markdown(
                    """
- **Estabilidad:** para cada campaña se construye el máximo NDVI libre de nubes. Cada
  campaña se normaliza respecto del propio lote; luego se calcula productividad relativa
  media y variabilidad temporal por píxel.
- **Ambientes:** se aplica **K-Means**, un agrupamiento rígido, sobre productividad relativa
  y variabilidad estandarizadas. No es fuzzy: cada píxel pertenece a una sola zona.
- Ambas variables tienen inicialmente el mismo peso después de estandarizarlas. No se usa
  densidad aparente ni un peso agronómico manual.
- La recomendación exige separación estadística y que ninguna zona ocupe menos del 10%.
  Aun así debe validarse con rendimiento, suelo, relieve y conocimiento del productor.
                    """
                )
            rasters = [stability, *alternatives]
            columns = st.columns(len(rasters))
            for column, raster in zip(columns, rasters, strict=True):
                column.image(
                    cached_preview(
                        str(raster), raster.stat().st_mtime, str(boundary), boundary.stat().st_mtime
                    ),
                    caption=raster.stem.replace("_", " ").capitalize(),
                    width="stretch",
                )
            recommendation_file = root / "ambientes" / "alternativas.json"
            if recommendation_file.exists():
                recommendation = json.loads(recommendation_file.read_text(encoding="utf-8"))
                if recommendation.get("recommended_k"):
                    st.info(
                        f"Alternativa estadísticamente recomendada: "
                        f"{recommendation.get('recommended_k')} ambientes. "
                        "Debe validarse agronómicamente."
                    )
                else:
                    st.warning(
                        "No hay una alternativa robusta para recomendar automáticamente: "
                        "al menos un ambiente sería demasiado pequeño. Revise las capas junto "
                        "con rendimiento, suelo y conocimiento del lote."
                    )
            statistics_file = root / "estabilidad" / "estadisticas.csv"
            if statistics_file.exists():
                st.dataframe(pd.read_csv(statistics_file), hide_index=True, width="stretch")

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
