"""Read one or many lots from common GIS exchange formats."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import fiona
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.ops import transform

from evaluador_lotes_mini.geometry import normalize_polygon, safe_slug
from evaluador_lotes_mini.models import Lot

NAME_FIELDS = ("name", "nombre", "lote", "LOTE", "Name", "NOMBRE", "id", "ID", "lote_id", "LOTE_ID")
ESTABLISHMENT_FIELDS = ("establecimiento", "ESTABLECIMIENTO", "campo", "CAMPO", "field", "FIELD")
MAX_ZIP_MEMBERS = 2_000
MAX_ZIP_UNCOMPRESSED_BYTES = 250 * 1024 * 1024


def read_uploaded_file(file_name: str, content: bytes) -> list[Lot]:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".kmz":
        return _read_kmz(file_name, content)
    if suffix == ".kml":
        return _read_kml(file_name, content)
    if suffix in {".json", ".geojson"}:
        return _read_geojson(file_name, content)
    if suffix == ".zip":
        return _read_zip(file_name, content)
    if suffix == ".gpkg":
        return _read_gpkg(file_name, content)
    raise ValueError(f"Formato no soportado: {suffix}")


def _read_geojson(file_name: str, content: bytes) -> list[Lot]:
    document = json.loads(content.decode("utf-8-sig"))
    if document.get("type") == "FeatureCollection":
        features = document.get("features", [])
    elif document.get("type") == "Feature":
        features = [document]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": document}]
    return _lots_from_features(features, Path(file_name).stem)


def _read_kmz(file_name: str, content: bytes) -> list[Lot]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        kml_files = [name for name in archive.namelist() if name.lower().endswith(".kml")]
        if not kml_files:
            raise ValueError("El KMZ no contiene un KML")
        return _read_kml(file_name, archive.read(kml_files[0]))


def _read_kml(file_name: str, content: bytes) -> list[Lot]:
    root = ET.fromstring(content)
    features: list[dict[str, Any]] = []
    placemarks = [node for node in root.iter() if node.tag.endswith("Placemark")]
    for index, placemark in enumerate(placemarks, start=1):
        name_node = next((n for n in placemark if n.tag.endswith("name")), None)
        name = (
            name_node.text.strip() if name_node is not None and name_node.text else f"Lote {index}"
        )
        for polygon in (n for n in placemark.iter() if n.tag.endswith("Polygon")):
            outer = next((n for n in polygon.iter() if n.tag.endswith("coordinates")), None)
            if outer is None or not outer.text:
                continue
            ring = _parse_kml_coordinates(outer.text)
            if len(ring) >= 3:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"name": name},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                )
    if not features:
        raise ValueError("No se encontraron polígonos en el KML/KMZ")
    return _lots_from_features(features, Path(file_name).stem)


def _parse_kml_coordinates(value: str) -> list[list[float]]:
    coords: list[list[float]] = []
    for token in value.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            coords.append([float(parts[0]), float(parts[1])])
    return coords


def _read_zip(file_name: str, content: bytes) -> list[Lot]:
    """Read a GeoJSON bundle or one or more Shapefiles from a safe ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        _validate_zip_members(members)
        geojson_members = [
            member
            for member in members
            if Path(member.filename).suffix.lower() in {".geojson", ".json"}
            and not Path(member.filename).name.startswith("._")
        ]
        if geojson_members:
            lots: list[Lot] = []
            errors: list[str] = []
            for member in geojson_members:
                try:
                    member_lots = _read_geojson(member.filename, archive.read(member))
                    member_slug = safe_slug(Path(member.filename).stem)
                    lots.extend(
                        replace(lot, lot_id=f"upload-{member_slug}-{safe_slug(lot.lot_id)}")
                        for lot in member_lots
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{member.filename}: {exc}")
            if lots:
                return make_lot_ids_unique(lots)
            detail = "; ".join(errors[:5])
            raise ValueError(f"El ZIP no contiene GeoJSON válidos. {detail}".strip())
    return _read_shapefile_zip(file_name, content)


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"El ZIP supera el máximo de {MAX_ZIP_MEMBERS} archivos")
    if sum(member.file_size for member in members) > MAX_ZIP_UNCOMPRESSED_BYTES:
        raise ValueError("El ZIP supera el máximo de 250 MB descomprimidos")
    for member in members:
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("El ZIP contiene una ruta insegura")


def _read_shapefile_zip(file_name: str, content: bytes) -> list[Lot]:
    with tempfile.TemporaryDirectory(prefix="elm-shp-") as directory:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            _validate_zip_members([member for member in archive.infolist() if not member.is_dir()])
            root = Path(directory).resolve()
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError("El ZIP contiene una ruta insegura")
            archive.extractall(directory)
        shapefiles = list(Path(directory).rglob("*.shp"))
        if not shapefiles:
            raise ValueError("El ZIP no contiene un Shapefile")
        return _read_vector_path(shapefiles[0], Path(file_name).stem)


def _read_gpkg(file_name: str, content: bytes) -> list[Lot]:
    with tempfile.TemporaryDirectory(prefix="elm-gpkg-") as directory:
        path = Path(directory) / Path(file_name).name
        path.write_bytes(content)
        layers = fiona.listlayers(path)
        lots: list[Lot] = []
        for layer in layers:
            lots.extend(_read_vector_path(path, layer, layer=layer))
        if not lots:
            raise ValueError("El GeoPackage no contiene polígonos")
        return lots


def make_lot_ids_unique(lots: list[Lot]) -> list[Lot]:
    """Return stable unique IDs so equal names from different files remain independent jobs."""
    occurrences: dict[str, int] = {}
    result: list[Lot] = []
    for lot in lots:
        base = lot.lot_id
        occurrences[base] = occurrences.get(base, 0) + 1
        occurrence = occurrences[base]
        result.append(lot if occurrence == 1 else replace(lot, lot_id=f"{base}-{occurrence}"))
    return result


def _read_vector_path(path: Path, fallback_name: str, layer: str | None = None) -> list[Lot]:
    features: list[dict[str, Any]] = []
    with fiona.open(path, layer=layer) as source:
        source_crs = CRS.from_user_input(source.crs) if source.crs else CRS.from_epsg(4326)
        transformer = None
        if source_crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(source_crs, 4326, always_xy=True)
        for feature in source:
            if not feature.get("geometry"):
                continue
            geom = shape(feature["geometry"])
            if transformer:
                geom = transform(transformer.transform, geom)
            features.append(
                {
                    "type": "Feature",
                    "properties": dict(feature.get("properties") or {}),
                    "geometry": geom.__geo_interface__,
                }
            )
    return _lots_from_features(features, fallback_name)


def _lots_from_features(features: list[dict[str, Any]], fallback_name: str) -> list[Lot]:
    lots: list[Lot] = []
    for index, feature in enumerate(features, start=1):
        if not feature.get("geometry"):
            continue
        try:
            geometry = normalize_polygon(shape(feature["geometry"]))
        except ValueError:
            continue
        properties = dict(feature.get("properties") or {})
        name = _first(properties, NAME_FIELDS) or f"{fallback_name} {index}"
        establishment = _first(properties, ESTABLISHMENT_FIELDS)
        external_id = _first(properties, ("lote_id", "LOTE_ID", "id", "ID"))
        lot_id = (
            str(external_id) if external_id is not None else f"upload-{safe_slug(name)}-{index}"
        )
        lots.append(
            Lot(
                lot_id=lot_id,
                name=str(name),
                geometry=geometry,
                source="upload",
                establishment=str(establishment) if establishment else None,
                metadata=properties,
            )
        )
    if not lots:
        raise ValueError("El archivo no contiene polígonos válidos")
    return lots


def _first(properties: dict[str, Any], fields: tuple[str, ...]) -> Any:
    return next((properties[f] for f in fields if properties.get(f) not in (None, "")), None)
