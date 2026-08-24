"""Read-only Snowflake lot adapter using RSA key-pair authentication."""

from __future__ import annotations

import base64
import binascii
import json
import re
from contextlib import closing
from typing import Any

import snowflake.connector
from cryptography.hazmat.primitives import serialization
from shapely.geometry import shape

from evaluador_lotes_mini.config import Settings
from evaluador_lotes_mini.geometry import normalize_polygon
from evaluador_lotes_mini.models import Lot

LOT_QUERY = """
SELECT LOTE_ID, LOTE, TIPO_COORDENADAS, ST_ASGEOJSON(GEOMETRY), ESTABLECIMIENTO, ZONA
FROM {table}
WHERE GEOMETRY IS NOT NULL AND COALESCE(TIENE_POLIGONO, TRUE) {filters}
ORDER BY LOTE
"""


def connection_available(settings: Settings) -> bool:
    has_private_key = bool(
        settings.snowflake_private_key_pem
        or settings.snowflake_private_key_b64
        or (
            settings.snowflake_private_key_path
            and settings.snowflake_private_key_path.exists()
        )
    )
    return bool(
        settings.snowflake_account
        and settings.snowflake_user
        and has_private_key
    )


def fetch_lots(
    settings: Settings,
    limit: int | None = None,
    *,
    lot_ids: list[str] | None = None,
) -> list[Lot]:
    if not connection_available(settings):
        raise RuntimeError("Snowflake no está configurado o no se encontró la clave RSA")
    table = ".".join(
        _safe_identifier(value)
        for value in (settings.snowflake_database, settings.snowflake_schema, "SIMA_LOTES_GIS")
    )
    parameters: list[Any] = []
    filters = ""
    if lot_ids:
        placeholders = ", ".join(["%s"] * len(lot_ids))
        filters = f"AND LOTE_ID IN ({placeholders})"
        parameters.extend(lot_ids)
    query = LOT_QUERY.format(table=table, filters=filters)
    if limit:
        query += " LIMIT %s"
        parameters.append(int(limit))
    private_key = _load_private_key(settings)
    with (
        closing(
            snowflake.connector.connect(
                account=settings.snowflake_account,
                user=settings.snowflake_user,
                role=settings.snowflake_role,
                warehouse=settings.snowflake_warehouse,
                database=settings.snowflake_database,
                schema=settings.snowflake_schema,
                authenticator="SNOWFLAKE_JWT",
                private_key=private_key,
                session_parameters={"QUERY_TAG": "evaluador-lotes-mini"},
            )
        ) as connection,
        closing(connection.cursor()) as cursor,
    ):
        cursor.execute(query, parameters or None)
        rows = cursor.fetchall()

    lots: list[Lot] = []
    for lot_id, name, coord_type, coordinates, establishment, zone in rows:
        try:
            geometry = _coordinates_to_geometry(coord_type, coordinates)
            lots.append(
                Lot(
                    lot_id=str(lot_id),
                    name=str(name or lot_id),
                    geometry=normalize_polygon(geometry),
                    source="snowflake",
                    establishment=str(establishment) if establishment else None,
                    metadata={"tipo_coordenadas": coord_type, "zona": zone},
                )
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return lots


def fetch_filter_groups(settings: Settings) -> list[dict[str, Any]]:
    """Return compact zone/establishment counts without transferring geometries."""
    table = _table_name(settings)
    query = f"""
SELECT
  COALESCE(NULLIF(TRIM(ZONA), ''), 'Sin zona') AS ZONA,
  COALESCE(NULLIF(TRIM(ESTABLECIMIENTO), ''), 'Sin campo') AS CAMPO,
  COUNT(*) AS LOTES
FROM {table}
WHERE GEOMETRY IS NOT NULL AND COALESCE(TIENE_POLIGONO, TRUE)
GROUP BY 1, 2
ORDER BY 1, 2
"""
    rows = _fetch_rows(settings, query)
    return [{"zona": row[0], "campo": row[1], "lotes": int(row[2])} for row in rows]


def search_lot_catalog(
    settings: Settings,
    *,
    zone: str | None = None,
    establishment: str | None = None,
    name_search: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Search lot metadata first; geometries are fetched only after user selection."""
    conditions = ["GEOMETRY IS NOT NULL", "COALESCE(TIENE_POLIGONO, TRUE)"]
    parameters: list[Any] = []
    if zone:
        conditions.append("COALESCE(NULLIF(TRIM(ZONA), ''), 'Sin zona') = %s")
        parameters.append(zone)
    if establishment:
        conditions.append("COALESCE(NULLIF(TRIM(ESTABLECIMIENTO), ''), 'Sin campo') = %s")
        parameters.append(establishment)
    if name_search and name_search.strip():
        conditions.append("LOTE ILIKE %s")
        parameters.append(f"%{name_search.strip()}%")
    parameters.append(max(1, min(2_000, int(limit))))
    query = f"""
SELECT LOTE_ID, TRIM(LOTE), TRIM(ESTABLECIMIENTO), TRIM(ZONA),
       COALESCE(AREA_POLIGONO_HA, SUPERFICIE)
FROM {_table_name(settings)}
WHERE {" AND ".join(conditions)}
ORDER BY ESTABLECIMIENTO, LOTE
LIMIT %s
"""
    rows = _fetch_rows(settings, query, parameters)
    return [
        {
            "id": str(row[0]),
            "lote": row[1] or str(row[0]),
            "campo": row[2] or "Sin campo",
            "zona": row[3] or "Sin zona",
            "hectáreas": round(float(row[4]), 1) if row[4] is not None else None,
        }
        for row in rows
    ]


def _load_private_key(settings: Settings) -> bytes:
    if settings.snowflake_private_key_pem:
        pem = settings.snowflake_private_key_pem.replace("\\n", "\n").encode("utf-8")
    elif settings.snowflake_private_key_b64:
        try:
            pem = base64.b64decode(settings.snowflake_private_key_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("SNOWFLAKE_PRIVATE_KEY_B64 no contiene Base64 válido") from exc
    elif settings.snowflake_private_key_path:
        pem = settings.snowflake_private_key_path.read_bytes()
    else:
        raise RuntimeError("No se configuró una clave privada RSA para Snowflake")

    password = (
        settings.snowflake_private_key_passphrase.encode("utf-8")
        if settings.snowflake_private_key_passphrase
        else None
    )
    key = serialization.load_pem_private_key(pem, password=password)
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _table_name(settings: Settings) -> str:
    return ".".join(
        _safe_identifier(value)
        for value in (settings.snowflake_database, settings.snowflake_schema, "SIMA_LOTES_GIS")
    )


def _fetch_rows(settings: Settings, query: str, parameters: list[Any] | None = None) -> list[tuple]:
    if not connection_available(settings):
        raise RuntimeError("Snowflake no está configurado o no se encontró la clave RSA")
    with (
        closing(
            snowflake.connector.connect(
                account=settings.snowflake_account,
                user=settings.snowflake_user,
                role=settings.snowflake_role,
                warehouse=settings.snowflake_warehouse,
                database=settings.snowflake_database,
                schema=settings.snowflake_schema,
                authenticator="SNOWFLAKE_JWT",
                private_key=_load_private_key(settings),
                session_parameters={"QUERY_TAG": "evaluador-lotes-mini"},
            )
        ) as connection,
        closing(connection.cursor()) as cursor,
    ):
        cursor.execute(query, parameters or None)
        return cursor.fetchall()


def _coordinates_to_geometry(coord_type: str | None, coordinates: Any):
    value = json.loads(coordinates) if isinstance(coordinates, str) else coordinates
    if isinstance(value, dict) and value.get("type"):
        return shape(value)
    if isinstance(value, dict) and value.get("coordinates"):
        geometry_type = value.get("type") or coord_type or "Polygon"
        return shape({"type": geometry_type, "coordinates": value["coordinates"]})
    if isinstance(value, list):
        return shape({"type": coord_type or "Polygon", "coordinates": value})
    raise ValueError("Formato de coordenadas Snowflake desconocido")


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", value):
        raise ValueError(f"Identificador Snowflake inválido: {value!r}")
    return value
