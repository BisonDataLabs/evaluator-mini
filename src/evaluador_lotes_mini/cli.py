"""Command-line entry point suited to unattended bulk processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluador_lotes_mini.config import load_settings
from evaluador_lotes_mini.ingestion.files import read_uploaded_file
from evaluador_lotes_mini.ingestion.snowflake import fetch_lots
from evaluador_lotes_mini.models import Lot, ProcessingOptions
from evaluador_lotes_mini.processor import process_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluador-lotes",
        description="Evaluación satelital y zonificación productiva de lotes.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", type=Path, help="KMZ/KML/GeoJSON/ZIP/GPKG")
    source.add_argument("--snowflake", action="store_true", help="Leer lotes desde Snowflake")
    parser.add_argument("--lot-id", action="append", default=[], help="Filtrar ID de Snowflake")
    parser.add_argument("--limit", type=int, help="Máximo de lotes Snowflake")
    parser.add_argument("--batch-name", help="Nombre estable; permite reanudar la corrida")
    parser.add_argument("--buffer-m", type=int, default=500)
    parser.add_argument("--start-year", type=int, default=1985)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--stability-seasons", type=int, default=8)
    parser.add_argument("--max-cloud", type=float, default=30)
    parser.add_argument("--zones", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--edge-exclusion-m", type=int, default=30)
    parser.add_argument("--no-scene-cache", action="store_true")
    parser.add_argument("--skip-quadrant-images", action="store_true")
    parser.add_argument("--skip-productivity", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    lots: list[Lot] = []
    if args.input:
        for path in args.input:
            lots.extend(read_uploaded_file(path.name, path.read_bytes()))
    else:
        lots = fetch_lots(settings, limit=args.limit)
        if args.lot_id:
            requested = set(args.lot_id)
            lots = [lot for lot in lots if lot.lot_id in requested]
    if not lots:
        print("No se encontraron lotes para procesar.", file=sys.stderr)
        return 2
    options = ProcessingOptions(
        buffer_m=args.buffer_m,
        start_year=args.start_year,
        end_year=args.end_year,
        stability_seasons=args.stability_seasons,
        max_cloud_percent=args.max_cloud,
        zone_counts=tuple(sorted(set(args.zones))),
        edge_exclusion_m=args.edge_exclusion_m,
        cache_review_arrays=not args.no_scene_cache,
        export_quadrant_imagery=not args.skip_quadrant_images,
        calculate_productivity=not args.skip_productivity,
    )

    def show(message: str, fraction: float) -> None:
        print(f"{fraction:6.1%}  {message}", flush=True)

    batch_dir, results = process_batch(
        lots,
        settings,
        options,
        batch_name=args.batch_name,
        resume=not args.no_resume,
        progress=show,
    )
    failures = [item for item in results if item.status == "failed"]
    print(f"Resultados: {batch_dir}")
    print(f"Completados/reanudados: {len(results) - len(failures)}; fallidos: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
