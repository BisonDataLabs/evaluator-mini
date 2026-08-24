"""Ejecutar desde la consola Python de QGIS con exec(open(ruta).read())."""

from pathlib import Path

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRendererCategory,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor

BASE = Path(__file__).resolve().parent.parent
PROJECT = QgsProject.instance()

for geopackage in (BASE / "gis").glob("*.gpkg"):
    for layer_name in (
        "perimetro",
        "estabilidad_5_clases",
        "ambientes_k2",
        "ambientes_k3",
        "ambientes_k4",
    ):
        layer = QgsVectorLayer(f"{geopackage}|layername={layer_name}", layer_name, "ogr")
        if layer.isValid():
            if layer_name.startswith("ambientes_"):
                count = int(layer_name[-1])
                palette = ["#a6611a", "#dfc27d", "#80cdc1", "#018571", "#00441b"]
                categories = []
                for zone in range(1, count + 1):
                    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
                    symbol.setColor(QColor(palette[zone - 1]))
                    categories.append(QgsRendererCategory(zone, symbol, f"Zona {zone}"))
                layer.setRenderer(QgsCategorizedSymbolRenderer("zone", categories))
            PROJECT.addMapLayer(layer)

for raster in BASE.rglob("*.tif"):
    layer = QgsRasterLayer(str(raster), raster.stem)
    if layer.isValid():
        PROJECT.addMapLayer(layer)

print("Evaluador de lotes: capas cargadas")
