"""
================================================================================
ETL PIPELINE — PluviAMap v1.0
Metodología: Índice de Riesgo por Amenaza Lluvia (R = 0.6*A + 0.4*V)
Frecuencia: Cada 6 horas (GitHub Actions)
================================================================================
AMENAZA (A) — 60% del peso total
  Pf  (Lluvia pronosticada 24h)     : Open-Meteo  → 50% interno
  Pa  (Lluvia acumulada 3d)         : Open-Meteo  → 35% interno
  Imax(Intensidad máxima horaria)   : Open-Meteo  → 15% interno

VULNERABILIDAD (V) — 40% del peso total (estática, leída de Supabase)
  S   (Pendiente del terreno)       : SRTM        → 40% interno
  D   (Proximidad a drenaje)        : HydroSHEDS  → 35% interno  [proxy: TWI]
  U   (Uso de suelo / permeabilidad): SoilGrids   → 25% interno

REGLA DE OVERRIDE (Seguridad):
  Si Pf > 80 mm/24h  → R mínimo = 0.75 (Muy alto)
  Si Pf > 150 mm/24h → R mínimo = 0.85 (Crítico)

ESCALA DE 6 NIVELES (0 – 1):
  0.00–0.15  Mínimo   (#1a5f3c)
  0.15–0.30  Bajo     (#2d8a4e)
  0.30–0.50  Moderado (#ffd166)
  0.50–0.70  Alto     (#f77f00)
  0.70–0.85  Muy alto (#e63946)
  0.85–1.00  Crítico  (#9d0208)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon
import geopandas as gpd
from supabase import create_client, Client

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://mhkbecpnmhiwcfpdazgu.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_KEY:
    raise ValueError("Error: No se encontró SUPABASE_SERVICE_ROLE_KEY en las variables de entorno.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Bounding box Venezuela continental (STEP = 0.10° ~ 11 km)
# NOTA: La metodología cita 0.05° (~5.5 km). La implementación actual usa 0.10°
#       porque es el mínimo común con ERA5-Land (0.1° nativo). Para migrar a
#       0.05° se requiere recalcular toda la grilla y las capas estáticas.
LAT_MIN, LAT_MAX = 0.50, 12.50
LON_MIN, LON_MAX = -73.50, -59.50
STEP = 0.10

# ──────────────────────────────────────────────────────────────────────────────
# UMBRALES DE NORMALIZACIÓN (0 → 0, umbral → 1)
# Ajustados a eventos extremos de referencia en Venezuela.
# ──────────────────────────────────────────────────────────────────────────────
UMBRAL_PF_24H   = 150.0   # mm/24h  (Vargas 1999: >300 mm/día)
UMBRAL_PA_3D    = 300.0   # mm/3d
UMBRAL_PA_7D    = 500.0   # mm/7d
UMBRAL_IMAX     = 80.0    # mm/h    (pico torrencial)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CREAR GRILLA ESPACIAL (determinística — se recrea idéntica cada vez)
# ──────────────────────────────────────────────────────────────────────────────
def crear_grilla_filtrada():
    """Genera la grilla continental de Venezuela a 0.10°.
    Incluye máscara rápida para descartar océano Caribe y Atlántico."""
    poligonos = []
    grid_id = 1
    for lat in np.arange(LAT_MIN, LAT_MAX, STEP):
        for lon in np.arange(LON_MIN, LON_MAX, STEP):
            c_lat = round(lat + (STEP / 2), 4)
            c_lon = round(lon + (STEP / 2), 4)

            # Máscara continental rápida
            if c_lat > 11.8:
                continue
            if c_lat > 10.8 and c_lon > -66.0:
                continue

            poly = Polygon([
                (lon, lat),
                (lon + STEP, lat),
                (lon + STEP, lat + STEP),
                (lon, lat + STEP)
            ])

            poligonos.append({
                'grid_id': grid_id,
                'geometry': poly,
                'c_lat': c_lat,
                'c_lon': c_lon
            })
            grid_id += 1

    return gpd.GeoDataFrame(poligonos, crs="EPSG:4326")


# ──────────────────────────────────────────────────────────────────────────────
# 2. LEER CAPA ESTÁTICA (Vulnerabilidad + Exposición) DESDE SUPABASE
# ──────────────────────────────────────────────────────────────────────────────
def obtener_grilla_estaticos():
    """Descarga los campos estáticos (V y E) de Supabase sin traer geom.
    Usa paginación para evitar el límite de 1,000 filas."""
    campos = (
        "grid_id,v_slope,v_slope_degrees,v_permeability,"
        "v_twi,v_flooding,e_worldpop,poblacion_estimada"
    )
    registros = []
    start = 0
    batch = 1000

    while True:
        try:
            resp = supabase.table('grilla_riesgo').select(campos).range(start, start + batch - 1).execute()
            if not resp.data:
                break
            registros.extend(resp.data)
            if len(resp.data) < batch:
                break
            start += batch
        except Exception as e:
            print(f"[ERROR] Leyendo estáticos en rango {start}: {e}")
            break

    df = pd.DataFrame(registros)
    # Si la tabla está vacía (primera corrida), devolvemos DataFrame vacío
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. DESCARGA METEOROLÓGICA (Open-Meteo)
# ──────────────────────────────────────────────────────────────────────────────
def consultar_open_meteo_chunk(lote_coords):
    """Consulta Open-Meteo para un lote de coordenadas.
    Devuelve: Pa(7d), Pa(3d), Pf(24h), Pf(48h), Imax(24h).
    """
    lats = ",".join([str(c[0]) for c in lote_coords])
    lons = ",".join([str(c[1]) for c in lote_coords])

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lats}&longitude={lons}"
        f"&hourly=precipitation&past_days=7&forecast_days=2"
    )

    try:
        r = requests.get(url, timeout=30).json()
        if not isinstance(r, list):
            r = [r]

        resultados = []
        for item in r:
            precip = item.get('hourly', {}).get('precipitation', [])
            # Esperamos 216 horas = 9 días × 24h (7 pasado + 2 forecast)
            if len(precip) < 216:
                print(f"[ADVERTENCIA] Open-Meteo devolvió {len(precip)} horas, esperadas 216.")
                resultados.append({
                    'pa7d': 0.0, 'pa3d': 0.0,
                    'pf24h': 0.0, 'pf48h': 0.0, 'imax': 0.0
                })
                continue

            # Índices temporales en el array hourly
            # [0:168]   → 7 días pasados
            # [168:192] → primeras 24h forecast (Pf 24h)
            # [168:216] → 48h forecast (Pf 48h, legacy)
            pa7d   = sum(precip[0:168])
            pa3d   = sum(precip[96:168])     # últimas 72h del pasado
            pf24h  = sum(precip[168:192])    # metodología formal
            pf48h  = sum(precip[168:216])    # contexto visual
            imax   = max(precip[168:192]) if precip[168:192] else 0.0

            resultados.append({
                'pa7d':  round(pa7d, 2),
                'pa3d':  round(pa3d, 2),
                'pf24h': round(pf24h, 2),
                'pf48h': round(pf48h, 2),
                'imax':  round(imax, 2)
            })
        return resultados

    except Exception as e:
        print(f"[ERROR] Fallo en chunk Open-Meteo: {e}")
        return [{'pa7d': 0.0, 'pa3d': 0.0, 'pf24h': 0.0, 'pf48h': 0.0, 'imax': 0.0}] * len(lote_coords)


# ──────────────────────────────────────────────────────────────────────────────
# 4. FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────
def normalizar(valor, umbral):
    """Normaliza un valor a escala 0–1. Saturado en 1.0."""
    if umbral == 0:
        return 0.0
    return min(float(valor) / umbral, 1.0)


def clasificar_riesgo(r):
    """Clasificación en 6 niveles según metodología formal v1.0."""
    if r < 0.15:
        return 'Mínimo',   '#1a5f3c'
    if r < 0.30:
        return 'Bajo',     '#2d8a4e'
    if r < 0.50:
        return 'Moderado', '#ffd166'
    if r < 0.70:
        return 'Alto',     '#f77f00'
    if r < 0.85:
        return 'Muy alto', '#e63946'
    return 'Crítico',      '#9d0208'


def calcular_dominante(v_slope_degrees):
    """Heurística física: pendientes > 15° direccionan a deslizamiento."""
    return 'Deslizamiento' if v_slope_degrees > 15 else 'Hidrológico'


# ──────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
def procesar_nacional():
    print("=" * 60)
    print("PLUVIAMAP ETL v1.0 — Metodología Formal")
    print(f"Inicio: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # ── 5.1 Crear grilla espacial (determinística) ──
    print("
[1/6] Generando grilla espacial (0.10°)...")
    gdf = crear_grilla_filtrada()
    total_celdas = len(gdf)
    print(f"      → {total_celdas} celdas continentales.")

    # ── 5.2 Leer capa estática (Vulnerabilidad + Exposición) ──
    print("
[2/6] Leyendo capa estática desde Supabase (V y E)...")
    df_estaticos = obtener_grilla_estaticos()
    if not df_estaticos.empty:
        print(f"      → {len(df_estaticos)} celdas con datos estáticos.")
        gdf = gdf.merge(df_estaticos, on='grid_id', how='left')
    else:
        print("      → Tabla vacía. Primera corrida: todos los valores estáticos serán 0.")
        for col in ['v_slope', 'v_slope_degrees', 'v_permeability',
                    'v_twi', 'v_flooding', 'e_worldpop', 'poblacion_estimada']:
            gdf[col] = 0.0

    # ── 5.3 Descargar meteorología en paralelo ──
    print("
[3/6] Descargando datos meteorológicos (Open-Meteo)...")
    batch_size = 50
    coords_lista = [(row['c_lat'], row['c_lon']) for _, row in gdf.iterrows()]
    lotes = [coords_lista[i:i + batch_size] for i in range(0, total_celdas, batch_size)]

    datos_meteo_flatt = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        resultados_lotes = list(executor.map(consultar_open_meteo_chunk, lotes))
    for res in resultados_lotes:
        datos_meteo_flatt.extend(res)

    print(f"      → Meteorología descargada para {len(datos_meteo_flatt)} celdas.")

    # ── 5.4 Calcular Índice de Riesgo celda por celda ──
    print("
[4/6] Calculando Índice de Riesgo (R = 0.6*A + 0.4*V)...")
    registros = []

    for idx, meteo in enumerate(datos_meteo_flatt):
        if idx >= total_celdas:
            break

        row = gdf.iloc[idx]

        # ── Variables meteorológicas brutas ──
        pf24h  = meteo['pf24h']
        pa3d   = meteo['pa3d']
        pa7d   = meteo['pa7d']
        imax   = meteo['imax']
        pf48h  = meteo['pf48h']

        # ── Normalizar (0 → 1) ──
        n_pf   = normalizar(pf24h, UMBRAL_PF_24H)
        n_pa3d = normalizar(pa3d,  UMBRAL_PA_3D)
        n_imax = normalizar(imax,  UMBRAL_IMAX)

        # ── Factor AMENAZA (A) ──
        A = (0.50 * n_pf) + (0.35 * n_pa3d) + (0.15 * n_imax)

        # ── Factor VULNERABILIDAD (V) — leído de capa estática ──
        # Si aún no se han cargado capas estáticas (SRTM, etc.), V = 0.
        # Esto es esperado en las primeras corridas.
        S = float(row.get('v_slope', 0) or 0)
        D = float(row.get('v_twi', 0) or 0)      # proxy: TWI actúa como drenaje
        U_suelo = float(row.get('v_permeability', 0) or 0)
        V = (0.40 * S) + (0.35 * D) + (0.25 * U_suelo)
        V = min(V, 1.0)

        # ── Riesgo compuesto ──
        R = (0.6 * A) + (0.4 * V)

        # ── Regla de OVERRIDE de emergencia ──
        # Pf > 80 mm/24h  → mínimo Muy alto (0.75)
        # Pf > 150 mm/24h → mínimo Crítico (0.85)
        if pf24h > 80:
            R = max(R, 0.75)
        if pf24h > 150:
            R = max(R, 0.85)

        R = round(min(R, 1.0), 3)

        # ── Clasificación y metadatos ──
        clase, color = clasificar_riesgo(R)
        v_slope_deg = float(row.get('v_slope_degrees', 0) or 0)
        dominant = calcular_dominante(v_slope_deg)

        # ── Preparar registro completo para Supabase ──
        wkt_geom = row['geometry'].wkt
        registros.append({
            "grid_id": int(row['grid_id']),
            "geom": f"SRID=4326;{wkt_geom}",

            # Amenaza (dinámica)
            "h_p7d": round(pa7d, 2),
            "h_p3d": round(pa3d, 2),
            "h_era5_suelo": 0,   # TODO: integrar ERA5-Land real vía GEE o CDS API
            "h_openmeteo_24h": round(pf24h, 2),
            "h_openmeteo_48h": round(pf48h, 2),
            "h_imax": round(imax, 2),

            # Vulnerabilidad (estática — preservada desde lectura previa)
            "v_slope": round(S, 3),
            "v_slope_degrees": round(v_slope_deg, 2),
            "v_permeability": round(U_suelo, 3),
            "v_twi": round(D, 3),
            "v_flooding": round(float(row.get('v_flooding', 0) or 0), 3),

            # Exposición (estática)
            "e_worldpop": round(float(row.get('e_worldpop', 0) or 0), 3),
            "poblacion_estimada": int(row.get('poblacion_estimada', 0) or 0),

            # Índices calculados
            "r_hydro": round(R, 3),
            "r_slide": round(R, 3),
            "r_max": round(R, 3),
            "dominant": dominant,
            "clase": clase,
            "color": color,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

    print(f"      → {len(registros)} registros preparados.")

    # ── 5.5 Guardar en Supabase ──
    print("
[5/6] Sincronizando con Supabase...")
    try:
        # Eliminamos registros antiguos (atómico)
        supabase.table('grilla_riesgo').delete().neq('grid_id', 0).execute()
        print("      → Registros anteriores eliminados.")
    except Exception as e:
        print(f"      → [ADVERTENCIA] Error al eliminar: {e}")

    print("
[6/6] Insertando bloques de 500...")
    for k in range(0, len(registros), 500):
        bloque = registros[k:k + 500]
        try:
            supabase.table('grilla_riesgo').insert(bloque).execute()
            print(f"      → Insertados {k + len(bloque)} / {len(registros)}")
        except Exception as e:
            print(f"      → [ERROR] Insertando bloque {k}: {e}")

    print("
" + "=" * 60)
    print(f"ETL completado: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    procesar_nacional()
