import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import requests
from supabase import create_client, Client
from concurrent.futures import ThreadPoolExecutor

# --- Configuración Supabase ---
SUPABASE_URL = "https://mhkbecpnmhiwcfpdazgu.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_KEY:
    raise ValueError("Error: No se encontró SUPABASE_SERVICE_ROLE_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Bounding Box de Venezuela (0.10° ~ 11 km) ---
LAT_MIN, LAT_MAX = 0.50, 12.50
LON_MIN, LON_MAX = -73.50, -59.50
STEP = 0.10  

def crear_grilla_filtrada():
    poligonos = []
    grid_id = 1
    for lat in np.arange(LAT_MIN, LAT_MAX, STEP):
        for lon in np.arange(LON_MIN, LON_MAX, STEP):
            c_lat = round(lat + (STEP / 2), 4)
            c_lon = round(lon + (STEP / 2), 4)
            
            # Máscara continental rápida (descarta océano Caribe profundo y Atlántico)
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

def consultar_open_meteo_chunk(lote_coords):
    lats = ",".join([str(c[0]) for c in lote_coords])
    lons = ",".join([str(c[1]) for c in lote_coords])
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&hourly=precipitation&past_days=3&forecast_days=2"
    
    try:
        r = requests.get(url, timeout=20).json()
        if not isinstance(r, list):
            r = [r]
            
        resultados = []
        for item in r:
            precip = item.get('hourly', {}).get('precipitation', [])
            p3d = sum(precip[:72]) if len(precip) >= 72 else 0.0
            openmeteo_48h = sum(precip[72:120]) if len(precip) >= 120 else 0.0
            resultados.append((p3d, openmeteo_48h))
        return resultados
    except Exception:
        return [(0.0, 0.0)] * len(lote_coords)

def procesar_nacional():
    print("Creando malla espacial continental (11 km)...")
    gdf = crear_grilla_filtrada()
    total_celdas = len(gdf)
    print(f"Malla lista: {total_celdas} celdas continentales.")
    
    batch_size = 50
    coords_lista = [(row['c_lat'], row['c_lon']) for idx, row in gdf.iterrows()]
    lotes = [coords_lista[i:i + batch_size] for i in range(0, total_celdas, batch_size)]
    
    print(f"Descargando datos en paralelo ({len(lotes)} lotes)...")
    datos_meteo_flatt = []
    
    # 10 hilos en paralelo para descargar la matriz completa en segundos
    with ThreadPoolExecutor(max_workers=10) as executor:
        resultados_lotes = list(executor.map(consultar_open_meteo_chunk, lotes))
        
    for res in resultados_lotes:
        datos_meteo_flatt.extend(res)
        
    registros = []
    for idx, (p3d_mm, openmeteo_48h_mm) in enumerate(datos_meteo_flatt):
        if idx >= total_celdas:
            break
        row = gdf.iloc[idx]
        grid_id = int(row['grid_id'])
        c_lat = row['c_lat']
        
        h_p3d = min(p3d_mm / 120.0, 1.0)
        h_openmeteo_48h = min(openmeteo_48h_mm / 100.0, 1.0)
        
        es_zona_montanosa = (c_lat > 7.5 and row['c_lon'] < -65.0)
        v_slope = 0.80 if es_zona_montanosa else 0.25
        h_era5_suelo = min((h_p3d * 0.7) + 0.1, 1.0)

        r_hydro = round((0.6 * h_p3d) + (0.4 * h_openmeteo_48h), 2)
        r_slide = round((0.4 * h_p3d) + (0.3 * h_era5_suelo) + (0.3 * v_slope), 2)
        r_max = round(max(r_hydro, r_slide), 2)

        if r_max < 0.30:
            clase, color = 'Bajo', '#1a9850'
        elif r_max < 0.55:
            clase, color = 'Moderado', '#ffffbf'
        elif r_max < 0.75:
            clase, color = 'Alto', '#fdae61'
        else:
            clase, color = 'Severo', '#d7191c'

        dominant = 'Hidrológico' if r_hydro >= r_slide else 'Deslizamiento'
        wkt_geom = row['geometry'].wkt

        registros.append({
            "grid_id": grid_id,
            "geom": f"SRID=4326;{wkt_geom}",
            "h_p3d": round(h_p3d, 2),
            "h_era5_suelo": round(h_era5_suelo, 2),
            "h_openmeteo_48h": round(h_openmeteo_48h, 2),
            "v_slope": round(v_slope, 2),
            "r_hydro": r_hydro,
            "r_slide": r_slide,
            "r_max": r_max,
            "dominant": dominant,
            "clase": clase,
            "color": color
        })

    print("Limpiando registros antiguos en Supabase...")
    supabase.table('grilla_riesgo').delete().neq('grid_id', 0).execute()
    
    print("Insertando datos en Supabase (Bloques de 500)...")
    for k in range(0, len(registros), 500):
        supabase.table('grilla_riesgo').insert(registros[k:k + 500]).execute()
        
    print("¡Proceso a 11 km completado con éxito!")

if __name__ == "__main__":
    procesar_nacional()
