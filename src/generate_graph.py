"""
Script para generar un grafo de la ciudad de Sogamoso usando OSMnx
Exporta nodos y aristas a archivos GeoJSON con información de distancia, velocidad y dirección
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd

# Configurar OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

def create_sogamoso_graph():
    """
    Crea el grafo de la red vial de Sogamoso, Boyacá, Colombia
    """
    print("Descargando datos de OpenStreetMap para Sogamoso...")
    
    # Obtener el grafo de la red vial de Sogamoso
    # network_type='drive' incluye todas las calles transitables en vehículo
    place_name = "Sogamoso, Boyacá, Colombia"
    
    try:
        G = ox.graph_from_place(place_name, network_type='drive')
        print(f"Grafo descargado exitosamente: {len(G.nodes)} nodos, {len(G.edges)} aristas")
    except Exception as e:
        print(f"Error al descargar el grafo: {e}")
        return None, None
    
    return G

def add_edge_attributes(G):
    """
    Añade y calcula atributos adicionales para las aristas
    """
    print("\nCalculando atributos de las aristas...")
    
    # Añadir velocidades y tiempos de viaje
    # speed_kph: velocidad en km/h basada en el tipo de vía
    G = ox.add_edge_speeds(G)
    
    # travel_time: tiempo de viaje en segundos
    G = ox.add_edge_travel_times(G)
    
    print("Atributos añadidos (velocidad y tiempo de viaje)")
    
    return G

def export_to_geojson(G, output_dir='.'):
    """
    Exporta nodos y aristas a archivos GeoJSON
    """
    print("\nExportando a archivos GeoJSON...")
    
    # Convertir nodos a GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G)
    
    # Preparar nodos
    # Mantener solo las columnas relevantes
    nodes_export = nodes[['geometry']].copy()
    nodes_export['node_id'] = nodes.index
    nodes_export['lat'] = nodes['y']
    nodes_export['lon'] = nodes['x']
    
    # Preparar aristas
    # Seleccionar columnas relevantes
    edges_columns = ['geometry', 'length', 'oneway', 'highway']
    
    # Añadir columnas si existen
    if 'speed_kph' in edges.columns:
        edges_columns.append('speed_kph')
    if 'travel_time' in edges.columns:
        edges_columns.append('travel_time')
    if 'maxspeed' in edges.columns:
        edges_columns.append('maxspeed')
    if 'lanes' in edges.columns:
        edges_columns.append('lanes')
    if 'name' in edges.columns:
        edges_columns.append('name')
    
    # Filtrar solo las columnas que existen
    edges_columns = [col for col in edges_columns if col in edges.columns]
    edges_export = edges[edges_columns].copy()
    
    # Añadir información de los nodos de origen y destino
    edges_export['from_node'] = edges.index.get_level_values(0)
    edges_export['to_node'] = edges.index.get_level_values(1)
    
    # Añadir dirección (basado en oneway)
    if 'oneway' in edges_export.columns:
        edges_export['direction'] = edges_export['oneway'].apply(
            lambda x: 'one-way' if x else 'bidirectional'
        )
    else:
        edges_export['direction'] = 'unknown'
    
    # Redondear valores numéricos para mejor legibilidad
    if 'length' in edges_export.columns:
        edges_export['length'] = edges_export['length'].round(2)
    if 'speed_kph' in edges_export.columns:
        edges_export['speed_kph'] = edges_export['speed_kph'].round(2)
    if 'travel_time' in edges_export.columns:
        edges_export['travel_time'] = edges_export['travel_time'].round(2)
    
    # Exportar a GeoJSON
    nodes_file = f"{output_dir}/nodes.geojson"
    edges_file = f"{output_dir}/edges.geojson"
    
    nodes_export.to_file(nodes_file, driver='GeoJSON')
    edges_export.to_file(edges_file, driver='GeoJSON')
    
    print(f"Nodos exportados a: {nodes_file}")
    print(f"  - Total de nodos: {len(nodes_export)}")
    print(f"\nAristas exportadas a: {edges_file}")
    print(f"  - Total de aristas: {len(edges_export)}")
    print(f"  - Columnas incluidas: {list(edges_export.columns)}")
    
    return nodes_export, edges_export

def print_statistics(G):
    """
    Imprime estadísticas del grafo
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL GRAFO DE SOGAMOSO")
    print("="*60)
    
    # Estadísticas básicas
    print(f"\nNodos: {len(G.nodes)}")
    print(f"Aristas: {len(G.edges)}")
    
    # Obtener GeoDataFrames para análisis
    nodes, edges = ox.graph_to_gdfs(G)
    
    # Estadísticas de distancia
    print(f"\nDistancias (metros):")
    print(f"  - Total: {edges['length'].sum():,.2f} m")
    print(f"  - Promedio: {edges['length'].mean():,.2f} m")
    print(f"  - Mínima: {edges['length'].min():,.2f} m")
    print(f"  - Máxima: {edges['length'].max():,.2f} m")
    
    # Estadísticas de velocidad
    if 'speed_kph' in edges.columns:
        print(f"\nVelocidades (km/h):")
        print(f"  - Promedio: {edges['speed_kph'].mean():,.2f} km/h")
        print(f"  - Mínima: {edges['speed_kph'].min():,.2f} km/h")
        print(f"  - Máxima: {edges['speed_kph'].max():,.2f} km/h")
    
    # Tipos de vías
    if 'highway' in edges.columns:
        print(f"\nTipos de vías:")
        highway_counts = edges['highway'].value_counts()
        for highway_type, count in highway_counts.head(10).items():
            print(f"  - {highway_type}: {count}")
    
    # Direccionalidad
    if 'oneway' in edges.columns:
        oneway_count = edges['oneway'].sum()
        print(f"\nDireccionalidad:")
        print(f"  - Vías de un solo sentido: {oneway_count}")
        print(f"  - Vías bidireccionales: {len(edges) - oneway_count}")
    
    print("="*60)

def main():
    """
    Función principal
    """
    print("="*60)
    print("GENERADOR DE GRAFO - SOGAMOSO, BOYACÁ")
    print("="*60)
    
    # Crear el grafo
    G = create_sogamoso_graph()
    
    if G is None:
        print("\nError: No se pudo crear el grafo")
        return
    
    # Añadir atributos adicionales
    G = add_edge_attributes(G)
    
    # Exportar a GeoJSON
    nodes, edges = export_to_geojson(G)
    
    # Mostrar estadísticas
    print_statistics(G)
    
    print("\nProceso completado exitosamente!")
    print("\nArchivos generados:")
    print("  - nodes.geojson: Contiene todos los nodos (intersecciones)")
    print("  - edges.geojson: Contiene todas las aristas (calles) con:")
    print("    • Distancia (length)")
    print("    • Velocidad (speed_kph)")
    print("    • Dirección (direction/oneway)")
    print("    • Tiempo de viaje (travel_time)")
    print("    • Tipo de vía (highway)")
    print("    • Otros atributos disponibles")

if __name__ == "__main__":
    main()
