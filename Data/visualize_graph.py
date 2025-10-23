"""
Script para visualizar el grafo de Sogamoso
Genera visualizaciones estáticas e interactivas del grafo
"""

import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from folium import plugins
import webbrowser
import os

def load_graph_from_geojson():
    """
    Carga el grafo desde los archivos GeoJSON
    """
    print("Cargando datos desde archivos GeoJSON...")
    
    try:
        import json
        
        # Cargar nodos
        nodes = gpd.read_file('nodes.geojson')
        
        # Cargar edges con todas las propiedades desde JSON
        with open('edges.geojson', 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        # Extraer propiedades y geometrías
        features = edges_data['features']
        properties_list = []
        geometries = []
        
        for feature in features:
            props = feature['properties'].copy()
            properties_list.append(props)
            geometries.append(feature['geometry'])
        
        # Crear GeoDataFrame desde las propiedades y geometrías
        import pandas as pd
        from shapely.geometry import shape
        
        edges_df = pd.DataFrame(properties_list)
        edges_df['geometry'] = [shape(geom) for geom in geometries]
        edges = gpd.GeoDataFrame(edges_df, geometry='geometry', crs='EPSG:4326')
        
        print(f"✓ Cargados {len(nodes)} nodos y {len(edges)} aristas")
        print(f"  Columnas de aristas: {list(edges.columns)}")
        return nodes, edges
    except Exception as e:
        print(f"❌ Error al cargar archivos: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_static_visualization(nodes, edges):
    """
    Crea visualización estática del grafo con matplotlib
    """
    print("\nCreando visualización estática...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Grafo de Sogamoso - Visualizaciones', fontsize=20, fontweight='bold')
    
    # 1. Vista general del grafo
    ax1 = axes[0, 0]
    edges.plot(ax=ax1, linewidth=0.5, color='#2b8cbe', alpha=0.6)
    nodes.plot(ax=ax1, markersize=1, color='red', alpha=0.3)
    ax1.set_title('Vista General del Grafo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    ax1.grid(True, alpha=0.3)
    
    # 2. Grafo coloreado por tipo de vía
    ax2 = axes[0, 1]
    
    # Definir colores para cada tipo de vía
    highway_colors = {
        'residential': '#78c679',
        'unclassified': '#d9d9d9',
        'tertiary': '#fc8d59',
        'primary': '#e31a1c',
        'secondary': '#fd8d3c',
        'tertiary_link': '#fdd49e',
        'primary_link': '#fb6a4a',
        'secondary_link': '#feb24c'
    }
    
    for highway_type, color in highway_colors.items():
        highway_edges = edges[edges['highway'] == highway_type]
        if len(highway_edges) > 0:
            highway_edges.plot(ax=ax2, linewidth=1, color=color, alpha=0.7, label=highway_type)
    
    ax2.set_title('Grafo por Tipo de Vía', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Grafo coloreado por velocidad
    ax3 = axes[1, 0]
    
    if 'speed_kph' in edges.columns:
        edges.plot(ax=ax3, column='speed_kph', linewidth=1, 
                  cmap='YlOrRd', legend=True, alpha=0.8,
                  legend_kwds={'label': 'Velocidad (km/h)', 'shrink': 0.8})
        ax3.set_title('Grafo por Velocidad', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Datos de velocidad no disponibles', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Velocidad no disponible', fontsize=14)
    
    ax3.set_xlabel('Longitud')
    ax3.set_ylabel('Latitud')
    ax3.grid(True, alpha=0.3)
    
    # 4. Grafo coloreado por longitud de arista
    ax4 = axes[1, 1]
    
    if 'length' in edges.columns:
        edges.plot(ax=ax4, column='length', linewidth=1, 
                  cmap='viridis', legend=True, alpha=0.8,
                  legend_kwds={'label': 'Longitud (m)', 'shrink': 0.8})
        ax4.set_title('Grafo por Longitud de Arista', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Datos de longitud no disponibles', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Longitud no disponible', fontsize=14)
    
    ax4.set_xlabel('Longitud')
    ax4.set_ylabel('Latitud')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    output_file = 'sogamoso_graph_static.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualización estática guardada en: {output_file}")
    
    plt.show()

def create_interactive_map(nodes, edges):
    """
    Crea un mapa interactivo con Folium
    """
    print("\nCreando mapa interactivo...")
    
    # Calcular el centro del mapa
    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()
    
    # Crear mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Añadir capas de tiles adicionales
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Crear grupos de características para organizar las capas
    edges_group = folium.FeatureGroup(name='Aristas (Calles)', show=True)
    nodes_group = folium.FeatureGroup(name='Nodos (Intersecciones)', show=False)
    
    # Colores por tipo de vía
    highway_colors = {
        'residential': '#78c679',
        'unclassified': '#969696',
        'tertiary': '#fc8d59',
        'primary': '#e31a1c',
        'secondary': '#fd8d3c',
        'tertiary_link': '#fdd49e',
        'primary_link': '#fb6a4a',
        'secondary_link': '#feb24c'
    }
    
    # Añadir aristas al mapa
    print("Añadiendo aristas al mapa...")
    for idx, row in edges.iterrows():
        if idx % 500 == 0:
            print(f"  Procesadas {idx}/{len(edges)} aristas...")
        
        # Obtener color según tipo de vía
        highway_type = row.get('highway', 'unclassified')
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'unclassified'
        color = highway_colors.get(highway_type, '#969696')
        
        # Crear popup con información
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Tipo:</b> {highway_type}<br>
            <b>Longitud:</b> {row.get('length', 'N/A'):.2f} m<br>
            <b>Velocidad:</b> {row.get('speed_kph', 'N/A')} km/h<br>
            <b>Dirección:</b> {row.get('direction', 'N/A')}<br>
            <b>Nombre:</b> {row.get('name', 'Sin nombre')}
        </div>
        """
        
        # Convertir geometría a coordenadas
        coords = [(coord[1], coord[0]) for coord in row.geometry.coords]
        
        folium.PolyLine(
            coords,
            color=color,
            weight=2,
            opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{highway_type}"
        ).add_to(edges_group)
    
    print(f"✓ {len(edges)} aristas añadidas")
    
    # Añadir algunos nodos (muestra) al mapa
    print("Añadiendo muestra de nodos al mapa...")
    sample_nodes = nodes.sample(min(200, len(nodes)))  # Muestra de nodos para no sobrecargar
    
    for idx, row in sample_nodes.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Node ID:</b> {row.get('node_id', 'N/A')}<br>
            <b>Lat:</b> {row.get('lat', 'N/A'):.6f}<br>
            <b>Lon:</b> {row.get('lon', 'N/A'):.6f}
        </div>
        """
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(nodes_group)
    
    print(f"✓ {len(sample_nodes)} nodos añadidos (muestra)")
    
    # Añadir grupos al mapa
    edges_group.add_to(m)
    nodes_group.add_to(m)
    
    # Añadir control de capas
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Añadir escala
    plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    
    # Añadir botón de pantalla completa
    plugins.Fullscreen(position='topright').add_to(m)
    
    # Añadir mini mapa
    plugins.MiniMap(toggle_display=True).add_to(m)
    
    # Añadir leyenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin:0; font-weight:bold; text-align:center;">Tipos de Vía</p>
    <p style="margin:5px 0;"><span style="background-color:#e31a1c; padding: 3px 10px;">&nbsp;</span> Primaria</p>
    <p style="margin:5px 0;"><span style="background-color:#fd8d3c; padding: 3px 10px;">&nbsp;</span> Secundaria</p>
    <p style="margin:5px 0;"><span style="background-color:#fc8d59; padding: 3px 10px;">&nbsp;</span> Terciaria</p>
    <p style="margin:5px 0;"><span style="background-color:#78c679; padding: 3px 10px;">&nbsp;</span> Residencial</p>
    <p style="margin:5px 0;"><span style="background-color:#969696; padding: 3px 10px;">&nbsp;</span> No clasificada</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Guardar mapa
    output_file = 'sogamoso_graph_interactive.html'
    m.save(output_file)
    print(f"✓ Mapa interactivo guardado en: {output_file}")
    
    return output_file

def create_speed_heatmap(edges):
    """
    Crea un mapa de calor de velocidades
    """
    print("\nCreando mapa de calor de velocidades...")
    
    if 'speed_kph' not in edges.columns:
        print("❌ No hay datos de velocidad disponibles")
        return
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    edges.plot(ax=ax, column='speed_kph', linewidth=2, 
              cmap='RdYlGn', legend=True, alpha=0.8,
              legend_kwds={'label': 'Velocidad (km/h)', 'shrink': 0.8})
    
    ax.set_title('Mapa de Calor - Velocidades en Sogamoso', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'sogamoso_speed_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Mapa de calor guardado en: {output_file}")
    
    plt.show()

def print_statistics(nodes, edges):
    """
    Imprime estadísticas del grafo
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL GRAFO")
    print("="*60)
    
    print(f"\n📍 Nodos: {len(nodes)}")
    print(f"🛣️  Aristas: {len(edges)}")
    
    if 'length' in edges.columns:
        print(f"\n📏 Distancia total: {edges['length'].sum()/1000:.2f} km")
        print(f"   Longitud promedio: {edges['length'].mean():.2f} m")
    
    if 'speed_kph' in edges.columns:
        print(f"\n⚡ Velocidad promedio: {edges['speed_kph'].mean():.2f} km/h")
        print(f"   Velocidad máxima: {edges['speed_kph'].max():.2f} km/h")
    
    if 'highway' in edges.columns:
        print(f"\n🚦 Tipos de vías:")
        highway_counts = edges['highway'].value_counts().head(5)
        for highway_type, count in highway_counts.items():
            print(f"   - {highway_type}: {count}")
    
    if 'direction' in edges.columns:
        oneway_count = (edges['direction'] == 'one-way').sum()
        bidirectional_count = (edges['direction'] == 'bidirectional').sum()
        print(f"\n➡️  Direccionalidad:")
        print(f"   - Un solo sentido: {oneway_count}")
        print(f"   - Bidireccionales: {bidirectional_count}")
    
    print("="*60)

def main():
    """
    Función principal
    """
    print("="*60)
    print("VISUALIZACIÓN DEL GRAFO DE SOGAMOSO")
    print("="*60)
    
    # Cargar datos
    nodes, edges = load_graph_from_geojson()
    
    if nodes is None or edges is None:
        print("\n❌ No se pudieron cargar los datos. Asegúrate de que los archivos")
        print("   nodes.geojson y edges.geojson existen en el directorio actual.")
        return
    
    # Imprimir estadísticas
    print_statistics(nodes, edges)
    
    # Preguntar qué visualizaciones crear
    print("\n¿Qué visualizaciones deseas crear?")
    print("1. Visualización estática (matplotlib)")
    print("2. Mapa interactivo (folium)")
    print("3. Mapa de calor de velocidades")
    print("4. Todas las anteriores")
    
    choice = input("\nSelecciona una opción (1-4) [Por defecto: 4]: ").strip()
    
    if not choice:
        choice = '4'
    
    if choice in ['1', '4']:
        create_static_visualization(nodes, edges)
    
    if choice in ['2', '4']:
        html_file = create_interactive_map(nodes, edges)
        
        # Preguntar si desea abrir el mapa
        open_map = input("\n¿Deseas abrir el mapa interactivo en el navegador? (s/n) [s]: ").strip().lower()
        if open_map != 'n':
            print(f"\nAbriendo {html_file} en el navegador...")
            webbrowser.open('file://' + os.path.realpath(html_file))
    
    if choice in ['3', '4']:
        create_speed_heatmap(edges)
    
    print("\n✅ Visualización completada!")
    print("\nArchivos generados:")
    if choice in ['1', '4']:
        print("  - sogamoso_graph_static.png")
    if choice in ['2', '4']:
        print("  - sogamoso_graph_interactive.html")
    if choice in ['3', '4']:
        print("  - sogamoso_speed_heatmap.png")

if __name__ == "__main__":
    main()
