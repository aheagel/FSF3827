import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import pyomo.environ as pyo
import contextily as ctx


def extract_route_from_x(model: pyo.Model, start_node: int = 0) -> List[int]: # CLAUDE 4.5 not bad did it one take some small errors that i fixed
    """
    Extract the tour route by following the x[i,j] binary variables.
    Works with any TSP formulation that uses x variables for arc selection.
    
    Args:
        model: Solved Pyomo model with x variables (binary arc selection)
        start_node: Starting node (depot), default is 0
    
    Returns:
        List of node indices in visit order (including return to start)
    """
    # Build adjacency from x variables where x[i,j] ≈ 1
    arcs = {}
    for (i, j) in model.x:
        if pyo.value(model.x[i, j]) > 0.5:  # Binary variable is "on"
            arcs[i] = j
    
    # Follow the path starting from start_node
    route = [start_node]
    current = start_node
    
    while True:
        next_node = arcs.get(current)
        if next_node is None:
            raise ValueError(f"Route broken at node {current}. Check solution feasibility.")
        if next_node == start_node:
            route.append(start_node)  # Close the tour
            break
        route.append(next_node)
        current = next_node
    
    return route


def plot_tsp_solution(
    data: Dict[str, Any],
    model: pyo.Model,
    start_node: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    show_labels: bool = True,
    title: str = None
) -> None:
    """
    Plot the TSP solution with route visualization.
    For GEO edge_weight_type, also creates an interactive map overlay.
    
    Args:
        data: Dictionary from parse_tsplib containing 'coords' and other metadata
        model: Solved Pyomo TSP model with x variables
        start_node: Depot/starting node index (default 0)
        figsize: Figure size tuple (width, height)
        show_labels: Whether to show node labels on the plot
        title: Custom plot title (defaults to problem name from data)
    """
    coords = data['coords']
    route = extract_route_from_x(model, start_node=start_node)
    edge_weight_type = data.get('edge_weight_type', '').upper()
    
    # If GEO type, use specialized map overlay plot
    if edge_weight_type == 'GEO':
        plot_tsp_with_map_overlay(data, model, route, start_node, figsize, show_labels, title)
        return
    
    # Extract route coordinates
    route_coords = coords[route]
    
    # Calculate total distance
    total_distance = pyo.value(model.obj)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='lightblue', s=100, zorder=2, edgecolors='black', linewidths=1)
    
    # Highlight depot (use start_node instead of hardcoded 0)
    ax.scatter(coords[0, 0], coords[0, 1], 
               c='red', s=200, zorder=3, marker='s', edgecolors='darkred', linewidths=2, label='Depot')
    
    # Plot route
    ax.plot(route_coords[:, 0], route_coords[:, 1], 
            'b-', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Add arrows to show direction
    for i in range(len(route) - 1):
        x1, y1 = coords[route[i]]
        x2, y2 = coords[route[i + 1]]
        dx, dy = x2 - x1, y2 - y1
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1, color='blue', alpha=0.4))
    
    # Add node labels
    if show_labels:
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i+start_node), (x, y), fontsize=8, ha='center', va='center')
    
    # Set title
    if title is None:
        title = f"TSP Solution: {data.get('name', 'Unknown')}"
    ax.set_title(f"{title}\nTotal Distance: {total_distance:.2f}", fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()


def print_route_summary(
    data: Dict[str, Any],
    model: pyo.Model,
    start_node: int = 0
) -> None:
    """
    Print a summary of the TSP solution including route order and total distance.
    
    Args:
        data: Dictionary from parse_tsplib
        model: Solved Pyomo TSP model with x variables
        start_node: Depot/starting node index (default 0)
    """
    route = extract_route_from_x(model, start_node=start_node)
    total_distance = pyo.value(model.obj)
    
    print("=" * 60)
    print(f"TSP Solution Summary: {data.get('name', 'Unknown')}")
    print("=" * 60)
    print(f"Problem Dimension: {data.get('dimension', len(data['coords']))} nodes")
    print(f"Total Distance: {total_distance:.2f}")
    print(f"\nRoute (visiting order):")
    print(" -> ".join(map(str, route)))
    print("=" * 60)


def _convert_tsplib_geo_to_latlon(coords: np.ndarray) -> np.ndarray:
    """
    Convert TSPLIB GEO coordinates to standard lat/lon in decimal degrees.
    
    TSPLIB GEO format: coordinates are given where the integer part is degrees
    and the fractional part represents minutes in a special encoding.
    
    Args:
        coords: np.ndarray of shape [n,2] in TSPLIB GEO format
    
    Returns:
        np.ndarray of shape [n,2] with [latitude, longitude] in decimal degrees
    """
    def geo_to_decimal(coord):
        """Convert TSPLIB GEO coordinate to decimal degrees."""
        deg = int(coord)
        min_part = coord - deg
        # TSPLIB uses: degrees + (5 * fractional_part / 3) as minutes
        minutes = 5.0 * min_part / 3.0 * 60.0
        return deg + minutes / 60.0
    
    latlon = np.zeros_like(coords)
    latlon[:, 0] = np.array([geo_to_decimal(lat) for lat in coords[:, 0]])  # Latitude
    latlon[:, 1] = np.array([geo_to_decimal(lon) for lon in coords[:, 1]])  # Longitude
    
    return latlon


def plot_tsp_with_map_overlay(
    data: Dict[str, Any],
    model: pyo.Model,
    route: List[int],
    start_node: int = 0,
    figsize: Tuple[int, int] = (12, 10),
    show_labels: bool = True,
    title: str = None
) -> None:
    """
    Plot TSP solution with a map overlay for GEO coordinates.
    
    Args:
        data: Dictionary from parse_tsplib with GEO coordinates
        model: Solved Pyomo TSP model with x variables
        route: Pre-extracted route
        start_node: Depot/starting node index (default 0)
        figsize: Figure size tuple (width, height)
        show_labels: Whether to show node labels on the plot
        title: Custom plot title
    """
    coords = data['coords']
    
    # Convert TSPLIB GEO coordinates to decimal lat/lon
    latlon = _convert_tsplib_geo_to_latlon(coords)
    
    # Convert lat/lon to Web Mercator (EPSG:3857) for contextily
    # Web Mercator formula: x = lon, y = lat (in degrees), then convert to meters
    from math import pi, log, tan
    
    def latlon_to_web_mercator(lat, lon):
        """Convert lat/lon to Web Mercator coordinates (EPSG:3857)."""
        x = lon * 20037508.34 / 180.0
        y = log(tan((90.0 + lat) * pi / 360.0)) / (pi / 180.0)
        y = y * 20037508.34 / 180.0
        return x, y
    
    # Convert all coordinates to Web Mercator
    web_merc_coords = np.array([latlon_to_web_mercator(lat, lon) for lat, lon in latlon])
    route_coords = web_merc_coords[route]
    
    # Calculate total distance
    total_distance = pyo.value(model.obj)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot route
    ax.plot(route_coords[:, 0], route_coords[:, 1], 
            'b-', linewidth=2.5, alpha=0.7, zorder=2, label='Route')
    
    # Plot all nodes
    ax.scatter(web_merc_coords[:, 0], web_merc_coords[:, 1], 
               c='lightblue', s=150, zorder=3, edgecolors='black', linewidths=1.5)
    
    # Highlight depot
    ax.scatter(web_merc_coords[start_node, 0], web_merc_coords[start_node, 1], 
               c='red', s=300, zorder=4, marker='s', edgecolors='darkred', 
               linewidths=2, label='Depot')
    
    # Add node labels
    if show_labels:
        for i, (x, y) in enumerate(web_merc_coords):
            ax.annotate(str(i), (x, y), fontsize=9, ha='center', va='center', 
                       fontweight='bold', color='white',
                       bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.6))
    
    # Add map overlay
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.6)
    
    # Set title
    if title is None:
        title = f"TSP Solution: {data.get('name', 'Unknown')}"
    ax.set_title(f"{title}\nTotal Distance: {total_distance:.2f} km", 
                fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
