#!/usr/bin/env python3
"""
Fixed board configuration for simplified RL training
"""

# Fixed point configuration - creates interesting strategic positions
FIXED_POINTS = [
    {'x': 150, 'y': 150, 'id': 0},
    {'x': 650, 'y': 150, 'id': 1},
    {'x': 400, 'y': 300, 'id': 2},
    {'x': 150, 'y': 450, 'id': 3},
    {'x': 650, 'y': 450, 'id': 4},
    {'x': 400, 'y': 150, 'id': 5},
    {'x': 250, 'y': 300, 'id': 6},
    {'x': 550, 'y': 300, 'id': 7},
]

# Pre-computed edges for this configuration (simplified Voronoi edges)
# In practice, you'd compute these using the same algorithm as the frontend
FIXED_EDGES = [
    {'id': 'edge_0', 'x1': 150, 'y1': 150, 'x2': 400, 'y2': 150},
    {'id': 'edge_1', 'x1': 400, 'y1': 150, 'x2': 650, 'y2': 150},
    {'id': 'edge_2', 'x1': 150, 'y1': 150, 'x2': 250, 'y2': 300},
    {'id': 'edge_3', 'x1': 650, 'y1': 150, 'x2': 550, 'y2': 300},
    {'id': 'edge_4', 'x1': 400, 'y1': 150, 'x2': 400, 'y2': 300},
    {'id': 'edge_5', 'x1': 250, 'y1': 300, 'x2': 400, 'y2': 300},
    {'id': 'edge_6', 'x1': 400, 'y1': 300, 'x2': 550, 'y2': 300},
    {'id': 'edge_7', 'x1': 250, 'y1': 300, 'x2': 150, 'y2': 450},
    {'id': 'edge_8', 'x1': 550, 'y1': 300, 'x2': 650, 'y2': 450},
    {'id': 'edge_9', 'x1': 400, 'y1': 300, 'x2': 400, 'y2': 450},
    {'id': 'edge_10', 'x1': 150, 'y1': 450, 'x2': 400, 'y2': 450},
    {'id': 'edge_11', 'x1': 400, 'y1': 450, 'x2': 650, 'y2': 450},
]

# Board configuration constants
BOARD_WIDTH = 800
BOARD_HEIGHT = 600
NUM_POINTS = len(FIXED_POINTS)
NUM_EDGES = len(FIXED_EDGES)

def get_fixed_board():
    """Return the fixed board configuration"""
    return {
        'points': FIXED_POINTS.copy(),
        'edges': FIXED_EDGES.copy(),
        'width': BOARD_WIDTH,
        'height': BOARD_HEIGHT
    }

def print_board_info():
    """Print information about the fixed board"""
    print("Fixed Board Configuration:")
    print(f"  Points: {NUM_POINTS}")
    print(f"  Edges: {NUM_EDGES}")
    print(f"  Board size: {BOARD_WIDTH}x{BOARD_HEIGHT}")
    print("\nPoints:")
    for point in FIXED_POINTS:
        print(f"  Point {point['id']}: ({point['x']}, {point['y']})")
    print("\nEdges:")
    for i, edge in enumerate(FIXED_EDGES):
        print(f"  Edge {i}: ({edge['x1']}, {edge['y1']}) -> ({edge['x2']}, {edge['y2']})")

if __name__ == "__main__":
    print_board_info()
