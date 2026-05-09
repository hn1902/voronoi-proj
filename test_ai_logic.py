#!/usr/bin/env python3
"""
Test the AI logic without the web interface
"""

import numpy as np
import random

def test_ai_strategy():
    """Test the AI strategy logic"""
    print("Testing AI strategy...")
    
    # Simulate a simple game state
    edges = [
        {'id': '100,100-200,100', 'x1': 100, 'y1': 100, 'x2': 200, 'y2': 100, 'claimed': False, 'player': None},
        {'id': '200,100-200,200', 'x1': 200, 'y1': 100, 'x2': 200, 'y2': 200, 'claimed': False, 'player': None},
        {'id': '200,200-100,200', 'x1': 200, 'y1': 200, 'x2': 100, 'y2': 200, 'claimed': False, 'player': None},
        {'id': '100,200-100,100', 'x1': 100, 'y1': 200, 'x2': 100, 'y2': 100, 'claimed': False, 'player': None},
        {'id': '150,150-250,150', 'x1': 150, 'y1': 150, 'x2': 250, 'y2': 150, 'claimed': False, 'player': None},
    ]
    
    # Simulate player 1 claiming some edges
    edges[0]['claimed'] = True
    edges[0]['player'] = 1
    edges[1]['claimed'] = True
    edges[1]['player'] = 1
    
    # Test AI edge selection
    available_edges = [edge for edge in edges if not edge['claimed']]
    
    def select_best_edge(available_edges, claimed_edges):
        """Simple AI strategy"""
        player1_edges = [edge_id for edge_id, player in claimed_edges.items() if player == 1]
        
        scored_edges = []
        for edge in available_edges:
            score = random.random()
            
            # Check if edge connects to player 1's edges (defensive)
            for player_edge_id in player1_edges:
                if edges_share_vertex(edge['id'], player_edge_id, edges):
                    score += 2
            
            scored_edges.append({'edge': edge, 'score': score})
        
        scored_edges.sort(key=lambda x: x['score'], reverse=True)
        return scored_edges[0]['edge'] if scored_edges else None
    
    def edges_share_vertex(edge_id1, edge_id2, all_edges):
        """Check if two edges share a vertex"""
        edge1 = next((e for e in all_edges if e['id'] == edge_id1), None)
        edge2 = next((e for e in all_edges if e['id'] == edge_id2), None)
        
        if not edge1 or not edge2:
            return False
        
        vertices1 = [f"{edge1['x1']},{edge1['y1']}", f"{edge1['x2']},{edge1['y2']}"]
        vertices2 = [f"{edge2['x1']},{edge2['y1']}", f"{edge2['x2']},{edge2['y2']}"]
        
        return any(v in vertices2 for v in vertices1)
    
    # Test AI selection
    claimed_edges = {edges[0]['id']: 1, edges[1]['id']: 1}
    selected_edge = select_best_edge(available_edges, claimed_edges)
    
    print(f"Available edges: {len(available_edges)}")
    print(f"AI selected edge: {selected_edge['id'] if selected_edge else 'None'}")
    print("AI strategy test completed successfully!")
    
    return True

if __name__ == "__main__":
    test_ai_strategy()
