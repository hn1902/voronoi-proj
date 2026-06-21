#!/usr/bin/env python3
"""
Simplified RL Environment for fixed Voronoi board
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import copy
from fixed_board_config import get_fixed_board, NUM_EDGES

class FixedVoronoiRLEnvironment:
    """
    Simplified RL Environment for fixed Voronoi Connect 4 board
    """
    
    def __init__(self):
        """Initialize with fixed board configuration"""
        board = get_fixed_board()
        self.points = board['points']
        self.edges = board['edges']
        self.width = board['width']
        self.height = board['height']
        
        self._edge_map = {e['id']: e for e in self.edges}
        
        # Build coordinate-to-edge-id mapping
        self._coord_to_edge_id = {}
        for edge in self.edges:
            x1, y1 = edge['x1'], edge['y1']
            x2, y2 = edge['x2'], edge['y2']
            v1 = f"{x1},{y1}"
            v2 = f"{x2},{y2}"
            coords = sorted([v1, v2])
            coord_key = f"{coords[0]}-{coords[1]}"
            self._coord_to_edge_id[coord_key] = edge['id']
            
        # Game state
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.claimed_edges = {}  # edge_id -> player
        self.current_player = 1
        self.player1_score = 0
        self.player2_score = 0
        self.game_over = False
        self.turn_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state as fixed-size numpy array"""
        # Edge states: 0=unclaimed, 1=claimed by player 1, 2=claimed by player 2
        edge_states = np.zeros(NUM_EDGES, dtype=np.float32)
        
        for i, edge in enumerate(self.edges):
            if edge['id'] in self.claimed_edges:
                edge_states[i] = self.claimed_edges[edge['id']]
        
        # Additional state features
        state_features = np.array([
            self.current_player / 2.0,  # Normalize to [0, 1]
            self.player1_score / 50.0,  # Normalize (max score ~50)
            self.player2_score / 50.0,
            (NUM_EDGES - len(self.claimed_edges)) / NUM_EDGES,  # Remaining edges ratio
            self.turn_count / (NUM_EDGES * 2)  # Turn progress
        ], dtype=np.float32)
        
        # Combine into single state vector
        state = np.concatenate([edge_states, state_features])
        
        return state
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices (unclaimed edges)"""
        valid_actions = []
        for i, edge in enumerate(self.edges):
            if edge['id'] not in self.claimed_edges:
                valid_actions.append(i)
        return valid_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action (claim an edge)
        
        Args:
            action: Index of edge to claim
            
        Returns:
            next_state: New state after action
            reward: Reward for this action
            done: Whether game is over
            info: Additional information
        """
        if self.game_over:
            return self._get_state(), 0, True, {"error": "Game is over"}
        
        # Check if action is valid
        if action < 0 or action >= len(self.edges):
            return self._get_state(), -10, False, {"error": "Invalid action"}
        
        edge = self.edges[action]
        if edge['id'] in self.claimed_edges:
            return self._get_state(), -5, False, {"error": "Edge already claimed"}
        
        # Claim the edge
        self.claimed_edges[edge['id']] = self.current_player
        self.turn_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Update scores
        self._update_scores()
        
        # Check if game is over
        if len(self.claimed_edges) == len(self.edges):
            self.game_over = True
            # Add win/loss reward
            if self.player1_score > self.player2_score:
                reward += 10 if self.current_player == 1 else -10
            elif self.player2_score > self.player1_score:
                reward += 10 if self.current_player == 2 else -10
            else:
                reward += 0  # Draw
        
        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        next_state = self._get_state()
        info = {
            "player1_score": self.player1_score,
            "player2_score": self.player2_score,
            "edges_claimed": len(self.claimed_edges),
            "total_edges": len(self.edges)
        }
        
        return next_state, reward, self.game_over, info
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for claiming an edge.
        
        Canonical scoring rule (matches ai-game.js):
        - Non-polygon edges: 1 point each
        - Polygon edges: 4 points each
        
        Since this is incremental, when a polygon closes we retroactively
        credit ALL k edges in it: bonus = 3 * k (upgrading each from 1pt to 4pt).
        We detect new polygons by comparing polygon count before vs after.
        """
        edge = self.edges[action]
        
        # Base reward for claiming an edge
        reward = 1.0
        
        # Get player's edges INCLUDING the newly claimed one
        player_edges = [eid for eid, player in self.claimed_edges.items() 
                       if player == self.current_player]
        
        # Find polygons after claiming this edge
        polygons_after = self._find_polygons(player_edges)
        
        # Find polygons without this edge (before claiming)
        player_edges_before = [eid for eid in player_edges if eid != edge['id']]
        polygons_before = self._find_polygons(player_edges_before)
        
        # Check for newly completed polygons
        new_polygon_count = len(polygons_after) - len(polygons_before)
        if new_polygon_count > 0:
            # Find which polygons are new (contain the new edge)
            for polygon in polygons_after:
                polygon_edge_ids = self._get_polygon_edge_ids(polygon)
                if edge['id'] in polygon_edge_ids:
                    # Retroactive bonus: upgrade ALL k edges from 1pt -> 4pt
                    k = len(polygon_edge_ids)
                    reward += 3.0 * k
        
        return reward
    
    def _get_polygon_edge_ids(self, polygon):
        """Convert polygon vertices to edge IDs."""
        edge_ids = set()
        for i in range(len(polygon)):
            v1 = polygon[i]
            v2 = polygon[(i + 1) % len(polygon)]
            # Create coordinate key (sorted for consistency)
            coords = sorted([v1, v2])
            coord_key = f"{coords[0]}-{coords[1]}"
            if coord_key in self._coord_to_edge_id:
                edge_ids.add(self._coord_to_edge_id[coord_key])
        return edge_ids
    
    def _update_scores(self):
        """Update player scores based on claimed edges"""
        self.player1_score = self._calculate_player_score(1)
        self.player2_score = self._calculate_player_score(2)
    
    def _calculate_player_score(self, player: int) -> int:
        """Calculate score for a player (full recompute).
        
        Canonical scoring rule (matches ai-game.js):
        - Non-polygon edges: 1 point each
        - Polygon edges: 4 points each
        """
        player_edges = [eid for eid, p in self.claimed_edges.items() if p == player]
        
        # Find all polygons
        polygons = self._find_polygons(player_edges)
        
        # Collect all edges that are part of any polygon
        edges_in_polygons = set()
        for polygon in polygons:
            polygon_edge_ids = self._get_polygon_edge_ids(polygon)
            edges_in_polygons.update(polygon_edge_ids)
        
        # Non-polygon edges: 1 point each, polygon edges: 4 points each
        non_polygon_count = len([e for e in player_edges if e not in edges_in_polygons])
        polygon_edge_count = len(edges_in_polygons)
        
        score = non_polygon_count + polygon_edge_count * 4
        
        return score
    
    def _find_polygons(self, player_edges: List[str]) -> List[List[str]]:
        """Find all polygons formed by player's edges (with deduplication)."""
        if len(player_edges) < 3:
            return []
        
        # Build adjacency map
        adjacency = {}
        for edge_id in player_edges:
            edge = self._edge_map.get(edge_id)
            if not edge:
                continue
            v1 = f"{edge['x1']},{edge['y1']}"
            v2 = f"{edge['x2']},{edge['y2']}"
            
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        # Find cycles using DFS
        polygons = []
        visited = set()
        
        def find_cycles_from_vertex(start_vertex: str) -> List[List[str]]:
            """Find all cycles starting from a vertex."""
            cycles = []
            path = [start_vertex]
            path_set = {start_vertex}
            
            def dfs(current: str):
                for neighbor in adjacency.get(current, []):
                    if neighbor == start_vertex and len(path) >= 3:
                        # Found a cycle
                        cycle = path.copy()
                        cycles.append(cycle)
                    elif neighbor not in path_set:
                        path.append(neighbor)
                        path_set.add(neighbor)
                        dfs(neighbor)
                        path.pop()
                        path_set.remove(neighbor)
            
            dfs(start_vertex)
            return cycles
        
        # Find cycles from each vertex
        for vertex in adjacency:
            if vertex not in visited:
                cycles = find_cycles_from_vertex(vertex)
                for cycle in cycles:
                    # Normalize cycle to avoid duplicates
                    min_idx = cycle.index(min(cycle))
                    normalized = cycle[min_idx:] + cycle[:min_idx]
                    
                    # Check if this polygon is already found
                    is_duplicate = False
                    for existing in polygons:
                        if set(existing) == set(normalized):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        polygons.append(normalized)
                
                visited.add(vertex)
        
        return polygons
    
    def get_state_size(self) -> int:
        """Get the size of the state space"""
        return NUM_EDGES + 5  # Edge states + 5 additional features
    
    def get_action_size(self) -> int:
        """Get the size of the action space"""
        return NUM_EDGES
    
    def render(self):
        """Print current game state"""
        print(f"Current Player: {self.current_player}")
        print(f"Player 1 Score: {self.player1_score}")
        print(f"Player 2 Score: {self.player2_score}")
        print(f"Edges Claimed: {len(self.claimed_edges)}/{len(self.edges)}")
        print(f"Game Over: {self.game_over}")
        
        # Show claimed edges
        print("\nClaimed edges:")
        for edge_id, player in self.claimed_edges.items():
            print(f"  {edge_id}: Player {player}")

def test_fixed_environment():
    """Test the fixed environment"""
    print("Testing Fixed Voronoi RL Environment...")
    
    env = FixedVoronoiRLEnvironment()
    
    print(f"State size: {env.get_state_size()}")
    print(f"Action size: {env.get_action_size()}")
    
    # Test a few steps
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for step in range(5):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        action = valid_actions[0]  # Take first valid action
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}: Action {action}, Reward {reward}, Done {done}")
        print(f"  Scores: P1={info['player1_score']}, P2={info['player2_score']}")
        
        if done:
            break
    
    print("Environment test completed!")

if __name__ == "__main__":
    test_fixed_environment()
