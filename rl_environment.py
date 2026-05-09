import numpy as np
from typing import List, Tuple, Dict, Optional
import copy

class VoronoiRLEnvironment:
    """
    Reinforcement Learning Environment for Voronoi Connect 4
    """
    
    def __init__(self, points: List[Dict], edges: List[Dict]):
        """
        Initialize the RL environment
        
        Args:
            points: List of point dictionaries with x, y, id
            edges: List of edge dictionaries with x1, y1, x2, y2, id
        """
        self.original_points = copy.deepcopy(points)
        self.original_edges = copy.deepcopy(edges)
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.points = copy.deepcopy(self.original_points)
        self.edges = copy.deepcopy(self.original_edges)
        self.claimed_edges = {}  # edge_id -> player
        self.current_player = 1
        self.player1_score = 0
        self.player2_score = 0
        self.game_over = False
        self.turn_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        # Edge states: 0=unclaimed, 1=claimed by player 1, 2=claimed by player 2
        edge_states = []
        for edge in self.edges:
            if edge['id'] in self.claimed_edges:
                edge_states.append(self.claimed_edges[edge['id']])
            else:
                edge_states.append(0)
        
        # Additional state information
        state_features = [
            self.current_player,  # Current player (1 or 2)
            self.player1_score,    # Player 1 score
            self.player2_score,    # Player 2 score
            len(self.edges) - len(self.claimed_edges),  # Remaining edges
            self.turn_count        # Turn number
        ]
        
        # Normalize scores (divide by max possible score)
        max_score = len(self.edges) * 5  # Maximum possible score
        state_features[1] = self.player1_score / max_score
        state_features[2] = self.player2_score / max_score
        
        # Combine edge states and features
        state = np.array(edge_states + state_features, dtype=np.float32)
        
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
        """Calculate reward for claiming an edge"""
        edge = self.edges[action]
        
        # Base reward for claiming an edge
        reward = 1.0
        
        # Check if this edge forms a polygon
        player_edges = [eid for eid, player in self.claimed_edges.items() 
                       if player == self.current_player]
        
        if self._forms_polygon(player_edges, edge['id']):
            reward += 4.0  # Bonus for forming polygon
        
        return reward
    
    def _forms_polygon(self, player_edges: List[str], new_edge_id: str) -> bool:
        """Check if adding this edge forms a polygon"""
        # Simplified polygon detection - check for cycles
        if len(player_edges) < 3:
            return False
        
        # Build adjacency map
        edge_map = {}
        for edge_id in player_edges:
            edge = next(e for e in self.edges if e['id'] == edge_id)
            v1 = f"{edge['x1']},{edge['y1']}"
            v2 = f"{edge['x2']},{edge['y2']}"
            
            if v1 not in edge_map:
                edge_map[v1] = []
            if v2 not in edge_map:
                edge_map[v2] = []
            
            edge_map[v1].append(v2)
            edge_map[v2].append(v1)
        
        # Check for cycles (simplified)
        visited = set()
        
        def dfs(vertex, parent, depth):
            if depth > 10:  # Prevent infinite recursion
                return False
            visited.add(vertex)
            
            for neighbor in edge_map.get(vertex, []):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True  # Found a cycle
                if dfs(neighbor, vertex, depth + 1):
                    return True
            
            return False
        
        # Check from any vertex
        for vertex in edge_map:
            if vertex not in visited:
                if dfs(vertex, None, 0):
                    return True
        
        return False
    
    def _update_scores(self):
        """Update player scores based on claimed edges"""
        self.player1_score = self._calculate_player_score(1)
        self.player2_score = self._calculate_player_score(2)
    
    def _calculate_player_score(self, player: int) -> int:
        """Calculate score for a player"""
        player_edges = [eid for eid, p in self.claimed_edges.items() if p == player]
        
        # Base score: 1 point per edge
        score = len(player_edges)
        
        # Find polygons for bonus points
        polygons = self._find_polygons(player_edges)
        score += len(polygons) * 4  # 4 bonus points per polygon
        
        return score
    
    def _find_polygons(self, player_edges: List[str]) -> List[List[str]]:
        """Find all polygons formed by player's edges"""
        # Simplified polygon detection
        # In a full implementation, this would use proper cycle detection
        return []  # Placeholder
    
    def get_state_size(self) -> int:
        """Get the size of the state space"""
        return len(self.edges) + 5  # Edge states + 5 additional features
    
    def get_action_size(self) -> int:
        """Get the size of the action space"""
        return len(self.edges)
    
    def render(self):
        """Print current game state"""
        print(f"Current Player: {self.current_player}")
        print(f"Player 1 Score: {self.player1_score}")
        print(f"Player 2 Score: {self.player2_score}")
        print(f"Edges Claimed: {len(self.claimed_edges)}/{len(self.edges)}")
        print(f"Game Over: {self.game_over}")
