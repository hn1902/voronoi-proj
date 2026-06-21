"""
Monte Carlo Tree Search (MCTS) AI for Voronoi Connect 4
Implements MCTS with heuristic-guided rollouts for strategic gameplay.
"""

import copy
import math
import random
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import time


class GameStateSnapshot:
    """
    Lightweight immutable game state for AI simulation.
    Independent from actual game UI/objects.
    """
    
    def __init__(self, points: List[Dict], edges: List[Dict], 
                 claimed_edges: Dict[str, int] = None,
                 current_player: int = 1,
                 player1_score: int = 0,
                 player2_score: int = 0):
        self.points = copy.deepcopy(points)
        self.edges = copy.deepcopy(edges)
        self.claimed_edges = copy.deepcopy(claimed_edges) if claimed_edges else {}
        self.current_player = current_player
        self.player1_score = player1_score
        self.player2_score = player2_score
        
        # Precompute adjacency for performance
        self._adjacency_map = None
        self._polygon_edges = None
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
    
    def clone(self) -> 'GameStateSnapshot':
        """Create deep copy of state for simulation."""
        return GameStateSnapshot(
            self.points,
            self.edges,
            self.claimed_edges,
            self.current_player,
            self.player1_score,
            self.player2_score
        )
    
    def get_valid_actions(self) -> List[int]:
        """Returns indices of unclaimed edges."""
        return [i for i, edge in enumerate(self.edges) 
                if edge['id'] not in self.claimed_edges]
    
    def apply_action(self, action: int) -> None:
        """
        Apply edge selection to state.
        Updates: edge ownership, score, polygon completion, turn switching.
        """
        if action < 0 or action >= len(self.edges):
            raise ValueError(f"Invalid action: {action}")
        
        edge = self.edges[action]
        if edge['id'] in self.claimed_edges:
            raise ValueError(f"Edge {edge['id']} already claimed")
        
        # Claim edge
        self.claimed_edges[edge['id']] = self.current_player
        
        # Calculate score change
        score_change = self._calculate_score_change(action)
        
        # Update player score
        if self.current_player == 1:
            self.player1_score += score_change
        else:
            self.player2_score += score_change
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1 <-> 2
        
        # Invalidate caches
        self._adjacency_map = None
        self._polygon_edges = None
    
    def _calculate_score_change(self, action: int) -> int:
        """Calculate score change for claiming an edge.
        
        Canonical scoring rule (matches ai-game.js):
        - Non-polygon edges: 1 point each
        - Polygon edges: 4 points each
        
        Since this is incremental (called once per move), when a polygon
        closes we must retroactively credit ALL edges in it. Each of the
        k edges in the polygon was previously worth 1pt and should now be
        worth 4pt, so the bonus is +3 per edge = 3*k total.
        """
        edge = self.edges[action]
        player = self.claimed_edges[edge['id']]
        
        # Base score: +1 for claiming edge
        score = 1
        
        # Check if this edge completes any new polygons
        polygons = self._find_polygons_for_player(player)
        
        for polygon in polygons:
            polygon_edge_ids = self._get_polygon_edge_ids(polygon)
            if edge['id'] in polygon_edge_ids:
                # Retroactive bonus: upgrade ALL k edges from 1pt -> 4pt
                k = len(polygon_edge_ids)
                score += 3 * k
        
        return score
    
    def _get_polygon_edge_ids(self, polygon: List[str]) -> Set[str]:
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
    
    def _get_adjacency_map(self) -> Dict[str, List[str]]:
        """Build vertex adjacency map from claimed edges."""
        if self._adjacency_map is not None:
            return self._adjacency_map
        
        adjacency = defaultdict(list)
        
        for edge_id, player in self.claimed_edges.items():
            edge = self._edge_map.get(edge_id)
            if not edge:
                continue
            v1 = f"{edge['x1']},{edge['y1']}"
            v2 = f"{edge['x2']},{edge['y2']}"
            
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        self._adjacency_map = dict(adjacency)
        return self._adjacency_map
    
    def _find_polygons_for_player(self, player: int) -> List[List[str]]:
        """Find all polygons formed by a player's edges."""
        player_edges = [eid for eid, pid in self.claimed_edges.items() if pid == player]
        
        if len(player_edges) < 3:
            return []
        
        # Build adjacency map for this player
        adjacency = defaultdict(list)
        
        for edge_id in player_edges:
            edge = self._edge_map.get(edge_id)
            if not edge:
                continue
            v1 = f"{edge['x1']},{edge['y1']}"
            v2 = f"{edge['x2']},{edge['y2']}"
            
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
    
    def is_terminal(self) -> bool:
        """Returns True if all edges are selected."""
        return len(self.claimed_edges) == len(self.edges)
    
    def get_winner(self) -> int:
        """
        Returns:
            1 -> player1 wins
            2 -> player2 wins  
            0 -> draw
        """
        if not self.is_terminal():
            return -1  # Game not finished
        
        if self.player1_score > self.player2_score:
            return 1
        elif self.player2_score > self.player1_score:
            return 2
        else:
            return 0
    
    def get_score(self, player: int) -> int:
        """Get score for a specific player."""
        if player == 1:
            return self.player1_score
        elif player == 2:
            return self.player2_score
        return 0
    
    def get_edge_by_id(self, edge_id: str) -> Optional[Dict]:
        """Get edge by its ID."""
        for edge in self.edges:
            if edge['id'] == edge_id:
                return edge
        return None


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state: GameStateSnapshot, parent: 'MCTSNode' = None, 
                 action_taken: int = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: List['MCTSNode'] = []
        self.untried_actions: List[int] = state.get_valid_actions()
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.current_player = state.current_player
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        UCB = (total_reward / visits) + c * sqrt(log(parent_visits) / child_visits)
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # Unvisited nodes get infinite exploration bonus
                ucb_score = float('inf')
            else:
                exploitation = child.total_reward / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self) -> 'MCTSNode':
        """Expand node by trying one untried action."""
        if not self.untried_actions:
            raise ValueError("No untried actions available")
        
        action = self.untried_actions.pop()
        new_state = self.state.clone()
        new_state.apply_action(action)
        
        child = MCTSNode(new_state, parent=self, action_taken=action)
        self.children.append(child)
        
        return child
    
    def update(self, reward: float) -> None:
        """Update node statistics with rollout reward."""
        self.visits += 1
        self.total_reward += reward
    
    def is_terminal(self) -> bool:
        """Check if this node represents terminal state."""
        return self.state.is_terminal()


class HeuristicRolloutPolicy:
    """Heuristic-guided rollout policy for MCTS simulation."""
    
    def __init__(self, ai_player: int = 2):
        self.ai_player = ai_player
    
    def select_move(self, state: GameStateSnapshot) -> int:
        """
        Select move using intelligent heuristics for better gameplay.
        Returns action index.
        """
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return -1
        
        # Score each action with better heuristics
        scored_actions = []
        for action in valid_actions:
            score = self._evaluate_action_smart(state, action)
            scored_actions.append((action, score))
        
        # Sort by score (descending)
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        # Weighted probabilistic selection - better moves get higher probability
        weights = [max(0.1, score) for _, score in scored_actions]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(valid_actions)
        
        # Select based on weights
        r = random.uniform(0, total_weight)
        cumulative = 0
        for action, weight in scored_actions:
            cumulative += weight
            if r <= cumulative:
                return action
        
        return scored_actions[0][0]
    
    def _evaluate_action_smart(self, state: GameStateSnapshot, action: int) -> float:
        """
        Smart evaluation using key heuristics for Voronoi game.
        """
        score = 1.0  # Base score
        current_player = state.current_player
        opponent = 3 - current_player
        
        # HEURISTIC 1: Check if move completes polygon (HIGHEST PRIORITY)
        if self._completes_polygon_fast(state, action, current_player):
            score += 1000.0  # Very high priority
        
        # HEURISTIC 2: Avoid moves that let opponent complete polygon next turn
        temp_state = state.clone()
        temp_state.apply_action(action)
        opponent_actions = temp_state.get_valid_actions()
        
        for opp_action in opponent_actions:
            if self._completes_polygon_fast(temp_state, opp_action, opponent):
                score -= 500.0  # Strongly avoid
                break
        
        # HEURISTIC 3: Prefer moves that create potential for future polygons
        connected_edges = self._count_connected_edges_fast(state, action)
        score += connected_edges * 10.0  # Higher bonus for connectivity
        
        # HEURISTIC 4: Prefer edges that create more opportunities (degree analysis)
        edge = state.edges[action]
        v1 = f"{edge['x1']},{edge['y1']}"
        v2 = f"{edge['x2']},{edge['y2']}"
        
        # Count unclaimed edges connected to this edge
        unclaimed_neighbors = 0
        for other_edge in state.edges:
            if other_edge['id'] not in state.claimed_edges:
                other_v1 = f"{other_edge['x1']},{other_edge['y1']}"
                other_v2 = f"{other_edge['x2']},{other_edge['y2']}"
                if v1 in (other_v1, other_v2) or v2 in (other_v1, other_v2):
                    unclaimed_neighbors += 1
        
        score += unclaimed_neighbors * 5.0  # Bonus for creating future options
        
        # HEURISTIC 5: Small random factor for exploration
        score += random.random() * 3.0
        
        return max(score, 0.1)  # Ensure positive weight
    
    def _completes_polygon_fast(self, state: GameStateSnapshot, action: int, player: int) -> bool:
        """
        Fast polygon completion check without expensive DFS.
        Uses simple heuristics to detect likely polygon completion.
        """
        temp_state = state.clone()
        temp_state.apply_action(action)
        
        # Quick check: if player has very few edges, unlikely to complete polygon
        player_edges = [eid for eid, pid in temp_state.claimed_edges.items() if pid == player]
        if len(player_edges) < 3:
            return False
        
        # Use the expensive check only when necessary
        polygons = temp_state._find_polygons_for_player(player)
        
        if not polygons:
            return False
        
        # Check if the newly claimed edge is part of any polygon
        edge = temp_state.edges[action]
        for polygon in polygons:
            polygon_edges = temp_state._get_polygon_edge_ids(polygon)
            if edge['id'] in polygon_edges:
                return True
        
        return False
    
    def _evaluate_action(self, state: GameStateSnapshot, action: int) -> float:
        """
        Evaluate action using heuristics.
        Higher score = better move.
        """
        score = 1.0  # Base score
        
        edge = state.edges[action]
        current_player = state.current_player
        opponent = 3 - current_player
        
        # HEURISTIC 1: Immediate polygon completion (HIGHEST PRIORITY)
        if self._completes_polygon(state, action, current_player):
            score += 100.0  # Very high priority
        
        # HEURISTIC 2: Avoid moves that let opponent complete polygon
        # Simulate opponent's next turn
        temp_state = state.clone()
        temp_state.apply_action(action)
        opponent_actions = temp_state.get_valid_actions()
        
        opponent_can_complete = False
        for opp_action in opponent_actions:
            if self._completes_polygon(temp_state, opp_action, opponent):
                opponent_can_complete = True
                break
        
        if opponent_can_complete:
            score -= 50.0  # Avoid this move
        
        # HEURISTIC 3: Prefer moves connected to already-owned edges
        connected_edges = self._count_connected_edges(state, action, current_player)
        score += connected_edges * 5.0
        
        # HEURISTIC 4: Prefer moves that increase polygon opportunities
        future_potential = self._estimate_polygon_potential(state, action, current_player)
        score += future_potential * 3.0
        
        # HEURISTIC 5: Random factor (lowest priority)
        score += random.random() * 2.0
        
        return max(score, 0.1)  # Ensure positive weight
    
    def _completes_polygon(self, state: GameStateSnapshot, action: int, player: int) -> bool:
        """Check if claiming this edge completes a polygon."""
        temp_state = state.clone()
        temp_state.apply_action(action)
        
        # Get polygons for this player after the move
        polygons = temp_state._find_polygons_for_player(player)
        
        if not polygons:
            return False
        
        # Check if the newly claimed edge is part of any polygon
        edge = temp_state.edges[action]
        for polygon in polygons:
            polygon_edges = temp_state._get_polygon_edge_ids(polygon)
            if edge['id'] in polygon_edges:
                return True
        
        return False
    
    def _count_connected_edges_fast(self, state: GameStateSnapshot, action: int) -> int:
        """Fast version that doesn't require player parameter - for rollout speed."""
        edge = state.edges[action]
        current_player = state.current_player
        return self._count_connected_edges(state, action, current_player)
    
    def _count_connected_edges(self, state: GameStateSnapshot, action: int, player: int) -> int:
        """Count how many of player's edges connect to this edge."""
        edge = state.edges[action]
        
        # Get vertices of this edge
        try:
            v1 = f"{edge['x1']},{edge['y1']}"
            v2 = f"{edge['x2']},{edge['y2']}"
        except KeyError:
            return 0
        
        # Count player's edges that share a vertex
        connected = 0
        player_edges = [eid for eid, pid in state.claimed_edges.items() if pid == player]
        
        for edge_id in player_edges:
            try:
                start, end = edge_id.split('-')
                edge_v1 = start.strip()
                edge_v2 = end.strip()
                
                if v1 in (edge_v1, edge_v2) or v2 in (edge_v1, edge_v2):
                    connected += 1
            except ValueError:
                continue
        
        return connected
    
    def _estimate_polygon_potential(self, state: GameStateSnapshot, action: int, player: int) -> float:
        """Estimate potential for future polygon formation."""
        temp_state = state.clone()
        temp_state.apply_action(action)
        
        # Count connected components in player's graph
        player_edges = [eid for eid, pid in temp_state.claimed_edges.items() if pid == player]
        
        if len(player_edges) < 2:
            return 0.0
        
        # Build adjacency
        adjacency = defaultdict(list)
        for edge_id in player_edges:
            try:
                start, end = edge_id.split('-')
                v1 = start.strip()
                v2 = end.strip()
                adjacency[v1].append(v2)
                adjacency[v2].append(v1)
            except ValueError:
                continue
        
        # Count vertices with multiple connections (potential polygon vertices)
        potential_vertices = sum(1 for v in adjacency if len(adjacency[v]) >= 2)
        
        return potential_vertices * 0.5


class MCTSAI:
    """MCTS-based AI player for Voronoi Connect 4."""
    
    def __init__(self, player_number: int = 2, simulations: int = 500, 
                 exploration_constant: float = 1.4, debug: bool = False):
        self.player_number = player_number
        self.simulations = simulations
        self.c = exploration_constant
        self.debug = debug
        self.rollout_policy = HeuristicRolloutPolicy(player_number)
        
        # Statistics for debugging
        self.last_search_stats = {}
    
    def get_best_move(self, game_state: GameStateSnapshot) -> int:
        """
        Get best move using MCTS.
        
        Args:
            game_state: Current game state snapshot
            
        Returns:
            Action index (edge to claim)
        """
        valid_actions = game_state.get_valid_actions()
        
        if not valid_actions:
            return -1
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Create root node
        root = MCTSNode(game_state.clone())
        
        start_time = time.time()
        
        # Run MCTS simulations
        for i in range(self.simulations):
            # 1. SELECTION
            node = self._select(root)
            
            # 2. EXPANSION
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # 3. SIMULATION (ROLLOUT)
            reward = self._rollout(node.state)
            
            # 4. BACKPROPAGATION
            self._backpropagate(node, reward)
        
        # Select best action
        best_child = self._select_best_action(root)
        
        if self.debug:
            elapsed = time.time() - start_time
            self.last_search_stats = {
                'simulations': self.simulations,
                'time_seconds': elapsed,
                'selected_move': best_child.action_taken if best_child else -1,
                'root_visits': root.visits,
                'best_visits': best_child.visits if best_child else 0,
                'best_reward': best_child.total_reward / best_child.visits if best_child and best_child.visits > 0 else 0
            }
            self._print_debug_info()
        
        return best_child.action_taken if best_child else valid_actions[0]
    
    def get_top_moves(self, game_state: GameStateSnapshot, top_n: int = 5) -> List[Dict]:
        """
        Get top N best moves using MCTS with rankings.
        
        Args:
            game_state: Current game state snapshot
            top_n: Number of top moves to return
            
        Returns:
            List of dictionaries with move info:
            [
                {
                    'action': edge_index,
                    'edge_id': edge_id,
                    'visits': visit_count,
                    'confidence': confidence_percentage,
                    'expected_reward': average_reward,
                    'rank': 1, 2, 3, etc.
                },
                ...
            ]
        """
        valid_actions = game_state.get_valid_actions()
        
        if not valid_actions:
            return []
        
        if len(valid_actions) == 1:
            edge = game_state.edges[valid_actions[0]]
            return [{
                'action': valid_actions[0],
                'edge_id': edge['id'],
                'visits': self.simulations,
                'confidence': 100.0,
                'expected_reward': 0.0,
                'rank': 1
            }]
        
        # Create root node
        root = MCTSNode(game_state.clone())
        
        import time
        start_time = time.time()
        # Time limit scales with simulation count (0.1s per simulation, min 5s, max 300s)
        max_time = max(5, min(300, self.simulations * 0.1))
        
        print(f"Starting MCTS with max {self.simulations} simulations, time limit {max_time:.1f}s")
        
        completed_sims = 0
        
        # Run MCTS simulations
        for i in range(self.simulations):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_time:
                print(f"MCTS time limit reached after {i} simulations ({elapsed:.2f}s)")
                break
            
            # Progress logging every 100 simulations
            if self.debug and i % 100 == 0:
                print(f"MCTS Progress: {i}/{self.simulations} simulations, {elapsed:.2f}s elapsed")
            
            try:
                # 1. SELECTION
                node = self._select(root)
                
                # 2. EXPANSION
                if not node.is_terminal() and not node.is_fully_expanded():
                    node = node.expand()
                
                # 3. SIMULATION (ROLLOUT)
                reward = self._rollout(node.state)
                
                # 4. BACKPROPAGATION
                self._backpropagate(node, reward)
                completed_sims += 1
                
            except Exception as e:
                print(f"Error in simulation {i}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"MCTS Complete: {completed_sims}/{self.simulations} simulations in {total_time:.2f}s")
        
        # Rank all children by visit count
        ranked_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        
        # Build results
        results = []
        for rank, child in enumerate(ranked_children[:top_n], 1):
            edge = game_state.edges[child.action_taken]
            confidence = (child.visits / root.visits) * 100 if root.visits > 0 else 0
            expected_reward = child.total_reward / child.visits if child.visits > 0 else 0
            
            results.append({
                'action': child.action_taken,
                'edge_id': edge['id'],
                'visits': child.visits,
                'confidence': round(confidence, 1),
                'expected_reward': round(expected_reward, 3),
                'rank': rank
            })
        
        return results
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB until leaf."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.c)
        return node
    
    def _rollout(self, state: GameStateSnapshot, max_steps: int = 100) -> float:
        """
        Simulation phase: play out game using heuristic policy.
        Returns reward from AI player's perspective.
        """
        import time
        start_time = time.time()
        
        sim_state = state.clone()
        step_count = 0
        
        while not sim_state.is_terminal() and step_count < max_steps:
            action = self.rollout_policy.select_move(sim_state)
            if action == -1:
                print(f"Rollout: No valid actions at step {step_count}, terminating early")
                break
            sim_state.apply_action(action)
            step_count += 1
            
            # Safety check - if rollout takes too long, abort
            if time.time() - start_time > 0.5:  # 500ms max per rollout
                print(f"Rollout timeout at step {step_count}")
                break
        
        # Calculate reward
        winner = sim_state.get_winner()
        
        if winner == self.player_number:
            reward = 1.0  # Win
        elif winner == 0:
            reward = 0.0  # Draw
        else:
            reward = -1.0  # Loss
        
        # Optional: Add score differential
        ai_score = sim_state.get_score(self.player_number)
        opponent = 3 - self.player_number
        opponent_score = sim_state.get_score(opponent)
        
        max_possible_score = len(sim_state.edges) * 4  # Rough upper bound
        score_diff = (ai_score - opponent_score) / max_possible_score
        reward += score_diff * 0.1  # Small bonus for score difference
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagation phase: update statistics up the tree."""
        current = node
        
        while current is not None:
            # Flip reward based on player perspective
            if current.current_player != self.player_number:
                current_reward = -reward  # Opponent's turn
            else:
                current_reward = reward
            
            current.update(current_reward)
            current = current.parent
    
    def _select_best_action(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Select best action based on visit counts."""
        if not root.children:
            return None
        
        # Select child with highest visit count (most robust)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child
    
    def _print_debug_info(self) -> None:
        """Print debug information about last search."""
        print("\n" + "="*50)
        print("MCTS SEARCH STATISTICS")
        print("="*50)
        for key, value in self.last_search_stats.items():
            print(f"{key}: {value}")
        print("="*50 + "\n")
    
    def set_simulations(self, simulations: int) -> None:
        """Adjust number of simulations (for difficulty tuning)."""
        self.simulations = simulations


def create_state_from_game(game_obj) -> GameStateSnapshot:
    """
    Create a GameStateSnapshot from the current game object.
    This bridges the actual game with the AI simulation.
    """
    return GameStateSnapshot(
        points=game_obj.points,
        edges=game_obj.edges,
        claimed_edges={k: v for k, v in game_obj.claimed_edges.items()},
        current_player=game_obj.current_player,
        player1_score=game_obj.player1_score,
        player2_score=game_obj.player2_score
    )
