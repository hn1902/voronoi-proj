"""
AlphaZero-style MCTS for Voronoi Game
Integrates neural network guidance with tree search using PUCT formula.
"""

import math
import time
import random
import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from mcts_ai import GameStateSnapshot, MCTSNode, HeuristicRolloutPolicy
from state_encoder import StateEncoder
from nn_model import VoronoiNet


class AZMCTSNode:
    """
    MCTS Node for AlphaZero-style search.
    Extends base node with neural network policy priors.
    """
    
    def __init__(self, state: GameStateSnapshot, parent: 'AZMCTSNode' = None,
                 action_taken: int = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior  # From neural network policy
        
        self.children: List['AZMCTSNode'] = []
        self.untried_actions: List[int] = state.get_valid_actions()
        
        self.visits: int = 0
        self.total_value: float = 0.0  # Sum of values (not rewards)
        self.current_player = state.current_player
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def value(self) -> float:
        """Mean value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def best_child_puct(self, c_puct: float = 1.5) -> 'AZMCTSNode':
        """
        Select best child using PUCT formula:
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            # Q value: average outcome
            q_value = child.value() if child.visits > 0 else 0.0
            
            # U value: exploration bonus based on prior
            if child.visits == 0:
                u_value = c_puct * child.prior * math.sqrt(self.visits + 1)
            else:
                u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, action: int, prior: float) -> 'AZMCTSNode':
        """Expand node with given action and prior probability."""
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        
        new_state = self.state.clone()
        new_state.apply_action(action)
        
        child = AZMCTSNode(new_state, parent=self, action_taken=action, prior=prior)
        self.children.append(child)
        
        return child
    
    def update(self, value: float) -> None:
        """Update node with a value estimate."""
        self.visits += 1
        self.total_value += value
    
    def update_recursive(self, value: float) -> None:
        """Update this node and all ancestors."""
        self.update(value)
        if self.parent is not None:
            # Flip sign for alternating players (zero-sum)
            self.parent.update_recursive(-value)


class AlphaZeroMCTS:
    """
    AlphaZero-style MCTS with neural network guidance.
    """
    
    def __init__(self, neural_net: VoronoiNet, encoder: StateEncoder,
                 c_puct: float = 1.5, simulations: int = 800,
                 use_value_net: bool = True, use_heuristic_fallback: bool = True,
                 temperature: float = 1.0):
        """
        Args:
            neural_net: Trained VoronoiNet model
            encoder: StateEncoder for converting states
            c_puct: PUCT exploration constant
            simulations: Number of MCTS simulations
            use_value_net: Whether to use NN value instead of rollouts
            use_heuristic_fallback: Whether to use heuristic rollouts as fallback
            temperature: Temperature for action selection (1.0 = normal, 0.0 = argmax)
        """
        self.neural_net = neural_net
        self.encoder = encoder
        self.c_puct = c_puct
        self.simulations = simulations
        self.use_value_net = use_value_net
        self.use_heuristic_fallback = use_heuristic_fallback
        self.temperature = temperature
        
        # Fallback rollout policy
        self.heuristic_policy = HeuristicRolloutPolicy(ai_player=2)
        
        # Statistics
        self.last_search_stats = {}
        self.debug = False
    
    def search(self, game_state: GameStateSnapshot, return_root: bool = False) -> Tuple[Dict[int, float], float]:
        """
        Run MCTS search and return action probabilities.
        
        Args:
            game_state: Current game state
            return_root: If True, also return root node
            
        Returns:
            Tuple of (action_probs, root_value)
            action_probs: Dict mapping action -> probability
        """
        valid_actions = game_state.get_valid_actions()
        
        if not valid_actions:
            if return_root:
                return {}, 0.0, None
            return {}, 0.0
        
        if len(valid_actions) == 1:
            probs = {valid_actions[0]: 1.0}
            if return_root:
                return probs, 0.0, None
            return probs, 0.0
        
        # Create root node
        root = AZMCTSNode(game_state.clone())
        
        # Run simulations
        start_time = time.time()
        
        for i in range(self.simulations):
            # 1. SELECTION (using PUCT)
            node = self._select(root)
            
            # 2. EVALUATION (NN value + policy)
            value = self._evaluate(node)
            
            # 3. BACKPROPAGATION
            node.update_recursive(value)
        
        # Compute action probabilities from visit counts
        action_probs = {}
        total_visits = sum(child.visits for child in root.children)
        
        if total_visits > 0:
            if self.temperature == 0:
                # Argmax
                best_child = max(root.children, key=lambda c: c.visits)
                action_probs = {child.action_taken: 1.0 if child == best_child else 0.0 
                              for child in root.children}
            else:
                # Temperature-scaled visit counts
                visits = torch.tensor([child.visits for child in root.children], dtype=torch.float32)
                # Apply temperature
                if self.temperature != 1.0:
                    visits = visits ** (1.0 / self.temperature)
                
                probs = F.softmax(visits, dim=0)
                
                for child, prob in zip(root.children, probs.tolist()):
                    action_probs[child.action_taken] = prob
        else:
            # Uniform if no visits
            n = len(root.children)
            for child in root.children:
                action_probs[child.action_taken] = 1.0 / n
        
        # Root value (from NN)
        root_value = self._get_nn_value(game_state)
        
        elapsed = time.time() - start_time
        self.last_search_stats = {
            'simulations': self.simulations,
            'time_seconds': elapsed,
            'root_visits': root.visits,
            'children': len(root.children),
            'root_value': root_value
        }
        
        if return_root:
            return action_probs, root_value, root
        
        return action_probs, root_value
    
    def get_best_move(self, game_state: GameStateSnapshot) -> int:
        """
        Get best move (argmax of visit counts).
        
        Args:
            game_state: Current game state
            
        Returns:
            Best action index
        """
        action_probs, _ = self.search(game_state)
        
        if not action_probs:
            valid_actions = game_state.get_valid_actions()
            return valid_actions[0] if valid_actions else -1
        
        return max(action_probs, key=action_probs.get)
    
    def get_top_moves(self, game_state: GameStateSnapshot, top_n: int = 5) -> List[Dict]:
        """
        Get top N moves with rankings.
        
        Args:
            game_state: Current game state
            top_n: Number of moves to return
            
        Returns:
            List of move info dicts
        """
        action_probs, _, root = self.search(game_state, return_root=True)
        
        results = []
        
        # Sort children by visit count
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        
        for rank, child in enumerate(sorted_children[:top_n], 1):
            edge = game_state.edges[child.action_taken]
            confidence = (child.visits / root.visits) * 100 if root.visits > 0 else 0
            
            results.append({
                'action': child.action_taken,
                'edge_id': edge['id'],
                'visits': child.visits,
                'confidence': round(confidence, 1),
                'expected_reward': round(child.value(), 3),
                'rank': rank,
                'prior': round(child.prior, 4),
                'nn_value': round(child.value(), 3)
            })
        
        return results
    
    def _select(self, node: AZMCTSNode) -> AZMCTSNode:
        """Select node using PUCT until leaf or expandable node."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                # Expand with NN policy prior
                return self._expand(node)
            else:
                # Select best child using PUCT
                node = node.best_child_puct(self.c_puct)
        
        return node
    
    def _expand(self, node: AZMCTSNode) -> AZMCTSNode:
        """Expand node using neural network policy priors."""
        if not node.untried_actions:
            return node
        
        # Get policy from NN
        policy_probs = self._get_nn_policy(node.state)
        
        # Find untried action with highest prior
        best_action = None
        best_prior = -1
        
        for action in node.untried_actions:
            prior = policy_probs.get(action, 1e-8)
            if prior > best_prior:
                best_prior = prior
                best_action = action
        
        if best_action is None:
            best_action = node.untried_actions[0]
            best_prior = 1.0 / len(node.untried_actions)
        
        return node.expand(best_action, best_prior)
    
    def _evaluate(self, node: AZMCTSNode) -> float:
        """
        Evaluate leaf node.
        Uses neural network value, with optional heuristic fallback.
        """
        if node.is_terminal():
            # Terminal state: exact value
            winner = node.state.get_winner()
            
            if winner == 2:  # AI wins
                return 1.0
            elif winner == 1:  # AI loses
                return -1.0
            else:
                return 0.0  # Draw
        
        if self.use_value_net:
            # Use NN value prediction
            value = self._get_nn_value(node.state)
            
            # If value is near zero and we have fallback, use heuristic rollout
            if abs(value) < 0.1 and self.use_heuristic_fallback:
                value = self._heuristic_rollout(node.state)
            
            return value
        else:
            # Use heuristic rollout
            return self._heuristic_rollout(node.state)
    
    def _get_nn_policy(self, state: GameStateSnapshot) -> Dict[int, float]:
        """Get policy probabilities from neural network."""
        try:
            state_tensor = self.encoder.encode(state)
            policy_probs, _ = self.neural_net.predict(state_tensor)
            
            # Convert to dict
            valid_actions = state.get_valid_actions()
            policy_dict = {}
            
            for action in valid_actions:
                policy_dict[action] = policy_probs[action].item()
            
            return policy_dict
        except Exception as e:
            print(f"NN policy error: {e}")
            # Uniform fallback
            valid_actions = state.get_valid_actions()
            n = len(valid_actions)
            return {a: 1.0 / n for a in valid_actions}
    
    def _get_nn_value(self, state: GameStateSnapshot) -> float:
        """Get value prediction from neural network."""
        try:
            state_tensor = self.encoder.encode(state)
            _, value = self.neural_net.predict(state_tensor)
            return value.item()
        except Exception as e:
            print(f"NN value error: {e}")
            return 0.0
    
    def _heuristic_rollout(self, state: GameStateSnapshot) -> float:
        """Run a heuristic rollout (fallback)."""
        sim_state = state.clone()
        max_steps = len(sim_state.edges) - len(sim_state.claimed_edges)
        step = 0
        
        while not sim_state.is_terminal() and step < max_steps:
            action = self.heuristic_policy.select_move(sim_state)
            if action == -1:
                break
            sim_state.apply_action(action)
            step += 1
        
        # Calculate reward from AI perspective
        winner = sim_state.get_winner()
        
        if winner == 2:
            return 1.0
        elif winner == 1:
            return -1.0
        else:
            return 0.0


def create_az_mcts(model_path: str = None, simulations: int = 800,
                   use_value_net: bool = True) -> AlphaZeroMCTS:
    """
    Factory function to create AlphaZero MCTS.
    
    Args:
        model_path: Path to trained model. If None, creates untrained model.
        simulations: Number of MCTS simulations
        use_value_net: Whether to use NN value estimates
        
    Returns:
        AlphaZeroMCTS instance
    """
    from nn_model import create_model, load_trained_model
    from state_encoder import StateEncoder
    
    encoder = StateEncoder()
    
    if model_path and os.path.exists(model_path):
        model = load_trained_model(model_path)
    else:
        model = create_model(input_size=encoder.get_feature_size())
    
    return AlphaZeroMCTS(
        neural_net=model,
        encoder=encoder,
        simulations=simulations,
        use_value_net=use_value_net
    )
