"""
State Encoder for Voronoi Game
Converts GameStateSnapshot into fixed-size numerical tensors for neural network input.
"""

import torch
import numpy as np
from typing import Dict, List
from fixed_board_config import NUM_EDGES, FIXED_EDGES


class StateEncoder:
    """
    Encodes game state into fixed-size tensor.
    
    Output features:
    - Edge ownership: NUM_EDGES values (0=unclaimed, 1=player1, 2=player2)
    - Current player: 1 value (0=player1, 1=player2) 
    - Player scores: 2 values (normalized)
    - Total edges claimed: 1 value (normalized)
    
    Total: NUM_EDGES + 4 features
    """
    
    def __init__(self, num_edges: int = NUM_EDGES):
        self.num_edges = num_edges
        self.num_features = num_edges + 4  # edges + current_player + 2 scores + claim_ratio
        
        # Build edge index mapping for consistent ordering
        self.edge_to_idx = {edge['id']: i for i, edge in enumerate(FIXED_EDGES)}
    
    def encode(self, state) -> torch.Tensor:
        """
        Encode a game state into a fixed-size tensor.
        
        Args:
            state: GameStateSnapshot object or dict with required attributes
            
        Returns:
            torch.Tensor of shape (self.num_features,)
        """
        features = []
        
        # 1. Edge ownership vector (0=unclaimed, 1=p1, 2=p2)
        edge_ownership = [0.0] * self.num_edges
        
        # Handle both GameStateSnapshot and dict inputs
        if hasattr(state, 'claimed_edges'):
            claimed = state.claimed_edges
            edges = state.edges
        else:
            claimed = state.get('claimed_edges', {})
            edges = state.get('edges', FIXED_EDGES)
        
        for edge_id, player in claimed.items():
            idx = self.edge_to_idx.get(edge_id)
            if idx is not None:
                edge_ownership[idx] = float(player)  # 1.0 or 2.0
        
        features.extend(edge_ownership)
        
        # 2. Current player (0 for p1, 1 for p2)
        if hasattr(state, 'current_player'):
            current = state.current_player
        else:
            current = state.get('current_player', 1)
        features.append(0.0 if current == 1 else 1.0)
        
        # 3. Scores (normalized by max possible)
        if hasattr(state, 'player1_score'):
            p1_score = state.player1_score
            p2_score = state.player2_score
        else:
            p1_score = state.get('player1_score', 0)
            p2_score = state.get('player2_score', 0)
        
        max_score = self.num_edges * 4  # Rough upper bound
        features.append(p1_score / max_score)
        features.append(p2_score / max_score)
        
        # 4. Claim ratio (progress indicator)
        num_claimed = len(claimed)
        features.append(num_claimed / self.num_edges)
        
        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32)
        return tensor
    
    def encode_batch(self, states: List) -> torch.Tensor:
        """
        Encode a batch of states.
        
        Args:
            states: List of game states
            
        Returns:
            torch.Tensor of shape (batch_size, self.num_features)
        """
        encoded = [self.encode(s) for s in states]
        return torch.stack(encoded)
    
    def get_feature_size(self) -> int:
        """Return the size of encoded state vector."""
        return self.num_features


def encode_state(state) -> torch.Tensor:
    """
    Convenience function to encode a single state.
    
    Args:
        state: GameStateSnapshot or dict
        
    Returns:
        torch.Tensor
    """
    encoder = StateEncoder()
    return encoder.encode(state)


def decode_policy(policy_tensor: torch.Tensor, valid_actions: List[int]) -> Dict[int, float]:
    """
    Decode policy tensor into action probabilities.
    Only returns probabilities for valid actions.
    
    Args:
        policy_tensor: Output from policy head (shape: NUM_EDGES)
        valid_actions: List of valid action indices
        
    Returns:
        Dict mapping action index to probability
    """
    probs = torch.softmax(policy_tensor, dim=0)
    
    # Only keep valid actions
    valid_probs = {a: probs[a].item() for a in valid_actions}
    
    # Renormalize
    total = sum(valid_probs.values())
    if total > 0:
        valid_probs = {a: p / total for a, p in valid_probs.items()}
    else:
        # Uniform if all invalid
        n = len(valid_actions)
        valid_probs = {a: 1.0 / n for a in valid_actions}
    
    return valid_probs
