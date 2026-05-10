"""
Neural Network for Voronoi AlphaZero-style RL
Policy + Value network with shared body and separate heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from fixed_board_config import NUM_EDGES


class VoronoiNet(nn.Module):
    """
    Neural network that predicts policy and value from game state.
    
    Architecture:
    - Shared body: FC layers with ReLU
    - Policy head: FC -> softmax over edges
    - Value head: FC -> tanh scalar
    """
    
    def __init__(self, input_size: int = None, num_edges: int = NUM_EDGES,
                 hidden_sizes: list = [256, 256, 128]):
        """
        Args:
            input_size: Size of encoded state vector. If None, computed from num_edges.
            num_edges: Number of edges on the board
            hidden_sizes: List of hidden layer sizes
        """
        super(VoronoiNet, self).__init__()
        
        self.num_edges = num_edges
        self.input_size = input_size or (num_edges + 4)  # edges + current_player + 2 scores + claim_ratio
        
        # Shared body
        layers = []
        prev_size = self.input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.shared_body = nn.Sequential(*layers)
        self.body_output_size = prev_size
        
        # Policy head: predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(self.body_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_edges)
        )
        
        # Value head: predicts game outcome
        self.value_head = nn.Sequential(
            nn.Linear(self.body_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,)
            
        Returns:
            tuple: (policy_logits, value)
                - policy_logits: (batch_size, num_edges) or (num_edges,)
                - value: (batch_size, 1) or (1,)
        """
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # Shared representation
        shared = self.shared_body(x)
        
        # Policy prediction
        policy_logits = self.policy_head(shared)
        
        # Value prediction
        value = self.value_head(shared)
        value = torch.tanh(value)  # Clamp to [-1, 1]
        
        # Remove batch dim if single input
        if single_input:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
        
        return policy_logits, value
    
    def predict(self, state_tensor: torch.Tensor) -> tuple:
        """
        Convenience method for inference.
        
        Args:
            state_tensor: Encoded state
            
        Returns:
            tuple: (policy_probs, value)
                - policy_probs: probability distribution over edges
                - value: predicted game outcome in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=-1)
        
        return policy_probs, value
    
    def save_model(self, filepath: str, optimizer_state: dict = None):
        """
        Save model weights and optionally optimizer state.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'num_edges': self.num_edges,
            'body_output_size': self.body_output_size
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, strict: bool = True) -> dict:
        """
        Load model weights from checkpoint.
        
        Args:
            filepath: Path to checkpoint
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            dict: Checkpoint data (may contain optimizer state)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        print(f"Model loaded from {filepath}")
        return checkpoint


def create_model(input_size: int = None, num_edges: int = NUM_EDGES) -> VoronoiNet:
    """
    Factory function to create a new model.
    
    Args:
        input_size: Size of encoded state
        num_edges: Number of edges
        
    Returns:
        VoronoiNet instance
    """
    return VoronoiNet(input_size=input_size, num_edges=num_edges)


def load_trained_model(filepath: str, input_size: int = None, num_edges: int = NUM_EDGES) -> VoronoiNet:
    """
    Load a trained model from checkpoint.
    
    Args:
        filepath: Path to checkpoint
        input_size: Size of encoded state
        num_edges: Number of edges
        
    Returns:
        VoronoiNet instance with loaded weights
    """
    model = create_model(input_size, num_edges)
    model.load_model(filepath)
    return model
