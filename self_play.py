"""
Self-Play System for Voronoi AlphaZero RL
Generates training data by having the AI play against itself.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Tuple
from mcts_ai import GameStateSnapshot
from state_encoder import StateEncoder
from nn_model import VoronoiNet
from alpha_zero_mcts import AlphaZeroMCTS
from fixed_board_config import FIXED_POINTS, FIXED_EDGES


class TrainingExample:
    """
    Single training example from self-play.
    Stores: state, policy target (MCTS visit counts), value target (game outcome)
    """
    
    def __init__(self, state_tensor: torch.Tensor, policy_target: torch.Tensor,
                 value_target: float, current_player: int):
        self.state_tensor = state_tensor
        self.policy_target = policy_target
        self.value_target = value_target
        self.current_player = current_player
    
    def to_tuple(self) -> Tuple:
        return (self.state_tensor, self.policy_target, self.value_target, self.current_player)


class SelfPlayEngine:
    """
    Generates self-play games using MCTS+NN for both players.
    """
    
    def __init__(self, neural_net: VoronoiNet, encoder: StateEncoder,
                 num_simulations: int = 400, temperature: float = 1.0,
                 temperature_drop: int = 10):
        """
        Args:
            neural_net: Trained neural network
            encoder: State encoder
            num_simulations: MCTS simulations per move
            temperature: Action selection temperature
            temperature_drop: Move number after which temperature drops to 0
        """
        self.neural_net = neural_net
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.temperature_drop = temperature_drop
    
    def play_game(self, verbose: bool = False) -> Tuple[List[TrainingExample], Dict]:
        """
        Play one complete self-play game.
        
        Args:
            verbose: Print game progress
            
        Returns:
            Tuple of (training_examples, game_stats)
        """
        # Initialize game state with fixed board
        state = GameStateSnapshot(
            points=FIXED_POINTS,
            edges=FIXED_EDGES,
            claimed_edges={},
            current_player=1,
            player1_score=0,
            player2_score=0
        )
        
        game_history = []  # List of (state_tensor, policy_probs, current_player)
        move_count = 0
        
        if verbose:
            print("Starting self-play game...")
        
        while not state.is_terminal():
            move_count += 1
            
            # Adjust temperature (explore early, exploit late)
            temp = self.temperature if move_count <= self.temperature_drop else 0.25
            
            # Create MCTS for this player
            # Both players use the same NN but from their perspective
            az_mcts = AlphaZeroMCTS(
                neural_net=self.neural_net,
                encoder=self.encoder,
                simulations=self.num_simulations,
                temperature=temp,
                use_value_net=True,
                use_heuristic_fallback=True
            )
            
            # Search
            action_probs, root_value = az_mcts.search(state)
            
            if not action_probs:
                break
            
            # Encode state
            state_tensor = self.encoder.encode(state)
            
            # Store training data
            # Policy target: full probability vector over all edges
            policy_target = torch.zeros(len(FIXED_EDGES), dtype=torch.float32)
            for action, prob in action_probs.items():
                policy_target[action] = prob
            
            game_history.append((state_tensor, policy_target, state.current_player))
            
            # Select action based on probabilities (sampling)
            actions = list(action_probs.keys())
            probs = np.array(list(action_probs.values()), dtype=np.float64)
            
            # Normalize to ensure sum = 1 (floating point safety)
            probs = probs / probs.sum()
            
            # Temperature-adjusted sampling
            if temp == 0:
                chosen_action = actions[np.argmax(probs)]
            else:
                chosen_action = np.random.choice(actions, p=probs)
            
            if verbose:
                print(f"  Move {move_count}: Player {state.current_player} chose action {chosen_action} "
                      f"(value={root_value:.3f})")
            
            # Apply action
            state.apply_action(chosen_action)
        
        # Game over - determine winner
        winner = state.get_winner()
        
        if verbose:
            print(f"Game complete! Winner: {winner}")
            print(f"  Player 1 score: {state.player1_score}")
            print(f"  Player 2 score: {state.player2_score}")
            print(f"  Total moves: {move_count}")
        
        # Create training examples with correct value targets
        training_examples = []
        
        for state_tensor, policy_target, player in game_history:
            # Value from perspective of the player who made this move
            if winner == 0:  # Draw
                value = 0.0
            elif winner == player:
                value = 1.0  # This player won
            else:
                value = -1.0  # This player lost
            
            example = TrainingExample(state_tensor, policy_target, value, player)
            training_examples.append(example)
        
        # Game statistics
        stats = {
            'winner': winner,
            'player1_score': state.player1_score,
            'player2_score': state.player2_score,
            'total_moves': move_count,
            'num_examples': len(training_examples)
        }
        
        return training_examples, stats
    
    def generate_games(self, num_games: int = 100, verbose: bool = False) -> Tuple[List[TrainingExample], List[Dict]]:
        """
        Generate multiple self-play games.
        
        Args:
            num_games: Number of games to play
            verbose: Print progress
            
        Returns:
            Tuple of (all_examples, all_stats)
        """
        all_examples = []
        all_stats = []
        
        start_time = time.time()
        
        for game_idx in range(num_games):
            if verbose and game_idx % 10 == 0:
                print(f"Playing game {game_idx + 1}/{num_games}...")
            
            examples, stats = self.play_game(verbose=(verbose and game_idx < 3))
            
            all_examples.extend(examples)
            all_stats.append(stats)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nSelf-play complete!")
            print(f"  Games played: {num_games}")
            print(f"  Total examples: {len(all_examples)}")
            print(f"  Time: {elapsed:.1f}s ({elapsed/num_games:.1f}s per game)")
            
            # Win statistics
            p1_wins = sum(1 for s in all_stats if s['winner'] == 1)
            p2_wins = sum(1 for s in all_stats if s['winner'] == 2)
            draws = sum(1 for s in all_stats if s['winner'] == 0)
            print(f"  Results: P1 wins={p1_wins}, P2 wins={p2_wins}, Draws={draws}")
        
        return all_examples, all_stats


def generate_training_data(model: VoronoiNet, num_games: int = 100,
                           num_simulations: int = 400, verbose: bool = True) -> List[TrainingExample]:
    """
    Convenience function to generate training data.
    
    Args:
        model: Neural network model
        num_games: Number of self-play games
        num_simulations: MCTS simulations per move
        verbose: Print progress
        
    Returns:
        List of TrainingExample objects
    """
    encoder = StateEncoder()
    engine = SelfPlayEngine(
        neural_net=model,
        encoder=encoder,
        num_simulations=num_simulations
    )
    
    examples, stats = engine.generate_games(num_games, verbose=verbose)
    return examples
