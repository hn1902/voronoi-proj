"""
Evaluation System for Voronoi AlphaZero RL
Compare trained NN+MCTS vs heuristic MCTS vs random agent.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple
from mcts_ai import GameStateSnapshot, MCTSAI
from state_encoder import StateEncoder
from nn_model import VoronoiNet, load_trained_model
from alpha_zero_mcts import AlphaZeroMCTS
from fixed_board_config import FIXED_POINTS, FIXED_EDGES


class RandomAgent:
    """Random move agent for baseline comparison."""
    
    def select_move(self, state: GameStateSnapshot) -> int:
        valid = state.get_valid_actions()
        return np.random.choice(valid) if valid else -1


class HeuristicMCTSAgent:
    """Wrapper for existing heuristic MCTS."""
    
    def __init__(self, player_number: int = 2, simulations: int = 400):
        self.mcts = MCTSAI(player_number=player_number, simulations=simulations)
    
    def select_move(self, state: GameStateSnapshot) -> int:
        return self.mcts.get_best_move(state)


class AlphaZeroAgent:
    """Wrapper for AlphaZero MCTS."""
    
    def __init__(self, neural_net: VoronoiNet, player_number: int = 2,
                 simulations: int = 400, temperature: float = 0.0):
        self.neural_net = neural_net
        self.encoder = StateEncoder()
        self.player_number = player_number
        self.simulations = simulations
        self.temperature = temperature
        self.az_mcts = AlphaZeroMCTS(
            neural_net=neural_net,
            encoder=self.encoder,
            simulations=simulations,
            temperature=temperature,
            use_value_net=True
        )
    
    def select_move(self, state: GameStateSnapshot) -> int:
        return self.az_mcts.get_best_move(state)


def play_match(player1_agent, player2_agent, verbose: bool = False) -> Tuple[int, Dict]:
    """
    Play one match between two agents.
    
    Args:
        player1_agent: Agent for player 1
        player2_agent: Agent for player 2
        verbose: Print game moves
        
    Returns:
        Tuple of (winner, game_stats)
    """
    state = GameStateSnapshot(
        points=FIXED_POINTS,
        edges=FIXED_EDGES,
        claimed_edges={},
        current_player=1,
        player1_score=0,
        player2_score=0
    )
    
    move_count = 0
    
    while not state.is_terminal():
        move_count += 1
        
        if state.current_player == 1:
            action = player1_agent.select_move(state)
        else:
            action = player2_agent.select_move(state)
        
        if action == -1:
            break
        
        state.apply_action(action)
        
        if verbose:
            print(f"  Move {move_count}: P{state.current_player} -> action {action}")
    
    winner = state.get_winner()
    
    stats = {
        'winner': winner,
        'player1_score': state.player1_score,
        'player2_score': state.player2_score,
        'total_moves': move_count
    }
    
    return winner, stats


def evaluate_agents(num_games: int = 100, model_path: str = None,
                    az_simulations: int = 400, heuristic_simulations: int = 400,
                    verbose: bool = True) -> Dict:
    """
    Evaluate AlphaZero agent against baseline agents.
    
    Args:
        num_games: Number of games per matchup
        model_path: Path to trained AZ model (None = untrained)
        az_simulations: Simulations for AZ agent
        heuristic_simulations: Simulations for heuristic agent
        verbose: Print results
        
    Returns:
        Dict with evaluation results
    """
    print("=" * 60)
    print("Voronoi AlphaZero Evaluation")
    print("=" * 60)
    
    # Create agents
    encoder = StateEncoder()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = load_trained_model(model_path)
    else:
        print("Using untrained model (random weights)")
        from nn_model import create_model
        model = create_model(input_size=encoder.get_feature_size())
    
    az_agent = AlphaZeroAgent(model, player_number=2, simulations=az_simulations)
    heuristic_agent = HeuristicMCTSAgent(player_number=2, simulations=heuristic_simulations)
    random_agent = RandomAgent()
    
    results = {
        'az_vs_random': {'wins': 0, 'losses': 0, 'draws': 0, 'games': []},
        'az_vs_heuristic': {'wins': 0, 'losses': 0, 'draws': 0, 'games': []},
        'heuristic_vs_random': {'wins': 0, 'losses': 0, 'draws': 0, 'games': []}
    }
    
    # Evaluate AZ vs Random (AZ plays as P2)
    print(f"\n[1/3] AZ vs Random ({num_games} games)...")
    for i in range(num_games):
        winner, stats = play_match(random_agent, az_agent)
        
        if winner == 2:  # AZ wins
            results['az_vs_random']['wins'] += 1
        elif winner == 1:  # AZ loses
            results['az_vs_random']['losses'] += 1
        else:
            results['az_vs_random']['draws'] += 1
        
        results['az_vs_random']['games'].append(stats)
    
    # Evaluate AZ vs Heuristic (AZ plays as P2)
    print(f"[2/3] AZ vs Heuristic ({num_games} games)...")
    for i in range(num_games):
        winner, stats = play_match(heuristic_agent, az_agent)
        
        if winner == 2:
            results['az_vs_heuristic']['wins'] += 1
        elif winner == 1:
            results['az_vs_heuristic']['losses'] += 1
        else:
            results['az_vs_heuristic']['draws'] += 1
        
        results['az_vs_heuristic']['games'].append(stats)
    
    # Evaluate Heuristic vs Random (Heuristic plays as P2)
    print(f"[3/3] Heuristic vs Random ({num_games} games)...")
    for i in range(num_games):
        winner, stats = play_match(random_agent, heuristic_agent)
        
        if winner == 2:
            results['heuristic_vs_random']['wins'] += 1
        elif winner == 1:
            results['heuristic_vs_random']['losses'] += 1
        else:
            results['heuristic_vs_random']['draws'] += 1
        
        results['heuristic_vs_random']['games'].append(stats)
    
    # Print results
    if verbose:
        print_results(results)
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for matchup, data in results.items():
        total = data['wins'] + data['losses'] + data['draws']
        win_rate = data['wins'] / total * 100 if total > 0 else 0
        
        print(f"\n{matchup.replace('_', ' ').title()}:")
        print(f"  Wins: {data['wins']}/{total} ({win_rate:.1f}%)")
        print(f"  Losses: {data['losses']}/{total}")
        print(f"  Draws: {data['draws']}/{total}")
    
    print("=" * 60)


def visualize_policy(model: VoronoiNet, state: GameStateSnapshot = None):
    """
    Visualize neural network policy output for a given state.
    
    Args:
        model: Trained neural network
        state: Game state (uses initial state if None)
    """
    from state_encoder import encode_state
    
    if state is None:
        state = GameStateSnapshot(
            points=FIXED_POINTS,
            edges=FIXED_EDGES,
            claimed_edges={},
            current_player=1,
            player1_score=0,
            player2_score=0
        )
    
    encoder = StateEncoder()
    state_tensor = encoder.encode(state)
    
    model.eval()
    with torch.no_grad():
        policy_logits, value = model.forward(state_tensor)
        policy_probs = torch.softmax(policy_logits, dim=-1)
    
    print("\nPolicy Visualization:")
    print(f"Value estimate: {value.item():.3f}")
    print("\nEdge probabilities:")
    
    valid_actions = state.get_valid_actions()
    for action in valid_actions:
        prob = policy_probs[action].item()
        edge = state.edges[action]
        print(f"  Edge {action} ({edge['id']}): {prob:.4f}")



if __name__ == '__main__':
    print("Voronoi AlphaZero Evaluation")
    print("Usage: python evaluate.py")
    print("\nTo evaluate a trained model:")
    print("  evaluate_agents(model_path='models/model_iter_50.pth')")
