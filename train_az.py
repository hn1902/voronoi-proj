#!/usr/bin/env python3
"""
Command-line training script for Voronoi AlphaZero RL.
Run: python train_az.py --iterations 50 --games 20
"""

import argparse
import sys
import os

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_from_scratch
from evaluate import evaluate_agents


def main():
    parser = argparse.ArgumentParser(description='Train Voronoi AlphaZero RL')
    
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of training iterations (default: 50)')
    parser.add_argument('--games', type=int, default=20,
                        help='Self-play games per iteration (default: 20)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate trained model after training')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Number of evaluation games (default: 50)')
    parser.add_argument('--checkpoint-dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Voronoi AlphaZero Training")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Games/iteration: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 60)
    
    # Phase 1: Training
    print("\nPhase 1: Training from scratch...")
    train_from_scratch(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=args.simulations,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Phase 2: Evaluation (optional)
    if args.eval:
        print("\nPhase 2: Evaluating trained model...")
        
        # Find latest checkpoint
        checkpoints = sorted([
            f for f in os.listdir(args.checkpoint_dir)
            if f.startswith('model_iter_') and f.endswith('.pth')
        ])
        
        if checkpoints:
            model_path = os.path.join(args.checkpoint_dir, checkpoints[-1])
            print(f"Evaluating: {model_path}")
            
            evaluate_agents(
                num_games=args.eval_games,
                model_path=model_path,
                az_simulations=args.simulations,
                heuristic_simulations=args.simulations
            )
        else:
            print("No checkpoints found to evaluate!")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
