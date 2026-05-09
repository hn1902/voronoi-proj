#!/usr/bin/env python3
"""
Training script for Voronoi Connect 4 RL Agent
"""

import numpy as np
import random
import json
import time
from typing import List, Dict
import rl_environment
from dqn_agent import DQNAgent, VoronoiAI

def generate_random_points(count: int, width: int = 800, height: int = 600) -> List[Dict]:
    """Generate random points for Voronoi diagram"""
    points = []
    margin = 50
    
    for i in range(count):
        points.append({
            'x': margin + random.random() * (width - 2 * margin),
            'y': margin + random.random() * (height - 2 * margin),
            'id': i
        })
    
    return points

def extract_edges_from_voronoi(points: List[Dict]) -> List[Dict]:
    """
    Extract edges from Voronoi diagram (simplified version)
    In a real implementation, this would use D3.js or similar
    """
    # This is a simplified edge generation
    # In practice, you'd want to use the same Voronoi generation as the frontend
    edges = []
    edge_id = 0
    
    # Create a simple triangulation-based edge set
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points[i+1:], i+1):
            # Add edge if points are reasonably close (simplified)
            dist = np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
            if dist < 200:  # Threshold for edge creation
                edges.append({
                    'id': f"{point1['x']:.1f},{point1['y']:.1f}-{point2['x']:.1f},{point2['y']:.1f}",
                    'x1': point1['x'],
                    'y1': point1['y'],
                    'x2': point2['x'],
                    'y2': point2['y']
                })
                edge_id += 1
    
    return edges

def train_agent(episodes: int = 1000, save_path: str = 'models/voronoi_dqn.pth'):
    """Train the DQN agent"""
    print("Starting training...")
    
    # Initialize agent
    # We'll use a fixed size for now, in practice this should be dynamic
    state_size = 45  # Max edges (40) + 5 features
    action_size = 40  # Max edges
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    episode_rewards = []
    win_rates = []
    
    for episode in range(episodes):
        # Generate random game configuration
        num_points = random.randint(5, 12)
        points = generate_random_points(num_points)
        edges = extract_edges_from_voronoi(points)
        
        # Adjust action size for this specific game
        current_action_size = len(edges)
        if current_action_size > action_size:
            # Skip games with too many edges for now
            continue
        
        # Create environment
        env = rl_environment.VoronoiRLEnvironment(points, edges)
        
        # Train one episode
        reward = agent.train_episode(env)
        episode_rewards.append(reward)
        
        # Evaluate performance every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Evaluate win rate
            wins = 0
            eval_episodes = 20
            for _ in range(eval_episodes):
                eval_reward, p1_score, p2_score = agent.evaluate_episode(env)
                if p2_score > p1_score:  # Agent plays as player 2
                    wins += 1
            
            win_rate = wins / eval_episodes
            win_rates.append(win_rate)
            print(f"Win Rate (as Player 2): {win_rate:.2%}")
    
    # Save the trained model
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    return agent, episode_rewards, win_rates

def self_play_training(episodes: int = 500, save_path: str = 'models/voronoi_dqn_selfplay.pth'):
    """Train agent through self-play"""
    print("Starting self-play training...")
    
    # Initialize two agents
    state_size = 45
    action_size = 40
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    
    episode_rewards = []
    
    for episode in range(episodes):
        # Generate random game configuration
        num_points = random.randint(5, 12)
        points = generate_random_points(num_points)
        edges = extract_edges_from_voronoi(points)
        
        current_action_size = len(edges)
        if current_action_size > action_size:
            continue
        
        env = rl_environment.VoronoiRLEnvironment(points, edges)
        state = env.reset()
        total_reward1 = 0
        total_reward2 = 0
        done = False
        
        while not done:
            # Agent 1's turn
            if env.current_player == 1:
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    action = agent1.act(state, valid_actions, training=True)
                    next_state, reward, done, info = env.step(action)
                    agent1.remember(state, action, reward, next_state, done)
                    total_reward1 += reward
                    state = next_state
                    agent1.replay()
                else:
                    break
            
            # Agent 2's turn
            elif env.current_player == 2:
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    action = agent2.act(state, valid_actions, training=True)
                    next_state, reward, done, info = env.step(action)
                    agent2.remember(state, action, reward, next_state, done)
                    total_reward2 += reward
                    state = next_state
                    agent2.replay()
                else:
                    break
        
        episode_rewards.append((total_reward1, total_reward2))
        
        if (episode + 1) % 100 == 0:
            avg_reward1 = np.mean([r[0] for r in episode_rewards[-100:]])
            avg_reward2 = np.mean([r[1] for r in episode_rewards[-100:]])
            print(f"Episode {episode + 1}/{episodes}, Avg Rewards - P1: {avg_reward1:.2f}, P2: {avg_reward2:.2f}")
    
    # Save both agents
    agent1.save(save_path.replace('.pth', '_agent1.pth'))
    agent2.save(save_path.replace('.pth', '_agent2.pth'))
    print(f"Self-play models saved")
    
    return agent1, agent2, episode_rewards

def evaluate_agent(model_path: str, eval_episodes: int = 100):
    """Evaluate a trained agent"""
    print(f"Evaluating agent from {model_path}...")
    
    # Load agent
    state_size = 45
    action_size = 40
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Evaluation metrics
    wins = 0
    draws = 0
    losses = 0
    total_rewards = []
    
    for episode in range(eval_episodes):
        # Generate game
        num_points = random.randint(5, 12)
        points = generate_random_points(num_points)
        edges = extract_edges_from_voronoi(points)
        
        if len(edges) > action_size:
            continue
        
        env = rl_environment.VoronoiRLEnvironment(points, edges)
        reward, p1_score, p2_score = agent.evaluate_episode(env)
        total_rewards.append(reward)
        
        if p2_score > p1_score:
            wins += 1
        elif p1_score > p2_score:
            losses += 1
        else:
            draws += 1
    
    win_rate = wins / eval_episodes
    draw_rate = draws / eval_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"Evaluation Results:")
    print(f"  Episodes: {eval_episodes}")
    print(f"  Wins: {wins} ({win_rate:.2%})")
    print(f"  Draws: {draws} ({draw_rate:.2%})")
    print(f"  Losses: {losses} ({(1-win_rate-draw_rate):.2%})")
    print(f"  Average Reward: {avg_reward:.2f}")
    
    return win_rate, avg_reward

def create_sample_model():
    """Create a simple sample model for testing"""
    print("Creating sample model...")
    
    # Create a simple agent
    state_size = 45
    action_size = 40
    agent = DQNAgent(state_size, action_size)
    
    # Train for a few episodes
    for episode in range(10):
        num_points = 8
        points = generate_random_points(num_points)
        edges = extract_edges_from_voronoi(points)
        
        if len(edges) > action_size:
            continue
        
        env = rl_environment.VoronoiRLEnvironment(points, edges)
        reward = agent.train_episode(env)
        print(f"Sample training episode {episode + 1}: reward = {reward}")
    
    # Save the sample model
    sample_path = 'models/sample_model.pth'
    agent.save(sample_path)
    print(f"Sample model saved to {sample_path}")
    
    return agent

if __name__ == "__main__":
    import os
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("Voronoi Connect 4 RL Training")
    print("=" * 40)
    
    # Create sample model for quick testing
    print("\n1. Creating sample model...")
    sample_agent = create_sample_model()
    
    # Uncomment below lines for full training
    # print("\n2. Starting full training...")
    # trained_agent, rewards, win_rates = train_agent(episodes=1000)
    
    # print("\n3. Evaluating trained agent...")
    # evaluate_agent('models/voronoi_dqn.pth')
    
    print("\nTraining complete! Try running the web app and enabling AI mode.")
