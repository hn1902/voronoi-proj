#!/usr/bin/env python3
"""
Training script for fixed Voronoi Connect 4 RL Agent
"""

import numpy as np
import random
import os
import time
from typing import List, Dict
import fixed_rl_environment
from fixed_dqn_agent import FixedDQNAgent, FixedVoronoiAI

def train_fixed_agent(episodes: int = 1000, save_path: str = 'models/fixed_voronoi_dqn.pth'):
    """Train the DQN agent on fixed board"""
    print("Starting training on fixed Voronoi board...")
    
    # Initialize environment and agent
    env = fixed_rl_environment.FixedVoronoiRLEnvironment()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    print(f"Environment: State size={state_size}, Action size={action_size}")
    
    agent = FixedDQNAgent(state_size, action_size)
    
    # Training metrics
    episode_rewards = []
    win_rates = []
    
    for episode in range(episodes):
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
    os.makedirs('models', exist_ok=True)
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    return agent, episode_rewards, win_rates

def self_play_training(episodes: int = 500, save_path: str = 'models/fixed_voronoi_selfplay.pth'):
    """Train agent through self-play on fixed board"""
    print("Starting self-play training on fixed board...")
    
    # Initialize environment and two agents
    env = fixed_rl_environment.FixedVoronoiRLEnvironment()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    agent1 = FixedDQNAgent(state_size, action_size)
    agent2 = FixedDQNAgent(state_size, action_size)
    
    episode_rewards = []
    
    for episode in range(episodes):
        env.reset()
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
    os.makedirs('models', exist_ok=True)
    agent1.save(save_path.replace('.pth', '_agent1.pth'))
    agent2.save(save_path.replace('.pth', '_agent2.pth'))
    print(f"Self-play models saved")
    
    return agent1, agent2, episode_rewards

def evaluate_fixed_agent(model_path: str, eval_episodes: int = 100):
    """Evaluate a trained agent on fixed board"""
    print(f"Evaluating agent from {model_path}...")
    
    # Load agent
    env = fixed_rl_environment.FixedVoronoiRLEnvironment()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    agent = FixedDQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Evaluation metrics
    wins = 0
    draws = 0
    losses = 0
    total_rewards = []
    
    for episode in range(eval_episodes):
        env.reset()
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

def quick_test():
    """Quick test of the fixed board system"""
    print("Quick test of fixed board system...")
    
    # Test environment
    env = fixed_rl_environment.FixedVoronoiRLEnvironment()
    print(f"State size: {env.get_state_size()}")
    print(f"Action size: {env.get_action_size()}")
    
    # Test agent
    agent = FixedDQNAgent(env.get_state_size(), env.get_action_size())
    
    # Play a quick game
    state = env.reset()
    done = False
    step = 0
    
    while not done and step < 20:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        action = random.choice(valid_actions)  # Random action
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}: Action {action}, Reward {reward}")
        print(f"  Scores: P1={info['player1_score']}, P2={info['player2_score']}")
        
        state = next_state
        step += 1
    
    print("Quick test completed!")

if __name__ == "__main__":
    import sys
    
    print("Fixed Voronoi Connect 4 RL Training")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            quick_test()
        elif command == "train":
            agent, rewards, win_rates = train_fixed_agent(episodes=1000)
        elif command == "selfplay":
            agent1, agent2, rewards = self_play_training(episodes=500)
        elif command == "evaluate":
            evaluate_fixed_agent('models/fixed_voronoi_dqn.pth')
        else:
            print("Usage: python train_fixed_ai.py [test|train|selfplay|evaluate]")
    else:
        # Default: quick test
        quick_test()
        print("\nTo run full training, use:")
        print("  python train_fixed_ai.py train")
        print("  python train_fixed_ai.py selfplay")
        print("  python train_fixed_ai.py evaluate")
