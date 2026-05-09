import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List
import rl_environment

class DQN(nn.Module):
    """Deep Q-Network for Voronoi Connect 4"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    """DQN Agent for playing Voronoi Connect 4"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        self.update_target_frequency = 100
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            valid_actions: List of valid action indices
            training: Whether agent is training (affects exploration)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() <= self.epsilon:
            # Explore: choose random valid action
            return random.choice(valid_actions) if valid_actions else 0
        
        # Exploit: choose best valid action
        # Pad state to match expected input size
        padded_state = np.pad(state, (0, max(0, self.action_size + 5 - len(state))), 'constant')
        state_tensor = torch.FloatTensor(padded_state).unsqueeze(0).to(self.device)
        
        # Get Q values for available actions only
        q_values = self.q_network(state_tensor)
        
        # Mask invalid actions
        mask = torch.ones(self.action_size) * float('-inf')
        mask[valid_actions] = 0
        q_values = q_values + mask.to(self.device)
        
        return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Pad states to same length
        max_state_len = max(len(e[0]) for e in batch)
        padded_states = []
        padded_next_states = []
        
        for e in batch:
            state = np.pad(e[0], (0, max_state_len - len(e[0])), 'constant')
            next_state = np.pad(e[3], (0, max_state_len - len(e[3])), 'constant')
            padded_states.append(state)
            padded_next_states.append(next_state)
        
        states = torch.FloatTensor(padded_states).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(padded_next_states).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_episode(self, env: rl_environment.VoronoiRLEnvironment) -> float:
        """
        Train agent for one episode
        
        Args:
            env: The game environment
            
        Returns:
            Total reward for the episode
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            # Choose action
            action = self.act(state, valid_actions, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train on batch
            self.replay()
            
            # Update target network periodically
            if steps % self.update_target_frequency == 0:
                self.update_target_network()
            
            if done:
                break
        
        return total_reward
    
    def evaluate_episode(self, env: rl_environment.VoronoiRLEnvironment) -> Tuple[float, int, int]:
        """
        Evaluate agent performance for one episode (no exploration)
        
        Args:
            env: The game environment
            
        Returns:
            Tuple of (total_reward, player1_score, player2_score)
        """
        state = env.reset()
        total_reward = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            # Choose action (no exploration)
            action = self.act(state, valid_actions, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        return total_reward, info.get("player1_score", 0), info.get("player2_score", 0)
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class VoronoiAI:
    """AI player that uses trained DQN agent"""
    
    def __init__(self, agent: DQNAgent, player_number: int = 2):
        self.agent = agent
        self.player_number = player_number
    
    def get_action(self, env: rl_environment.VoronoiRLEnvironment) -> int:
        """
        Get AI's next move
        
        Args:
            env: Current game environment
            
        Returns:
            Selected action index
        """
        state = env._get_state()
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return -1
        
        # Use agent to choose action (no exploration)
        action = self.agent.act(state, valid_actions, training=False)
        return action
    
    def make_move(self, env: rl_environment.VoronoiRLEnvironment) -> Tuple[int, float, bool, dict]:
        """
        Make a move in the game
        
        Args:
            env: Current game environment
            
        Returns:
            Tuple of (action, reward, done, info)
        """
        action = self.get_action(env)
        if action == -1:
            return -1, 0, True, {"error": "No valid moves"}
        
        return env.step(action)
