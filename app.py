from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import os
from dqn_agent import DQNAgent, VoronoiAI
from rl_environment import VoronoiRLEnvironment

app = Flask(__name__)

# Global variables for AI
dqn_agent = None
voronoi_ai = None
current_env = None

def initialize_ai():
    """Initialize the DQN agent with a pre-trained model or create new one"""
    global dqn_agent, voronoi_ai, current_env
    
    # Try to load a pre-trained model
    model_path = 'dqn_model.pth'
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        # Create a dummy environment to get state/action sizes
        dummy_points = [{'x': 100, 'y': 100, 'id': 0} for _ in range(8)]
        dummy_edges = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'id': f'0,0-100,100'} for _ in range(20)]
        current_env = VoronoiRLEnvironment(dummy_points, dummy_edges)
        
        state_size = current_env.get_state_size()
        action_size = current_env.get_action_size()
        
        dqn_agent = DQNAgent(state_size, action_size)
        dqn_agent.load(model_path)
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. Creating new agent...")
        # Create a dummy environment to get state/action sizes
        dummy_points = [{'x': 100, 'y': 100, 'id': 0} for _ in range(8)]
        dummy_edges = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'id': f'0,0-100,100'} for _ in range(20)]
        current_env = VoronoiRLEnvironment(dummy_points, dummy_edges)
        
        state_size = current_env.get_state_size()
        action_size = current_env.get_action_size()
        
        dqn_agent = DQNAgent(state_size, action_size)
        print("New agent created. Will use random moves initially.")
    
    voronoi_ai = VoronoiAI(dqn_agent, player_number=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """API endpoint for AI to make a move"""
    global voronoi_ai, current_env
    
    if not voronoi_ai:
        return jsonify({'error': 'AI not initialized'}), 500
    
    try:
        data = request.get_json()
        points = data.get('points', [])
        edges = data.get('edges', [])
        claimed_edges = data.get('claimed_edges', {})
        current_player = data.get('current_player', 2)
        
        # Create environment with current game state
        if current_env is None or len(current_env.edges) != len(edges):
            current_env = VoronoiRLEnvironment(points, edges)
        
        # Update environment state
        current_env.claimed_edges = claimed_edges
        current_env.current_player = current_player
        current_env._update_scores()
        
        # Get AI action
        action = voronoi_ai.get_action(current_env)
        
        if action == -1:
            return jsonify({'error': 'No valid moves available'}), 400
        
        # Execute the action to get the result
        next_state, reward, done, info = current_env.step(action)
        
        # Return the move information
        selected_edge = edges[action]
        return jsonify({
            'edge_index': action,
            'edge': selected_edge,
            'reward': reward,
            'done': done,
            'info': info
        })
        
    except Exception as e:
        print(f"Error in AI move: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/initialize_game', methods=['POST'])
def initialize_game():
    """Initialize a new game with the given points and edges"""
    global current_env
    
    try:
        data = request.get_json()
        points = data.get('points', [])
        edges = data.get('edges', [])
        
        # Create new environment
        current_env = VoronoiRLEnvironment(points, edges)
        
        return jsonify({
            'status': 'success',
            'state_size': current_env.get_state_size(),
            'action_size': current_env.get_action_size()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_agent', methods=['POST'])
def train_agent():
    """Train the DQN agent (optional endpoint)"""
    global dqn_agent, current_env
    
    if not dqn_agent or not current_env:
        return jsonify({'error': 'Agent or environment not initialized'}), 500
    
    try:
        data = request.get_json()
        episodes = data.get('episodes', 100)
        
        total_rewards = []
        for episode in range(episodes):
            reward = dqn_agent.train_episode(current_env)
            total_rewards.append(reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                print(f"Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}")
        
        # Save the trained model
        dqn_agent.save('dqn_model.pth')
        
        return jsonify({
            'status': 'success',
            'episodes_trained': episodes,
            'average_reward': np.mean(total_rewards)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize AI when starting the app
    initialize_ai()
    app.run(debug=True, host='0.0.0.0', port=5000)