from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import os
from dqn_agent import DQNAgent, VoronoiAI
from rl_environment import VoronoiRLEnvironment
from mcts_ai import MCTSAI, GameStateSnapshot, create_state_from_game

app = Flask(__name__)

# Global variables for AI
dqn_agent = None
voronoi_ai = None
current_env = None
mcts_ai = None  # MCTS AI instance

def initialize_ai():
    """Initialize the DQN agent with a pre-trained model or create new one"""
    global dqn_agent, voronoi_ai, current_env, mcts_ai
    
    # Initialize MCTS AI (always available)
    print("Initializing MCTS AI...")
    mcts_ai = MCTSAI(player_number=2, simulations=500, exploration_constant=1.4, debug=True)
    print("MCTS AI initialized with 500 simulations")
    
    # Try to load a pre-trained DQN model (optional)
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
        print("No pre-trained DQN model found. DQN agent not initialized.")
        dqn_agent = None
    
    if dqn_agent:
        voronoi_ai = VoronoiAI(dqn_agent, player_number=2)
    else:
        voronoi_ai = None

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


# ==================== MCTS AI ENDPOINTS ====================

@app.route('/api/mcts_move', methods=['POST'])
def mcts_move():
    """API endpoint for MCTS AI to make a move"""
    global mcts_ai
    
    if not mcts_ai:
        return jsonify({'error': 'MCTS AI not initialized'}), 500
    
    try:
        data = request.get_json()
        points = data.get('points', [])
        edges = data.get('edges', [])
        claimed_edges = data.get('claimed_edges', {})
        current_player = data.get('current_player', 2)
        player1_score = data.get('player1_score', 0)
        player2_score = data.get('player2_score', 0)
        simulations = data.get('simulations', 500)  # Allow frontend to customize
        
        # Update simulation count if provided
        if simulations != mcts_ai.simulations:
            mcts_ai.set_simulations(simulations)
        
        # Create game state snapshot
        state = GameStateSnapshot(
            points=points,
            edges=edges,
            claimed_edges=claimed_edges,
            current_player=current_player,
            player1_score=player1_score,
            player2_score=player2_score
        )
        
        # Get best move from MCTS
        action = mcts_ai.get_best_move(state)
        
        if action == -1:
            return jsonify({'error': 'No valid moves available'}), 400
        
        # Get the selected edge details
        selected_edge = edges[action]
        
        # Simulate the move to get resulting state info
        temp_state = state.clone()
        temp_state.apply_action(action)
        
        return jsonify({
            'status': 'success',
            'edge_index': action,
            'edge': selected_edge,
            'simulations_run': mcts_ai.simulations,
            'debug_stats': mcts_ai.last_search_stats if mcts_ai.debug else None,
            'new_scores': {
                'player1': temp_state.player1_score,
                'player2': temp_state.player2_score
            },
            'is_terminal': temp_state.is_terminal(),
            'winner': temp_state.get_winner() if temp_state.is_terminal() else None
        })
        
    except Exception as e:
        print(f"Error in MCTS move: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/mcts_config', methods=['POST'])
def mcts_config():
    """Configure MCTS AI parameters"""
    global mcts_ai
    
    if not mcts_ai:
        return jsonify({'error': 'MCTS AI not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Update parameters if provided
        if 'simulations' in data:
            mcts_ai.set_simulations(data['simulations'])
        
        if 'exploration_constant' in data:
            mcts_ai.c = data['exploration_constant']
        
        if 'debug' in data:
            mcts_ai.debug = data['debug']
        
        return jsonify({
            'status': 'success',
            'current_config': {
                'simulations': mcts_ai.simulations,
                'exploration_constant': mcts_ai.c,
                'debug': mcts_ai.debug
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mcts_stats', methods=['GET'])
def mcts_stats():
    """Get MCTS AI statistics from last search"""
    global mcts_ai
    
    if not mcts_ai:
        return jsonify({'error': 'MCTS AI not initialized'}), 500
    
    return jsonify({
        'status': 'success',
        'last_search_stats': mcts_ai.last_search_stats,
        'current_config': {
            'simulations': mcts_ai.simulations,
            'exploration_constant': mcts_ai.c,
            'debug': mcts_ai.debug
        }
    })


if __name__ == '__main__':
    # Initialize AI when starting the app
    initialize_ai()
    app.run(debug=True, host='0.0.0.0', port=5000)