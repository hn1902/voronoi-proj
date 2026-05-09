from flask import Flask, render_template, request, jsonify
import numpy as np
# import torch  # DQN dependencies - not currently used
# import os
# from dqn_agent import DQNAgent, VoronoiAI  # DQN - not currently used
# from rl_environment import VoronoiRLEnvironment  # DQN - not currently used
from mcts_ai import MCTSAI, GameStateSnapshot, create_state_from_game

app = Flask(__name__)

# Global variables for AI
# dqn_agent = None  # DQN - not currently used
# voronoi_ai = None  # DQN - not currently used
# current_env = None  # DQN - not currently used
mcts_ai = None  # MCTS AI instance - ACTIVE

def initialize_ai():
    """Initialize the MCTS AI agent"""
    global mcts_ai
    
    # Initialize MCTS AI (active system)
    print("Initializing MCTS AI...")
    mcts_ai = MCTSAI(player_number=2, simulations=500, exploration_constant=1.4, debug=True)
    print("MCTS AI initialized with 500 simulations")
    
    # DQN agent initialization - commented out (not currently used)
    # model_path = 'dqn_model.pth'
    # if os.path.exists(model_path):
    #     print(f"Loading pre-trained DQN model from {model_path}")
    #     # ... DQN initialization code ...
    # else:
    #     print("No pre-trained DQN model found.")

@app.route('/')
def index():
    return render_template('index.html')

# DQN endpoints - commented out (not currently used)
# @app.route('/api/ai_move', methods=['POST'])
# def ai_move():
#     """API endpoint for DQN AI to make a move - NOT ACTIVE"""
#     pass

# @app.route('/api/initialize_game', methods=['POST'])
# def initialize_game():
#     """Initialize DQN game environment - NOT ACTIVE"""
#     pass

# @app.route('/api/train_agent', methods=['POST'])
# def train_agent():
#     """Train the DQN agent - NOT ACTIVE"""
#     pass


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


@app.route('/api/mcts_suggestions', methods=['POST'])
def mcts_suggestions():
    """
    Get top N move suggestions from MCTS AI.
    Returns ranked list of best moves with confidence scores.
    """
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
        top_n = data.get('top_n', 5)  # Number of suggestions to return
        simulations = data.get('simulations', 500)
        
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
        
        # Get top N suggestions
        suggestions = mcts_ai.get_top_moves(state, top_n=top_n)
        
        return jsonify({
            'status': 'success',
            'suggestions': suggestions,
            'total_simulations': mcts_ai.simulations,
            'current_player': current_player
        })
        
    except Exception as e:
        print(f"Error in MCTS suggestions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize AI when starting the app
    initialize_ai()
    app.run(debug=True, host='0.0.0.0', port=5000)