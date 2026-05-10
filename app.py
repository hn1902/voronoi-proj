from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import os

from mcts_ai import MCTSAI, GameStateSnapshot, create_state_from_game

# AlphaZero imports
from state_encoder import StateEncoder
from nn_model import VoronoiNet, create_model, load_trained_model
from alpha_zero_mcts import AlphaZeroMCTS

app = Flask(__name__)

# Global variables for AI
mcts_ai = None  # Heuristic MCTS AI instance
az_model = None  # AlphaZero neural network
az_mcts = None  # AlphaZero MCTS
state_encoder = None  # State encoder for AlphaZero

def initialize_ai():
    """Initialize both MCTS and AlphaZero AI agents"""
    global mcts_ai, az_model, az_mcts, state_encoder
    
    # Initialize Heuristic MCTS
    print("Initializing MCTS AI...")
    mcts_ai = MCTSAI(player_number=2, simulations=500, exploration_constant=1.4, debug=True)
    print("MCTS AI initialized with 500 simulations")
    
    # Initialize AlphaZero
    print("Initializing AlphaZero...")
    state_encoder = StateEncoder()
    
    # Try to load trained model, otherwise create new one
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading trained AlphaZero model from {model_path}")
        az_model = load_trained_model(model_path)
    else:
        print("No trained AlphaZero model found. Creating new untrained model.")
        az_model = create_model(input_size=state_encoder.get_feature_size())
    
    # Create AlphaZero MCTS
    az_mcts = AlphaZeroMCTS(
        neural_net=az_model,
        encoder=state_encoder,
        simulations=400,
        c_puct=1.5,
        use_value_net=True,
        use_heuristic_fallback=True
    )
    print("AlphaZero initialized")

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


@app.route('/api/ping', methods=['GET', 'POST'])
def ping():
    """Simple ping endpoint to test connectivity"""
    print("PING received!")
    return jsonify({'status': 'pong', 'timestamp': str(__import__('time').time())})

@app.route('/api/mcts_suggestions', methods=['POST'])
def mcts_suggestions():
    """
    Get top N move suggestions from MCTS AI.
    Returns ranked list of best moves with confidence scores.
    """
    global mcts_ai
    
    print("="*50)
    print("Received /api/mcts_suggestions request")
    print(f"mcts_ai initialized: {mcts_ai is not None}")
    
    if not mcts_ai:
        print("ERROR: MCTS AI not initialized")
        return jsonify({'error': 'MCTS AI not initialized'}), 500
    
    try:
        data = request.get_json()
        print(f"Request data received: {data.keys() if data else 'None'}")
        points = data.get('points', [])
        edges = data.get('edges', [])
        claimed_edges = data.get('claimed_edges', {})
        current_player = data.get('current_player', 2)
        player1_score = data.get('player1_score', 0)
        player2_score = data.get('player2_score', 0)
        top_n = data.get('top_n', 5)  # Number of suggestions to return
        simulations = data.get('simulations', 100)  # Default to 100 for speed
        
        print(f"Processing request: edges={len(edges)}, claimed={len(claimed_edges)}, simulations={simulations}")
        
        # Cap simulations for responsiveness (max 2000)
        simulations = min(simulations, 2000)
        
        # Update simulation count if provided
        if simulations != mcts_ai.simulations:
            mcts_ai.set_simulations(simulations)
        
        print(f"Creating game state snapshot...")
        # Create game state snapshot
        state = GameStateSnapshot(
            points=points,
            edges=edges,
            claimed_edges=claimed_edges,
            current_player=current_player,
            player1_score=player1_score,
            player2_score=player2_score
        )
        
        print(f"Running MCTS with {simulations} simulations...")
        # Get top N suggestions
        suggestions = mcts_ai.get_top_moves(state, top_n=top_n)
        print(f"MCTS returned {len(suggestions)} suggestions")
        
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


# ==================== ALPHAZERO ENDPOINTS ====================

@app.route('/api/az_move', methods=['POST'])
def az_move():
    """API endpoint for AlphaZero AI to make a move"""
    global az_mcts
    
    if not az_mcts:
        return jsonify({'error': 'AlphaZero not initialized'}), 500
    
    try:
        data = request.get_json()
        points = data.get('points', [])
        edges = data.get('edges', [])
        claimed_edges = data.get('claimed_edges', {})
        current_player = data.get('current_player', 2)
        player1_score = data.get('player1_score', 0)
        player2_score = data.get('player2_score', 0)
        simulations = data.get('simulations', 400)
        
        # Create game state
        state = GameStateSnapshot(
            points=points,
            edges=edges,
            claimed_edges=claimed_edges,
            current_player=current_player,
            player1_score=player1_score,
            player2_score=player2_score
        )
        
        # Update simulations if needed
        if simulations != az_mcts.simulations:
            az_mcts.simulations = simulations
        
        # Get best move
        action = az_mcts.get_best_move(state)
        
        if action == -1:
            return jsonify({'error': 'No valid moves'}), 400
        
        selected_edge = edges[action]
        
        # Simulate for scores
        temp_state = state.clone()
        temp_state.apply_action(action)
        
        return jsonify({
            'status': 'success',
            'edge_index': action,
            'edge': selected_edge,
            'simulations_run': az_mcts.simulations,
            'new_scores': {
                'player1': temp_state.player1_score,
                'player2': temp_state.player2_score
            },
            'is_terminal': temp_state.is_terminal()
        })
        
    except Exception as e:
        print(f"Error in AZ move: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/az_suggestions', methods=['POST'])
def az_suggestions():
    """Get top N move suggestions from AlphaZero MCTS"""
    global az_mcts
    
    if not az_mcts:
        return jsonify({'error': 'AlphaZero not initialized'}), 500
    
    try:
        data = request.get_json()
        points = data.get('points', [])
        edges = data.get('edges', [])
        claimed_edges = data.get('claimed_edges', {})
        current_player = data.get('current_player', 2)
        player1_score = data.get('player1_score', 0)
        player2_score = data.get('player2_score', 0)
        top_n = data.get('top_n', 5)
        simulations = data.get('simulations', 400)
        
        # Create game state
        state = GameStateSnapshot(
            points=points,
            edges=edges,
            claimed_edges=claimed_edges,
            current_player=current_player,
            player1_score=player1_score,
            player2_score=player2_score
        )
        
        # Update simulations
        az_mcts.simulations = simulations
        
        # Get suggestions
        suggestions = az_mcts.get_top_moves(state, top_n=top_n)
        
        return jsonify({
            'status': 'success',
            'suggestions': suggestions,
            'total_simulations': az_mcts.simulations,
            'current_player': current_player,
            'ai_type': 'alphazero'
        })
        
    except Exception as e:
        print(f"Error in AZ suggestions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/az_status', methods=['GET'])
def az_status():
    """Get AlphaZero status and model info"""
    global az_model, az_mcts, state_encoder
    
    if not az_model or not az_mcts:
        return jsonify({'error': 'AlphaZero not initialized'}), 500
    
    # Count model parameters
    total_params = sum(p.numel() for p in az_model.parameters())
    trainable_params = sum(p.numel() for p in az_model.parameters() if p.requires_grad)
    
    return jsonify({
        'status': 'success',
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': state_encoder.get_feature_size(),
            'num_edges': az_model.num_edges,
            'hidden_sizes': [256, 256, 128]
        },
        'mcts_config': {
            'simulations': az_mcts.simulations,
            'c_puct': az_mcts.c_puct,
            'use_value_net': az_mcts.use_value_net,
            'use_heuristic_fallback': az_mcts.use_heuristic_fallback
        }
    })


@app.route('/api/az_train', methods=['POST'])
def az_train():
    """Start AlphaZero training"""
    try:
        data = request.get_json() or {}
        num_iterations = data.get('iterations', 10)
        games_per_iteration = data.get('games_per_iteration', 10)
        num_simulations = data.get('simulations', 200)
        
        # Run training in background (non-blocking)
        import threading
        
        def train_thread():
            from train import IterativeTrainer
            
            trainer = IterativeTrainer(
                model=az_model,
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration,
                num_simulations=num_simulations,
                checkpoint_dir='models'
            )
            trainer.run(verbose=True)
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'training_started',
            'config': {
                'iterations': num_iterations,
                'games_per_iteration': games_per_iteration,
                'simulations': num_simulations
            }
        })
        
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize AI when starting the app
    initialize_ai()
    app.run(debug=True, host='0.0.0.0', port=5000)