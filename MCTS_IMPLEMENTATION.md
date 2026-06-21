# MCTS AI Implementation for Voronoi Connect 4

## Overview

This project now features a **Monte Carlo Tree Search (MCTS)** AI opponent with **heuristic-guided rollouts** for the Voronoi Connect 4 game. The AI uses sophisticated game tree search combined with intelligent move selection to make strategic decisions.

## Architecture

### Backend (Python)

#### Core Files
- **`mcts_ai.py`** - Main MCTS implementation with:
  - `GameStateSnapshot` - Immutable game state for simulation
  - `MCTSNode` - Tree node with UCB1 selection
  - `HeuristicRolloutPolicy` - Smart move selection during rollouts
  - `MCTSAI` - Main AI controller

- **`app.py`** - Flask backend with MCTS endpoints:
  - `/api/mcts_move` - Get AI's next move
  - `/api/mcts_config` - Configure MCTS parameters
  - `/api/mcts_stats` - View AI performance statistics

### Frontend (JavaScript)

- **`ai-game.js`** - Updated to call MCTS backend API:
  - `makeAIMove()` - Async call to MCTS endpoint
  - `fallbackAIMove()` - Simple heuristic if MCTS fails

## How MCTS Works

### 1. Game State Snapshot System

```python
class GameStateSnapshot:
    - points: List of Voronoi points
    - edges: List of selectable edges
    - claimed_edges: Dictionary mapping edge_id -> player
    - current_player: 1 or 2
    - player1_score, player2_score: Current scores
    
    Methods:
    - clone() - Deep copy for simulation
    - get_valid_actions() - Unclaimed edge indices
    - apply_action(action) - Claim edge, update scores
    - is_terminal() - All edges claimed?
    - get_winner() - 1, 2, or 0 (draw)
```

### 2. MCTS Algorithm (4 Phases)

#### Phase 1: Selection
- Traverse tree using **UCB1 formula**:
  ```
  UCB = (total_reward / visits) + c * sqrt(log(parent_visits) / child_visits)
  ```
- `c = 1.4` (exploration constant)
- Continue until reaching unexpanded node

#### Phase 2: Expansion
- Create child node for one untried action
- Add to tree structure

#### Phase 3: Simulation (Rollout)
- Play random game to completion
- Use **heuristic-guided move selection** (not purely random)
- See Heuristic Policy below

#### Phase 4: Backpropagation
- Update statistics for all visited nodes
- Propagate reward up the tree

### 3. Heuristic Rollout Policy

During simulation, moves are prioritized using heuristics:

**Priority 1 (Highest):** Complete polygons immediately
- Score: +100

**Priority 2:** Avoid moves that let opponent complete polygons next turn  
- Score: -50

**Priority 3:** Connect to already-owned edges
- Score: +5 per connection

**Priority 4:** Increase future polygon potential
- Score: +3 per potential vertex

**Priority 5 (Lowest):** Random factor
- Score: 0-2 random value

### 4. Reward Function

```python
if AI wins:     reward = +1.0
if AI loses:    reward = -1.0
if draw:        reward = 0.0

# Plus score differential bonus:
reward += (ai_score - opponent_score) / max_possible_score * 0.1
```

## Gameplay Flow

### Human vs AI Mode

1. **Enable AI Mode** - Toggle "AI Mode" switch in UI
2. **Generate Board** - Create Voronoi diagram
3. **Your Turn (Player 1)** - Click any edge to claim it
4. **AI Turn (Player 2)** - Backend runs MCTS:
   - 500 simulations
   - ~1-2 second thinking time
   - Displays "MCTS AI is thinking..."
5. **AI Claims Edge** - Automatically selects and claims edge
6. **Repeat** - Until all edges claimed

### AI Configuration

Adjust MCTS strength via API:

```bash
# Increase difficulty (more simulations)
POST /api/mcts_config
{
    "simulations": 1000,
    "exploration_constant": 1.4,
    "debug": true
}

# View stats
GET /api/mcts_stats
```

## Scoring Integration

The MCTS AI respects the existing scoring system:
- **Non-polygon edge:** 1 point per claimed edge not part of a completed polygon
- **Polygon edge:** 4 points per claimed edge that is part of a completed polygon

## Performance Optimization

1. **Cached Adjacency Maps** - Precompute vertex connections
2. **Efficient Polygon Detection** - DFS with cycle normalization
3. **Parallel Consideration** - Could add multithreading for rollouts
4. **Configurable Simulations** - Trade thinking time for strength

## Debugging Tools

When `debug: true`, console shows:
```
==================================================
MCTS SEARCH STATISTICS
==================================================
simulations: 500
time_seconds: 1.234
selected_move: 15
root_visits: 500
best_visits: 127
best_reward: 0.73
==================================================
```

## Testing Verification

The AI implementation was verified for:
- ✅ Never selects occupied edges
- ✅ Scores calculated correctly  
- ✅ Polygons detected properly
- ✅ Simulations don't affect real game state
- ✅ Performance acceptable (< 2 seconds per move)
- ✅ Fallback to heuristic if backend fails
- ✅ Human vs Player mode still works

## Future Enhancements

1. **Training Mode** - Self-play to improve heuristics
2. **Opening Book** - Pre-computed good first moves
3. **Progressive Widening** - Deeper search for promising moves
4. **Neural Network Policy** - Learned move priors
5. **Multi-threading** - Parallel simulations

## API Reference

### POST /api/mcts_move

Request body:
```json
{
    "points": [...],
    "edges": [...],
    "claimed_edges": {"x1,y1-x2,y2": 1, ...},
    "current_player": 2,
    "player1_score": 10,
    "player2_score": 8,
    "simulations": 500
}
```

Response:
```json
{
    "status": "success",
    "edge_index": 15,
    "edge": {"id": "...", "x1": 0, "y1": 0, "x2": 100, "y2": 100},
    "simulations_run": 500,
    "debug_stats": {...},
    "new_scores": {"player1": 10, "player2": 12},
    "is_terminal": false,
    "winner": null
}
```

### POST /api/mcts_config

Request body:
```json
{
    "simulations": 1000,
    "exploration_constant": 1.4,
    "debug": false
}
```

### GET /api/mcts_stats

Response:
```json
{
    "status": "success",
    "last_search_stats": {...},
    "current_config": {
        "simulations": 500,
        "exploration_constant": 1.4,
        "debug": true
    }
}
```

## Files Modified/Created

### New Files
- `mcts_ai.py` - Core MCTS implementation (635 lines)
- `MCTS_IMPLEMENTATION.md` - This documentation

### Modified Files
- `app.py` - Added MCTS endpoints and integration
- `static/js/ai-game.js` - Frontend MCTS API calls
- `requirements.txt` - Dependencies unchanged (pure Python)

## Running the Application

```bash
# Start the Flask server
python app.py

# Open browser
http://localhost:5000

# Enable AI Mode and play!
```

The MCTS AI provides a challenging opponent that makes strategically sound decisions based on probabilistic game tree search and intelligent heuristics.
