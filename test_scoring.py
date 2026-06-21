#!/usr/bin/env python3
"""
Unit test for canonical scoring rule verification.

Canonical rule (matches ai-game.js):
- Non-polygon edges: 1 point each
- Polygon edges: 4 points each

Tests both incremental scoring (GameStateSnapshot._calculate_score_change)
and full-recompute scoring (FixedVoronoiRLEnvironment._calculate_player_score).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts_ai import GameStateSnapshot
from fixed_rl_environment import FixedVoronoiRLEnvironment
from fixed_board_config import FIXED_POINTS, FIXED_EDGES


# Known quadrilateral on the fixed board:
# Edge 0: (150,150) -> (400,150)
# Edge 2: (150,150) -> (250,300)
# Edge 4: (400,150) -> (400,300)
# Edge 5: (250,300) -> (400,300)
# Vertices: (150,150), (400,150), (400,300), (250,300) — 4 edges, 4 vertices
KNOWN_POLYGON_EDGES = [0, 2, 4, 5]
POLYGON_SIZE = 4  # 4 edges


def test_non_polygon_edge_scoring():
    """Test that a single claimed edge (no polygon) scores 1 point."""
    
    print("=" * 60)
    print("TEST 1: Non-polygon edge scoring")
    print("=" * 60)
    
    state = GameStateSnapshot(
        points=FIXED_POINTS,
        edges=FIXED_EDGES,
        claimed_edges={},
        current_player=1,
        player1_score=0,
        player2_score=0
    )
    
    # Claim just one edge
    state.apply_action(0)
    
    expected = 1
    actual = state.player1_score
    
    print("  Claimed 1 edge (no polygon)")
    print("  Expected score: %d" % expected)
    print("  Actual score:   %d" % actual)
    
    if actual == expected:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


def test_incremental_scoring():
    """Test that GameStateSnapshot._calculate_score_change gives correct
    retroactive polygon bonus when a polygon closes."""
    
    print("\n" + "=" * 60)
    print("TEST 2: Incremental scoring (GameStateSnapshot)")
    print("  Polygon: 4-edge quad (edges %s)" % str(KNOWN_POLYGON_EDGES))
    print("=" * 60)
    
    state = GameStateSnapshot(
        points=FIXED_POINTS,
        edges=FIXED_EDGES,
        claimed_edges={},
        current_player=1,
        player1_score=0,
        player2_score=0
    )
    
    # Claim all polygon edges for player 1, with filler moves for player 2
    filler_edges = [i for i in range(len(FIXED_EDGES)) if i not in KNOWN_POLYGON_EDGES]
    filler_idx = 0
    
    for i, action in enumerate(KNOWN_POLYGON_EDGES):
        edge = state.edges[action]
        print("  Move %d: Player 1 claims edge %d (%s)" % (i+1, action, edge['id']))
        state.apply_action(action)
        print("    Player 1 score: %d" % state.player1_score)
        
        # Player 2 filler move (apply_action switches player)
        if i < len(KNOWN_POLYGON_EDGES) - 1 and filler_idx < len(filler_edges):
            filler = filler_edges[filler_idx]
            print("  Move %db: Player 2 claims edge %d (filler)" % (i+1, filler))
            state.apply_action(filler)
            filler_idx += 1
    
    expected = POLYGON_SIZE * 4  # 4 edges * 4 points = 16
    actual = state.player1_score
    
    print("\n  Expected player 1 score: %d (= %d edges x 4 pts)" % (expected, POLYGON_SIZE))
    print("  Actual player 1 score:   %d" % actual)
    
    if actual == expected:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


def test_full_recompute_scoring():
    """Test that FixedVoronoiRLEnvironment._calculate_player_score gives
    correct scores using full recompute."""
    
    print("\n" + "=" * 60)
    print("TEST 3: Full recompute scoring (FixedVoronoiRLEnvironment)")
    print("  Polygon: 4-edge quad (edges %s)" % str(KNOWN_POLYGON_EDGES))
    print("=" * 60)
    
    env = FixedVoronoiRLEnvironment()
    env.reset()
    
    filler_edges = [i for i in range(len(env.edges)) if i not in KNOWN_POLYGON_EDGES]
    filler_idx = 0
    
    for i, action in enumerate(KNOWN_POLYGON_EDGES):
        edge = env.edges[action]
        print("  Move %d: Player %d claims edge %d (%s)" % (i+1, env.current_player, action, edge['id']))
        next_state, reward, done, info = env.step(action)
        print("    Reward: %.1f, P1: %d, P2: %d" % (reward, info['player1_score'], info['player2_score']))
        
        # Player 2 filler
        if i < len(KNOWN_POLYGON_EDGES) - 1 and not done and filler_idx < len(filler_edges):
            filler = filler_edges[filler_idx]
            print("  Move %db: Player %d claims edge %d (filler)" % (i+1, env.current_player, filler))
            next_state, reward, done, info = env.step(filler)
            filler_idx += 1
    
    expected = POLYGON_SIZE * 4  # 4 edges * 4 pts = 16
    actual = env.player1_score
    
    print("\n  Expected player 1 score: %d (= %d edges x 4 pts)" % (expected, POLYGON_SIZE))
    print("  Actual player 1 score:   %d" % actual)
    
    if actual == expected:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


def test_mixed_scoring():
    """Test scoring with both polygon and non-polygon edges."""
    
    print("\n" + "=" * 60)
    print("TEST 4: Mixed polygon + non-polygon scoring")
    print("=" * 60)
    
    state = GameStateSnapshot(
        points=FIXED_POINTS,
        edges=FIXED_EDGES,
        claimed_edges={},
        current_player=1,
        player1_score=0,
        player2_score=0
    )
    
    # Player 1 claims the polygon edges + 1 extra non-polygon edge
    extra_edge = [i for i in range(len(FIXED_EDGES)) if i not in KNOWN_POLYGON_EDGES][0]
    all_p1_edges = KNOWN_POLYGON_EDGES + [extra_edge]
    
    filler_edges = [i for i in range(len(FIXED_EDGES)) 
                    if i not in all_p1_edges]
    filler_idx = 0
    
    for i, action in enumerate(all_p1_edges):
        state.apply_action(action)
        if i < len(all_p1_edges) - 1 and filler_idx < len(filler_edges):
            state.apply_action(filler_edges[filler_idx])
            filler_idx += 1
    
    # 4 polygon edges * 4pts + 1 non-polygon edge * 1pt = 17
    expected = POLYGON_SIZE * 4 + 1
    actual = state.player1_score
    
    print("  Player 1 has %d polygon edges + 1 non-polygon edge" % POLYGON_SIZE)
    print("  Expected score: %d (= %d*4 + 1*1)" % (expected, POLYGON_SIZE))
    print("  Actual score:   %d" % actual)
    
    if actual == expected:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


if __name__ == '__main__':
    results = []
    results.append(test_non_polygon_edge_scoring())
    results.append(test_incremental_scoring())
    results.append(test_full_recompute_scoring())
    results.append(test_mixed_scoring())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print("RESULTS: %d/%d tests passed" % (passed, total))
    if all(results):
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)
    
    sys.exit(0 if all(results) else 1)
