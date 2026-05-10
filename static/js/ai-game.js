class VoronoiAIGame {
    constructor() {
        this.svg = d3.select('#voronoiSvg');
        this.width = 800;
        this.height = 600;
        this.points = [];
        this.voronoi = null;
        this.edges = [];
        this.claimedEdges = new Map();
        this.currentPlayer = 1;
        this.player1Score = 0;
        this.player2Score = 0;
        this.gameActive = false;
        this.timerActive = false;
        this.timerInterval = null;
        this.timeRemaining = 60;
        this.tooltip = null;
        this.aiEnabled = false;
        this.aiThinking = false;
        
        // AI Suggestion Mode properties
        this.aiMoveHistory = [];  // Track AI's move history
        this.currentSuggestions = [];  // Current AI suggestions
        this.suggestionMode = 'advisor';  // 'advisor' = show suggestions, 'auto' = AI plays
        this.aiSimulationCount = 100;  // Default to 100 for responsiveness
        
        this.initializeEventListeners();
        this.createTooltip();
    }

    initializeEventListeners() {
        // Generate button
        document.getElementById('generateBtn').addEventListener('click', () => {
            this.generateVoronoi();
        });

        // AI toggle
        const aiToggle = document.getElementById('aiToggle');
        if (aiToggle) {
            aiToggle.addEventListener('change', async (e) => {
                this.aiEnabled = e.target.checked;
                this.toggleAISidebar(this.aiEnabled);
                this.resetGame();
                this.aiMoveHistory = [];  // Clear history on new game
                this.updateAIMoveHistoryDisplay();
                
                if (this.aiEnabled) {
                    this.showStatus('AI Advisor mode enabled! Testing backend connection...', 'info');
                    // Test backend connectivity
                    const connected = await this.testBackendConnection();
                    if (connected) {
                        this.showStatus('AI Advisor mode enabled! You are Player 1. AI will suggest moves for Player 2.', 'info');
                    } else {
                        this.showStatus('AI mode enabled but backend connection failed!', 'error');
                    }
                } else {
                    this.showStatus('AI mode disabled.', 'info');
                }
            });
        }

        // AI Settings - Simulation count
        const simulationCount = document.getElementById('simulationCount');
        if (simulationCount) {
            simulationCount.addEventListener('change', (e) => {
                this.aiSimulationCount = parseInt(e.target.value);
                this.showStatus(`AI simulations set to ${this.aiSimulationCount}`, 'info');
            });
        }

        // Refresh suggestions button
        const refreshBtn = document.getElementById('refreshSuggestions');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                if (this.aiEnabled && this.currentPlayer === 2 && this.gameActive) {
                    this.fetchAISuggestions();
                }
            });
        }

        // Timer toggle
        document.getElementById('timerToggle').addEventListener('change', (e) => {
            this.timerActive = e.target.checked;
            if (this.timerActive && this.gameActive) {
                this.startTimer();
            } else {
                this.stopTimer();
            }
        });

        // Reset and new game buttons
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetGame();
        });

        document.getElementById('newGameBtn').addEventListener('click', () => {
            this.newGame();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'r' || e.key === 'R') {
                this.resetGame();
            } else if (e.key === 'n' || e.key === 'N') {
                this.newGame();
            } else if (e.key === 't' || e.key === 'T') {
                const timerToggle = document.getElementById('timerToggle');
                if (timerToggle) timerToggle.click();
            }
        });
    }

    createTooltip() {
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);
    }

    generateVoronoi() {
        const pointCount = parseInt(document.getElementById('pointCount').value);
        if (pointCount < 3) {
            this.showStatus('Please enter at least 3 points', 'error');
            return;
        }

        this.clearSVG();
        this.generatePoints(pointCount);
        this.createVoronoiDiagram();
        this.setupGame();
        this.showStatus(this.aiEnabled ? 'Game started! You are Player 1 (Red).' : 'Game started! Click on edges to claim them.', 'info');
    }

    generatePoints(count) {
        this.points = [];
        const margin = 50;
        
        for (let i = 0; i < count; i++) {
            this.points.push({
                x: margin + Math.random() * (this.width - 2 * margin),
                y: margin + Math.random() * (this.height - 2 * margin),
                id: i
            });
        }
    }

    createVoronoiDiagram() {
        this.voronoi = d3.Delaunay.from(this.points, d => d.x, d => d.y).voronoi([0, 0, this.width, this.height]);
        
        // Draw Voronoi cells
        this.svg.selectAll('.voronoi-cell')
            .data(this.points)
            .enter()
            .append('path')
            .attr('class', 'voronoi-cell')
            .attr('d', (d, i) => this.voronoi.renderCell(i))
            .on('mouseover', function(event, d) {
                d3.select(this).style('fill', 'rgba(102, 126, 234, 0.1)');
            })
            .on('mouseout', function(event, d) {
                d3.select(this).style('fill', 'none');
            });

        // Draw Voronoi edges
        this.drawEdges();

        // Draw points
        this.svg.selectAll('.voronoi-point')
            .data(this.points)
            .enter()
            .append('circle')
            .attr('class', 'voronoi-point')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .on('mouseover', (event, d) => {
                this.tooltip
                    .style('opacity', 1)
                    .html(`Point ${d.id + 1}<br/>(${Math.round(d.x)}, ${Math.round(d.y)})`)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => {
                this.tooltip.style('opacity', 0);
            });
    }

    drawEdges() {
        this.edges = [];
        const edgeMap = new Map();

        // Extract edges from Voronoi diagram
        for (let i = 0; i < this.points.length; i++) {
            const cell = this.voronoi.renderCell(i);
            if (!cell) continue;

            const segments = this.getPathSegments(cell);

            for (let j = 0; j < segments.length; j++) {
                const segment = segments[j];
                const edgeId = this.getEdgeId(segment);
                
                if (!edgeMap.has(edgeId)) {
                    edgeMap.set(edgeId, {
                        id: edgeId,
                        x1: segment.x1,
                        y1: segment.y1,
                        x2: segment.x2,
                        y2: segment.y2,
                        claimed: false,
                        player: null
                    });
                    this.edges.push(edgeMap.get(edgeId));
                }
            }
        }

        // Draw edges
        this.svg.selectAll('.voronoi-edge')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'voronoi-edge')
            .attr('x1', d => d.x1)
            .attr('y1', d => d.y1)
            .attr('x2', d => d.x2)
            .attr('y2', d => d.y2)
            .on('click', async (event, d) => {
                if (!this.aiEnabled || this.currentPlayer === 1) {
                    await this.claimEdge(d);
                }
            })
            .on('mouseover', (event, d) => {
                if (!d.claimed && (!this.aiEnabled || this.currentPlayer === 1)) {
                    d3.select(event.target)
                        .style('stroke', this.currentPlayer === 1 ? '#e74c3c' : '#3498db')
                        .style('stroke-width', 4);
                }
            })
            .on('mouseout', (event, d) => {
                if (!d.claimed) {
                    d3.select(event.target)
                        .style('stroke', '#999')
                        .style('stroke-width', 2);
                }
            });
    }

    getPathSegments(pathString) {
        const segments = [];
        const commands = pathString.match(/[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*/g) || [];
        
        let currentX = 0, currentY = 0;
        
        for (let command of commands) {
            const type = command[0];
            const coords = command.slice(1).trim().split(/[\s,]+/).map(Number).filter(n => !isNaN(n));
            
            if (type === 'M' || type === 'm') {
                currentX = coords[0] + (type === 'm' ? currentX : 0);
                currentY = coords[1] + (type === 'm' ? currentY : 0);
            } else if (type === 'L' || type === 'l') {
                const x = coords[0] + (type === 'l' ? currentX : 0);
                const y = coords[1] + (type === 'l' ? currentY : 0);
                segments.push({
                    x1: currentX,
                    y1: currentY,
                    x2: x,
                    y2: y
                });
                currentX = x;
                currentY = y;
            } else if (type === 'Z' || type === 'z') {
                if (segments.length > 0) {
                    segments.push({
                        x1: currentX,
                        y1: currentY,
                        x2: segments[0].x1,
                        y2: segments[0].y1
                    });
                }
            }
        }
        
        return segments;
    }

    getEdgeId(segment) {
        const x1 = Math.round(segment.x1 * 100) / 100;
        const y1 = Math.round(segment.y1 * 100) / 100;
        const x2 = Math.round(segment.x2 * 100) / 100;
        const y2 = Math.round(segment.y2 * 100) / 100;
        
        const coords = [[x1, y1], [x2, y2]].sort((a, b) => {
            if (a[0] !== b[0]) return a[0] - b[0];
            return a[1] - b[1];
        });
        
        return `${coords[0][0]},${coords[0][1]}-${coords[1][0]},${coords[1][1]}`;
    }

    async claimEdge(edge) {
        if (!this.gameActive || edge.claimed || this.aiThinking) return;

        // Track score before move for AI history
        const previousScore = this.currentPlayer === 2 ? this.player2Score : this.player1Score;

        edge.claimed = true;
        edge.player = this.currentPlayer;
        this.claimedEdges.set(edge.id, this.currentPlayer);

        // Update visual appearance
        const edgeElement = this.svg.selectAll('.voronoi-edge')
            .filter(d => d.id === edge.id);
        
        edgeElement
            .classed(`claimed-player${this.currentPlayer}`, true)
            .style('stroke', this.currentPlayer === 1 ? '#e74c3c' : '#3498db')
            .style('stroke-width', 4);

        // Update score
        this.updateScore();
        
        // Track AI move if it's Player 2 and AI mode is enabled
        if (this.currentPlayer === 2 && this.aiEnabled) {
            const newScore = this.player2Score;
            const scoreGain = newScore - previousScore;
            this.addMoveToAIHistory(edge, scoreGain);
        }
        
        // Clear AI suggestions after any move
        this.clearSuggestions();
        
        await this.switchPlayer();
        this.checkGameEnd();
    }

    async makeAIMove() {
        if (!this.aiEnabled || this.currentPlayer !== 2 || this.aiThinking) return;
        
        this.aiThinking = true;
        this.showStatus('MCTS AI is thinking... Running 500 simulations', 'info');
        
        try {
            // Prepare game state for MCTS AI
            const gameState = {
                points: this.points,
                edges: this.edges,
                claimed_edges: Object.fromEntries(this.claimedEdges),
                current_player: this.currentPlayer,
                player1_score: this.player1Score,
                player2_score: this.player2Score,
                simulations: 500  // Configurable number of MCTS simulations
            };
            
            // Call MCTS backend API
            const response = await fetch('/api/mcts_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(gameState)
            });
            
            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                // Get the selected edge index
                const edgeIndex = result.edge_index;
                const selectedEdge = this.edges[edgeIndex];
                
                if (selectedEdge && !selectedEdge.claimed) {
                    // Show debug info if available
                    if (result.debug_stats) {
                        console.log('MCTS Debug Stats:', result.debug_stats);
                    }
                    
                    // Claim the edge
                    this.claimEdge(selectedEdge);
                    
                    // Show AI decision info
                    if (result.new_scores) {
                        const scoreDiff = result.new_scores.player2 - this.player2Score;
                        if (scoreDiff > 1) {
                            this.showStatus(`AI completed a polygon! +${scoreDiff} points`, 'info');
                        }
                    }
                } else {
                    console.error('MCTS selected invalid edge:', result);
                    this.showStatus('AI error: Invalid move selected', 'error');
                }
            } else {
                console.error('MCTS API error:', result.error);
                this.showStatus('AI error: ' + (result.error || 'Unknown error'), 'error');
                
                // Fallback to simple heuristic if MCTS fails
                await this.fallbackAIMove();
            }
            
        } catch (error) {
            console.error('Error calling MCTS API:', error);
            this.showStatus('AI connection error, using fallback', 'warning');
            
            // Fallback to simple heuristic
            await this.fallbackAIMove();
        }
        
        this.aiThinking = false;
    }

    async fallbackAIMove() {
        // Fallback simple heuristic when MCTS fails
        const availableEdges = this.edges.filter(edge => !edge.claimed);
        
        if (availableEdges.length > 0) {
            // Simple strategy: prioritize edges that might form polygons
            let selectedEdge = this.selectBestEdgeFallback(availableEdges);
            await this.claimEdge(selectedEdge);
        }
    }

    selectBestEdgeFallback(availableEdges) {
        // Fallback heuristic-based selection
        const player1Edges = Array.from(this.claimedEdges.entries())
            .filter(([edgeId, player]) => player === 1)
            .map(([edgeId]) => edgeId);
        
        const scoredEdges = availableEdges.map(edge => {
            let score = Math.random();
            
            for (const playerEdgeId of player1Edges) {
                if (this.edgesShareVertex(edge.id, playerEdgeId)) {
                    score += 2;
                }
            }
            
            if (this.couldFormPolygon(edge.id, 2)) {
                score += 3;
            }
            
            return { edge, score };
        });
        
        scoredEdges.sort((a, b) => b.score - a.score);
        return scoredEdges[0].edge;
    }

    edgesShareVertex(edgeId1, edgeId2) {
        const edge1 = this.edges.find(e => e.id === edgeId1);
        const edge2 = this.edges.find(e => e.id === edgeId2);
        
        if (!edge1 || !edge2) return false;
        
        const vertices1 = [`${edge1.x1},${edge1.y1}`, `${edge1.x2},${edge1.y2}`];
        const vertices2 = [`${edge2.x1},${edge2.y1}`, `${edge2.x2},${edge2.y2}`];
        
        return vertices1.some(v => vertices2.includes(v));
    }

    couldFormPolygon(edgeId, player) {
        // Simplified check - just count connected edges
        const playerEdges = Array.from(this.claimedEdges.entries())
            .filter(([eid, p]) => p === player)
            .map(([eid]) => eid);
        
        if (playerEdges.length < 2) return false;
        
        // Check if this edge connects to existing player edges
        let connections = 0;
        for (const playerEdgeId of playerEdges) {
            if (this.edgesShareVertex(edgeId, playerEdgeId)) {
                connections++;
            }
        }
        
        return connections >= 2; // Could potentially form a polygon
    }

    updateScore() {
        this.player1Score = this.calculatePlayerScore(1);
        this.player2Score = this.calculatePlayerScore(2);

        document.getElementById('player1Score').textContent = this.player1Score;
        document.getElementById('player2Score').textContent = this.player2Score;
    }

    calculatePlayerScore(player) {
        const playerEdges = Array.from(this.claimedEdges.entries())
            .filter(([edgeId, edgePlayer]) => edgePlayer === player)
            .map(([edgeId]) => edgeId);

        const polygons = this.findCompletePolygons(playerEdges);
        
        const edgesInPolygons = new Set();
        for (const polygon of polygons) {
            for (let i = 0; i < polygon.length; i++) {
                const v1 = polygon[i];
                const v2 = polygon[(i + 1) % polygon.length];
                const coords = [v1, v2].sort();
                edgesInPolygons.add(`${coords[0]}-${coords[1]}`);
            }
        }
        
        let score = playerEdges.filter(edgeId => !edgesInPolygons.has(edgeId)).length;
        score += edgesInPolygons.size * 4;

        return score;
    }

    findCompletePolygons(playerEdges) {
        const polygons = [];
        if (playerEdges.length < 3) return polygons;
        
        const edgeMap = new Map();
        
        for (const edgeId of playerEdges) {
            const [start, end] = edgeId.split('-');
            const [x1, y1] = start.split(',').map(Number);
            const [x2, y2] = end.split(',').map(Number);
            
            const v1 = `${x1},${y1}`;
            const v2 = `${x2},${y2}`;
            
            if (!edgeMap.has(v1)) edgeMap.set(v1, []);
            if (!edgeMap.has(v2)) edgeMap.set(v2, []);
            
            edgeMap.get(v1).push(v2);
            edgeMap.get(v2).push(v1);
        }

        const visited = new Set();
        const cycles = [];
        
        for (const vertex of edgeMap.keys()) {
            if (edgeMap.get(vertex).length < 2 || visited.has(vertex)) continue;
            
            const cyclesFromVertex = this.findCycles(vertex, edgeMap, visited);
            cycles.push(...cyclesFromVertex);
        }
        
        return cycles;
    }

    findCycles(startVertex, edgeMap, globallyVisited) {
        const foundCycles = new Set();
        const visitedInPath = new Set();
        
        const dfs = (current, path, visited) => {
            visited.add(current);
            path.push(current);
            
            const neighbors = edgeMap.get(current) || [];
            
            for (const neighbor of neighbors) {
                if (!path.includes(neighbor)) {
                    dfs(neighbor, path, visited);
                } else if (path.length > 2 && neighbor === path[0]) {
                    const cycle = [...path];
                    const minIndex = cycle.indexOf(cycle.reduce((min, v) => v < min ? v : min));
                    const normalizedCycle = [...cycle.slice(minIndex), ...cycle.slice(0, minIndex)];
                    const cycleKey = normalizedCycle.join('-');
                    
                    foundCycles.add(cycleKey);
                }
            }
            
            path.pop();
            visited.delete(current);
        };
        
        dfs(startVertex, [], visitedInPath);
        globallyVisited.add(startVertex);
        
        return Array.from(foundCycles).map(cycleKey => cycleKey.split('-'));
    }

    async switchPlayer() {
        this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
        document.getElementById('currentPlayerText').textContent = 
            this.aiEnabled ? 
                (this.currentPlayer === 1 ? "Your Turn" : "AI's Turn - View Suggestions →") :
                `Player ${this.currentPlayer}'s Turn`;
        
        if (this.timerActive) {
            this.startTimer();
        }
        
        // If AI mode enabled and it's Player 2's turn, fetch suggestions
        console.log('Switching to player:', this.currentPlayer, 'AI enabled:', this.aiEnabled, 'Game active:', this.gameActive);
        if (this.aiEnabled && this.currentPlayer === 2 && this.gameActive) {
            console.log('Fetching AI suggestions...');
            await this.fetchAISuggestions();
        } else {
            // Clear suggestions when it's not AI's turn
            this.clearSuggestions();
        }
    }

    startTimer() {
        this.stopTimer();
        this.timeRemaining = 60;
        this.updateTimerDisplay();
        
        this.timerInterval = setInterval(async () => {
            this.timeRemaining--;
            this.updateTimerDisplay();
            
            if (this.timeRemaining <= 0) {
                await this.timeUp();
            }
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    updateTimerDisplay() {
        const minutes = Math.floor(this.timeRemaining / 60);
        const seconds = this.timeRemaining % 60;
        document.getElementById('timerDisplay').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    async timeUp() {
        this.stopTimer();
        await this.switchPlayer();
        this.showStatus(`Time's up! ${this.aiEnabled ? 'AI' : `Player ${this.currentPlayer === 1 ? 2 : 1}`} gets another turn.`, 'warning');
    }

    checkGameEnd() {
        const totalEdges = this.edges.length;
        const claimedEdges = this.claimedEdges.size;
        
        if (claimedEdges === totalEdges) {
            this.endGame();
        }
    }

    endGame() {
        this.gameActive = false;
        this.stopTimer();
        
        let winner;
        if (this.player1Score > this.player2Score) {
            winner = this.aiEnabled ? 'You' : 'Player 1';
        } else if (this.player2Score > this.player1Score) {
            winner = this.aiEnabled ? 'AI' : 'Player 2';
        } else {
            this.showStatus("It's a draw! Both players have the same score.", 'draw');
            return;
        }
        
        this.showStatus(`${winner} win${winner === 'You' ? '' : 's'}! Final Score - P1: ${this.player1Score}, P2: ${this.player2Score}`, 'winner');
    }

    setupGame() {
        this.gameActive = true;
        this.currentPlayer = 1;
        this.player1Score = 0;
        this.player2Score = 0;
        this.claimedEdges.clear();
        this.aiThinking = false;
        
        this.edges.forEach(edge => {
            edge.claimed = false;
            edge.player = null;
        });
        
        this.updateScore();
        document.getElementById('currentPlayerText').textContent = 
            this.aiEnabled ? "Your Turn" : 'Player 1\'s Turn';
        
        if (this.timerActive) {
            this.startTimer();
        }
    }

    resetGame() {
        this.gameActive = false;
        this.stopTimer();
        this.clearSVG();
        this.aiThinking = false;
        this.showStatus('Game reset. Generate a new Voronoi diagram to start playing.', 'info');
    }

    newGame() {
        this.resetGame();
        this.generateVoronoi();
    }

    clearSVG() {
        this.svg.selectAll('*').remove();
    }

    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('gameStatus');
        statusElement.textContent = message;
        statusElement.className = `game-status ${type}`;
        
        if (type === 'info') {
            setTimeout(() => {
                statusElement.textContent = '';
                statusElement.className = 'game-status';
            }, 3000);
        }
    }

    // ==================== AI SIDEBAR & SUGGESTION METHODS ====================

    async testBackendConnection() {
        console.log('Testing backend connection...');
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch('/api/ping', {
                method: 'GET',
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const result = await response.json();
                console.log('Backend ping successful:', result);
                return true;
            } else {
                console.error('Backend ping failed:', response.status);
                return false;
            }
        } catch (error) {
            console.error('Backend connection test failed:', error);
            return false;
        }
    }

    toggleAISidebar(show) {
        const sidebar = document.getElementById('aiSidebar');
        if (sidebar) {
            sidebar.style.display = show ? 'block' : 'none';
        }
    }

    async fetchAISuggestions() {
        console.log('fetchAISuggestions called:', {
            aiEnabled: this.aiEnabled,
            currentPlayer: this.currentPlayer,
            gameActive: this.gameActive,
            edgesCount: this.edges?.length,
            claimedCount: this.claimedEdges?.size
        });
        
        if (!this.aiEnabled || this.currentPlayer !== 2 || !this.gameActive) {
            console.log('Early return - conditions not met');
            return;
        }
        
        this.aiThinking = true;
        this.showStatus('MCTS AI is analyzing... Running simulations', 'info');
        
        // Update UI to show loading state
        const container = document.getElementById('aiSuggestions');
        if (container) {
            container.innerHTML = '<p class="placeholder-text">🤖 AI is thinking...</p>';
        }
        
        try {
            const gameState = {
                points: this.points,
                edges: this.edges,
                claimed_edges: Object.fromEntries(this.claimedEdges),
                current_player: this.currentPlayer,
                player1_score: this.player1Score,
                player2_score: this.player2Score,
                top_n: 5,
                simulations: this.aiSimulationCount
            };
            
            console.log('Sending request to /api/mcts_suggestions with state:', gameState);
            
            // Create AbortController for timeout - scales with simulation count
            const controller = new AbortController();
            const timeoutMs = Math.max(10000, Math.min(300000, this.aiSimulationCount * 150)); // 150ms per sim, min 10s, max 5 minutes
            console.log(`Setting request timeout to ${timeoutMs}ms for ${this.aiSimulationCount} simulations`);
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
            
            const response = await fetch('/api/mcts_suggestions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(gameState),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            console.log('Response status:', response.status);
            console.log('Response headers:', [...response.headers.entries()]);
            
            const result = await response.json();
            console.log('Response result:', result);
            
            if (response.ok && result.status === 'success') {
                console.log('Got suggestions:', result.suggestions);
                this.currentSuggestions = result.suggestions;
                this.displaySuggestions(result.suggestions);
                this.highlightSuggestedEdges(result.suggestions);
                this.showStatus('AI suggestions ready! Click a highlighted edge or suggestion.', 'info');
            } else {
                console.error('MCTS suggestions error:', result.error);
                this.showStatus('AI analysis failed: ' + (result.error || 'Unknown error'), 'error');
                if (container) {
                    container.innerHTML = '<p class="placeholder-text" style="color: #e74c3c;">Error: ' + (result.error || 'Unknown error') + '</p>';
                }
            }
            
        } catch (error) {
            console.error('Error fetching AI suggestions:', error);
            console.error('Error name:', error.name);
            console.error('Error message:', error.message);
            
            if (error.name === 'AbortError') {
                this.showStatus('AI analysis timed out (30s)', 'error');
                if (container) {
                    container.innerHTML = '<p class="placeholder-text" style="color: #e74c3c;">Request timed out</p>';
                }
            } else {
                this.showStatus('AI connection error: ' + error.message, 'error');
                if (container) {
                    container.innerHTML = '<p class="placeholder-text" style="color: #e74c3c;">Connection error: ' + error.message + '</p>';
                }
            }
        }
        
        this.aiThinking = false;
    }

    displaySuggestions(suggestions) {
        const container = document.getElementById('aiSuggestions');
        if (!container) return;
        
        if (!suggestions || suggestions.length === 0) {
            container.innerHTML = '<p class="placeholder-text">No suggestions available</p>';
            return;
        }
        
        container.innerHTML = suggestions.map((sugg, index) => {
            const edge = this.edges[sugg.action];
            const edgeLabel = this.getEdgeLabel(edge);
            return `
                <div class="suggestion-item" data-edge-index="${sugg.action}" data-rank="${sugg.rank}">
                    <div class="suggestion-rank rank-${sugg.rank}">${sugg.rank}</div>
                    <div class="suggestion-info">
                        <div class="suggestion-edge">${edgeLabel}</div>
                        <div class="suggestion-confidence">${sugg.confidence}% confidence • ${sugg.visits} visits</div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Add click handlers to suggestion items
        container.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', async () => {
                const edgeIndex = parseInt(item.dataset.edgeIndex);
                const edge = this.edges[edgeIndex];
                if (edge && !edge.claimed) {
                    await this.claimEdge(edge);
                }
            });
        });
    }

    getEdgeLabel(edge) {
        // Create a short label for the edge
        const x1 = Math.round(edge.x1);
        const y1 = Math.round(edge.y1);
        const x2 = Math.round(edge.x2);
        const y2 = Math.round(edge.y2);
        return `(${x1},${y1})→(${x2},${y2})`;
    }

    highlightSuggestedEdges(suggestions) {
        // Remove existing highlights
        this.svg.selectAll('.voronoi-edge')
            .classed('edge-suggestion-1 edge-suggestion-2 edge-suggestion-3 edge-suggestion-4 edge-suggestion-5', false);
        
        // Add new highlights
        suggestions.forEach(sugg => {
            const edge = this.edges[sugg.action];
            if (edge && !edge.claimed) {
                this.svg.selectAll('.voronoi-edge')
                    .filter(d => d.id === edge.id)
                    .classed(`edge-suggestion-${sugg.rank}`, true);
            }
        });
    }

    clearSuggestions() {
        this.currentSuggestions = [];
        
        // Clear sidebar display
        const container = document.getElementById('aiSuggestions');
        if (container) {
            container.innerHTML = '<p class="placeholder-text">AI suggestions will appear on Player 2\'s turn</p>';
        }
        
        // Clear edge highlights
        this.svg.selectAll('.voronoi-edge')
            .classed('edge-suggestion-1 edge-suggestion-2 edge-suggestion-3 edge-suggestion-4 edge-suggestion-5', false);
    }

    addMoveToAIHistory(edge, score) {
        if (!this.aiEnabled) return;
        
        const moveNumber = this.aiMoveHistory.length + 1;
        const edgeLabel = this.getEdgeLabel(edge);
        
        this.aiMoveHistory.push({
            moveNumber,
            edgeLabel,
            edgeId: edge.id,
            score,
            timestamp: new Date().toLocaleTimeString()
        });
        
        this.updateAIMoveHistoryDisplay();
    }

    updateAIMoveHistoryDisplay() {
        const container = document.getElementById('aiMoveHistory');
        if (!container) return;
        
        if (this.aiMoveHistory.length === 0) {
            container.innerHTML = '<p class="placeholder-text">No AI moves yet</p>';
            return;
        }
        
        // Show last 10 moves, most recent first
        const recentMoves = this.aiMoveHistory.slice(-10).reverse();
        
        container.innerHTML = recentMoves.map(move => `
            <div class="move-history-item">
                <span class="move-number">#${move.moveNumber}</span>
                <span class="move-edge" title="${move.edgeId}">${move.edgeLabel}</span>
                <span class="move-score">+${move.score}</span>
            </div>
        `).join('');
    }

}

// Initialize the AI game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoronoiAIGame();
});
