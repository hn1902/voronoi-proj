class VoronoiConnect4 {
    constructor() {
        this.svg = d3.select('#voronoiSvg');
        this.width = 800;
        this.height = 600;
        this.points = [];
        this.voronoi = null;
        this.edges = [];
        this.claimedEdges = new Map(); // edgeId -> player
        this.currentPlayer = 1;
        this.player1Score = 0;
        this.player2Score = 0;
        this.gameActive = false;
        this.timerActive = false;
        this.timerInterval = null;
        this.timeRemaining = 60; // 60 seconds per turn
        this.tooltip = null;
        
        this.initializeEventListeners();
        this.createTooltip();
    }

    initializeEventListeners() {
        // Generate button
        document.getElementById('generateBtn').addEventListener('click', () => {
            this.generateVoronoi();
        });

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
                document.getElementById('timerToggle').click();
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
        this.showStatus('Game started! Click on edges to claim them.', 'info');
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
        // Create Voronoi diagram using D3
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

            const path = d3.path();
            path.path = cell;
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
            .on('click', (event, d) => {
                this.claimEdge(d);
            })
            .on('mouseover', (event, d) => {
                if (!d.claimed) {
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
                // Close path - connect to first point
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
        // Create a unique ID for each edge based on its endpoints
        const x1 = Math.round(segment.x1 * 100) / 100;
        const y1 = Math.round(segment.y1 * 100) / 100;
        const x2 = Math.round(segment.x2 * 100) / 100;
        const y2 = Math.round(segment.y2 * 100) / 100;
        
        // Sort coordinates to ensure consistent edge IDs regardless of direction
        const coords = [[x1, y1], [x2, y2]].sort((a, b) => {
            if (a[0] !== b[0]) return a[0] - b[0];
            return a[1] - b[1];
        });
        
        return `${coords[0][0]},${coords[0][1]}-${coords[1][0]},${coords[1][1]}`;
    }

    claimEdge(edge) {
        if (!this.gameActive || edge.claimed) return;

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
        this.switchPlayer();
        this.checkGameEnd();
    }

    updateScore() {
        // Calculate score based on claimed edges and formed polygons
        this.player1Score = this.calculatePlayerScore(1);
        this.player2Score = this.calculatePlayerScore(2);

        document.getElementById('player1Score').textContent = this.player1Score;
        document.getElementById('player2Score').textContent = this.player2Score;
    }

    calculatePlayerScore(player) {
        const playerEdges = Array.from(this.claimedEdges.entries())
            .filter(([edgeId, edgePlayer]) => edgePlayer === player)
            .map(([edgeId]) => edgeId);

        // Calculate polygon bonuses first
        const polygons = this.findCompletePolygons(playerEdges);
        
        // Track which edges are part of polygons
        const edgesInPolygons = new Set();
        for (const polygon of polygons) {
            for (let i = 0; i < polygon.length; i++) {
                const v1 = polygon[i];
                const v2 = polygon[(i + 1) % polygon.length];
                // Create edge ID (sorted for consistency)
                const coords = [v1, v2].sort();
                edgesInPolygons.add(`${coords[0]}-${coords[1]}`);
            }
        }
        
        // Base score: 1 point per edge that's NOT part of any polygon
        let score = playerEdges.filter(edgeId => !edgesInPolygons.has(edgeId)).length;
        
        // Bonus: 4 points per edge in a polygon (1 base + 3 bonus)
        score += edgesInPolygons.size * 4;

        return score;
    }

    findCompletePolygons(playerEdges) {
        // Find closed loops (polygons) formed by connected edges of the same player
        const polygons = [];
        if (playerEdges.length < 3) return polygons; // Need at least 3 edges for a polygon
        
        // Build adjacency map: vertex -> connected vertices
        const edgeMap = new Map();
        const edges = new Map(); // Store actual edge information
        
        for (const edgeId of playerEdges) {
            const [start, end] = edgeId.split('-');
            const [x1, y1] = start.split(',').map(Number);
            const [x2, y2] = end.split(',').map(Number);
            
            const v1 = `${x1},${y1}`;
            const v2 = `${x2},${y2}`;
            
            // Add to edge map
            if (!edgeMap.has(v1)) edgeMap.set(v1, []);
            if (!edgeMap.has(v2)) edgeMap.set(v2, []);
            
            edgeMap.get(v1).push(v2);
            edgeMap.get(v2).push(v1);
            
            // Store edge with vertices
            edges.set(edgeId, { v1, v2 });
        }

        // Find all cycles using DFS
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
        const foundCycles = new Set(); // Use Set to prevent duplicates
        const visitedInPath = new Set();
        
        const dfs = (current, path, visited) => {
            // Mark current as visited in this path
            visited.add(current);
            path.push(current);
            
            const neighbors = edgeMap.get(current) || [];
            
            for (const neighbor of neighbors) {
                if (!path.includes(neighbor)) {
                    // Continue exploring
                    dfs(neighbor, path, visited);
                } else if (path.length > 2 && neighbor === path[0]) {
                    // Found a cycle: path is a complete loop
                    const cycle = [...path];
                    // Normalize the cycle by rotating to start with the smallest vertex
                    const minIndex = cycle.indexOf(cycle.reduce((min, v) => v < min ? v : min));
                    const normalizedCycle = [...cycle.slice(minIndex), ...cycle.slice(0, minIndex)];
                    const cycleKey = normalizedCycle.join('-');
                    
                    foundCycles.add(cycleKey);
                }
            }
            
            // Backtrack
            path.pop();
            visited.delete(current);
        };
        
        dfs(startVertex, [], visitedInPath);
        globallyVisited.add(startVertex);
        
        // Convert Set to Array
        return Array.from(foundCycles).map(cycleKey => cycleKey.split('-'));
    }

    switchPlayer() {
        this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
        document.getElementById('currentPlayerText').textContent = `Player ${this.currentPlayer}'s Turn`;
        
        if (this.timerActive) {
            this.startTimer();
        }
    }

    startTimer() {
        this.stopTimer();
        this.timeRemaining = 60;
        this.updateTimerDisplay();
        
        this.timerInterval = setInterval(() => {
            this.timeRemaining--;
            this.updateTimerDisplay();
            
            if (this.timeRemaining <= 0) {
                this.timeUp();
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

    timeUp() {
        this.stopTimer();
        this.switchPlayer();
        this.showStatus(`Time's up! Player ${this.currentPlayer === 1 ? 2 : 1} gets another turn.`, 'warning');
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
            winner = 1;
        } else if (this.player2Score > this.player1Score) {
            winner = 2;
        } else {
            this.showStatus("It's a draw! Both players have the same score.", 'draw');
            return;
        }
        
        this.showStatus(`Player ${winner} wins! Final Score - P1: ${this.player1Score}, P2: ${this.player2Score}`, 'winner');
    }

    setupGame() {
        this.gameActive = true;
        this.currentPlayer = 1;
        this.player1Score = 0;
        this.player2Score = 0;
        this.claimedEdges.clear();
        
        // Reset all edges
        this.edges.forEach(edge => {
            edge.claimed = false;
            edge.player = null;
        });
        
        this.updateScore();
        document.getElementById('currentPlayerText').textContent = 'Player 1\'s Turn';
        
        if (this.timerActive) {
            this.startTimer();
        }
    }

    resetGame() {
        this.gameActive = false;
        this.stopTimer();
        this.clearSVG();
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
        
        // Auto-hide info messages after 3 seconds
        if (type === 'info') {
            setTimeout(() => {
                statusElement.textContent = '';
                statusElement.className = 'game-status';
            }, 3000);
        }
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoronoiConnect4();
});

