# Voronoi Connect 4 - Strategic Edge Game

A modern, interactive web-based game that combines Voronoi diagrams with strategic edge-capturing gameplay. Players compete to claim the most edges and form complete polygons to win.

## Features

- **Dynamic Voronoi Generation**: Generate Voronoi diagrams with 3-20 random points
- **Strategic Gameplay**: Click on edges to claim them for your player
- **Smart Scoring System**: Earn points for claimed edges and bonus points for complete polygons
- **Timer Mode**: Optional timed turns for faster gameplay
- **Professional UI**: Modern, responsive design with smooth animations
- **Real-time Feedback**: Visual indicators and status messages

## How to Play

1. **Setup**: Enter the number of points (3-20) and click "Generate Voronoi"
2. **Gameplay**: Players take turns clicking on edges to claim them
   - Player 1: Red edges
   - Player 2: Blue edges
3. **Scoring**: 
   - 1 point per claimed edge
   - 5 bonus points per complete polygon formed
4. **Winning**: The player with the highest score when all edges are claimed wins

## Controls

- **Generate Voronoi**: Create a new game board
- **Timer Mode**: Toggle timed turns (60 seconds per turn)
- **Reset Game**: Clear the current game
- **New Game**: Generate a new Voronoi and start fresh

## Keyboard Shortcuts

- `R`: Reset game
- `N`: New game
- `T`: Toggle timer mode

## Installation & Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Technical Details

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Voronoi Generation**: D3.js Delaunay triangulation
- **Backend**: Flask (Python)
- **Styling**: Custom CSS with modern design principles
- **Responsive**: Works on desktop and mobile devices

## Game Rules

- Players alternate turns
- Each edge can only be claimed once
- Edges are claimed by clicking on them
- Score is calculated based on:
  - Number of claimed edges
  - Number of complete polygons formed
- Game ends when all edges are claimed
- Player with highest score wins

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

Enjoy the strategic gameplay and beautiful Voronoi diagrams!

