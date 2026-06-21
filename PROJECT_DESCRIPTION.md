# Voronoi Connect 4 - Project Description

## 1. Project Overview

**Voronoi Connect 4** is a modern, interactive web-based strategic game that combines computational geometry with competitive gameplay. The game uses Voronoi diagrams as the foundation for a unique edge-capturing game where players compete to claim edges and form complete polygons to maximize their scores.

### Key Highlights
- Interactive web-based game accessible through any modern browser
- Dynamic Voronoi diagram generation with customizable complexity
- Strategic gameplay requiring tactical thinking and planning
- Modern, responsive user interface with dark mode support
- Real-time visual feedback and scoring system

---

## 2. Game Concept & Mechanics

### Core Concept
The game is built around **Voronoi diagrams** - geometric structures that partition a plane into regions based on proximity to a set of points. Each Voronoi diagram consists of:
- **Points**: Seed points that define the diagram structure
- **Edges**: Lines connecting vertices that form the boundaries between regions
- **Polygons**: Closed regions formed by connected edges

### Game Mechanics
- **Edge Claiming**: Players take turns clicking on edges to claim them
- **Scoring System**: 
  - 1 point for each claimed edge that is NOT part of a completed polygon
  - 4 points for each claimed edge that IS part of a completed polygon
- **Turn-Based Play**: Alternating turns between Player 1 (Red) and Player 2 (Blue)
- **Win Condition**: Player with the highest score when all edges are claimed wins

### Strategic Elements
- Players must balance claiming individual edges versus working toward polygon completion
- Polygon formation upgrades each contributing edge from 1 point to 4 points (4x multiplier)
- Limited edges create competitive resource management
- Spatial awareness of edge connections is crucial

---

## 3. Features

### Game Features
- **Dynamic Board Generation**: Generate Voronoi diagrams with 3-20 random points
- **Customizable Complexity**: Adjust difficulty by changing the number of seed points
- **Timer Mode**: Optional 60-second timer per turn for faster-paced gameplay
- **Real-Time Scoring**: Live score updates as edges are claimed
- **Visual Feedback**: Color-coded edges (Red for Player 1, Blue for Player 2)
- **Game Status Messages**: Clear indicators for turn changes, game end, and winner announcements

### User Interface Features
- **Modern Design**: Clean, professional interface with smooth animations
- **Dark Mode**: Toggle between light and dark themes
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Interactive SVG Graphics**: Clickable edges with hover effects
- **Keyboard Shortcuts**: 
  - `R`: Reset game
  - `N`: New game
  - `T`: Toggle timer mode
- **Game Controls**: Easy-to-use buttons for all game functions

### Technical Features
- **Client-Side Rendering**: Fast, responsive gameplay without server delays
- **Local Storage**: Theme preferences saved across sessions
- **Cross-Browser Compatibility**: Works on Chrome, Firefox, Safari, and Edge
- **No External Dependencies**: Self-contained game logic

---

## 4. Technical Architecture

### Frontend Technology
- **HTML5**: Semantic markup for structure
- **CSS3**: Modern styling with CSS variables for theming
- **JavaScript (ES6+)**: Game logic and interactivity
- **D3.js**: Voronoi diagram generation using Delaunay triangulation
- **SVG**: Scalable vector graphics for rendering game board

### Backend Technology
- **Flask**: Lightweight Python web framework
- **Python 3**: Server-side application logic
- **Static File Serving**: Efficient delivery of assets

### Architecture Pattern
- **Client-Side Game Logic**: All game state and logic handled in browser
- **Server-Side Rendering**: Flask serves the initial HTML template
- **RESTful Design**: Simple, clean API structure

---

## 5. User Interface Design

### Layout Structure
- **Header Section**: Game title and theme toggle
- **Control Panel**: Point input, game settings, and score display
- **Game Board**: Central SVG canvas for Voronoi diagram
- **Action Buttons**: Reset and New Game controls
- **Status Display**: Real-time game status messages

### Visual Design Elements
- **Color Scheme**: 
  - Player 1: Red (#dc2626)
  - Player 2: Blue (#2563eb)
  - Neutral: Gray tones for unclaimed edges
- **Typography**: Modern system fonts for readability
- **Spacing**: Generous padding and margins for clarity
- **Animations**: Smooth transitions for state changes
- **Hover Effects**: Visual feedback on interactive elements

### Responsive Design
- **Desktop**: Full-featured layout with optimal spacing
- **Tablet**: Adapted controls and sizing
- **Mobile**: Touch-friendly interface with adjusted dimensions

---

## 6. Gameplay Flow

### Initial Setup
1. User opens the web application in their browser
2. Default Voronoi diagram is displayed (or user generates new one)
3. Game board shows all unclaimed edges in neutral color

### Gameplay Sequence
1. **Turn Start**: Current player indicator shows active player
2. **Edge Selection**: Player clicks on an unclaimed edge
3. **Edge Claiming**: Edge changes to player's color
4. **Score Update**: Scores update immediately
5. **Polygon Check**: System checks for completed polygons
6. **Score Recalculation**: Edges in completed polygons score 4 points each instead of 1
7. **Turn Switch**: Next player's turn begins
8. **Game End**: When all edges claimed, winner is announced

### Game States
- **Initial**: Board generated, no edges claimed
- **In Progress**: Players taking turns
- **Completed**: All edges claimed, scores finalized
- **Reset**: Return to initial state

---

## 7. Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Modern web browser (Chrome 60+, Firefox 55+, Safari 12+, Edge 79+)

### Installation Steps
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application**:
   ```bash
   python app.py
   ```

3. **Access Game**:
   - Open browser
   - Navigate to `http://localhost:5000`

### Project Structure
```
voronoi-proj/
├── app.py                 # Flask application entry point
├── templates/
│   └── index.html        # Main game interface
├── static/
│   ├── css/
│   │   └── style.css    # Styling and themes
│   └── js/
│       ├── game.js      # Core game logic
│       └── ai-game.js   # Enhanced game features
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

---

## 8. Technology Stack Summary

### Frontend Stack
- **HTML5**: Structure and semantic markup
- **CSS3**: Styling, theming, and responsive design
- **JavaScript**: Game logic, event handling, state management
- **D3.js v7**: Voronoi diagram generation and manipulation
- **SVG**: Vector graphics rendering

### Backend Stack
- **Flask 2.3.3**: Web framework
- **Werkzeug 2.3.7**: WSGI utilities
- **Python 3**: Programming language

### Development Tools
- **Git**: Version control
- **Browser DevTools**: Debugging and testing

---

## 9. Key Algorithms & Implementation

### Voronoi Diagram Generation
- Uses D3.js Delaunay triangulation to generate Voronoi cells
- Converts triangulation results into edge and vertex data
- Handles edge cases for boundary conditions

### Polygon Detection
- Graph-based cycle detection algorithm
- Tracks edge connections to identify closed polygons
- Validates polygon completion for scoring

### Game State Management
- Tracks claimed edges per player
- Maintains current player turn
- Calculates scores in real-time
- Manages game completion state

---

## 10. Browser Compatibility & Performance

### Supported Browsers
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

### Performance Characteristics
- **Fast Rendering**: SVG-based graphics for smooth performance
- **Efficient Updates**: Minimal DOM manipulation
- **Responsive Interactions**: Immediate feedback on user actions
- **Scalable Graphics**: Vector-based rendering scales to any size

---

## 11. Future Enhancement Possibilities

### Potential Features
- Multiplayer online mode
- Game history and replay functionality
- Difficulty levels with different scoring systems
- Tournament mode with leaderboards
- Custom color themes
- Sound effects and music
- Tutorial mode for new players
- Statistics tracking

---

## 12. Project Goals & Achievements

### Primary Objectives
- Create an engaging strategic game using computational geometry
- Demonstrate modern web development practices
- Provide an intuitive user experience
- Showcase interactive data visualization

### Success Metrics
- Smooth gameplay experience
- Intuitive user interface
- Cross-browser compatibility
- Responsive design implementation
- Clean, maintainable codebase

---

## Conclusion

Voronoi Connect 4 successfully combines mathematical concepts (Voronoi diagrams) with engaging gameplay mechanics. The project demonstrates proficiency in modern web development, interactive graphics, and user interface design. The game provides a unique strategic experience that challenges players to think spatially while competing for the highest score.


