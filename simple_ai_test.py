#!/usr/bin/env python3
"""
Simple test for the AI game mode without complex RL training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
import threading
import time

def test_web_app():
    """Test the web application with AI mode"""
    print("Testing Voronoi Connect 4 with AI mode...")
    
    # Start the Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    print("Starting web server on http://localhost:5000")
    print("Instructions:")
    print("1. Open your browser and go to http://localhost:5000")
    print("2. Enable 'AI Mode' toggle")
    print("3. Click 'Generate' to create a Voronoi diagram")
    print("4. Play as Player 1 (Red) against the AI (Player 2 - Blue)")
    print("5. The AI uses a simple heuristic strategy")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    test_web_app()
