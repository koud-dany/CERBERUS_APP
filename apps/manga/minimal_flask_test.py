#!/usr/bin/env python3
"""
Minimal Flask test to identify startup issues
"""

from flask import Flask, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        'status': 'success',
        'message': 'Minimal Flask app is working!',
        'env_vars': {
            'CLAUDE_API_KEY': 'Set' if os.getenv('CLAUDE_API_KEY') else 'Missing',
            'OPENAI_API_KEY': 'Set' if os.getenv('OPENAI_API_KEY') else 'Missing'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting minimal Flask test...")
    print("Environment check:")
    print(f"  CLAUDE_API_KEY: {'Set' if os.getenv('CLAUDE_API_KEY') else 'Missing'}")
    print(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Missing'}")
    
    try:
        app.run(debug=True, port=5001)
    except Exception as e:
        print(f"Error starting Flask: {e}")
        import traceback
        traceback.print_exc()
