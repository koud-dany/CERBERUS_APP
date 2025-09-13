
from flask import Flask, render_template_string, send_from_directory
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import os, sys

# Make 'apps' importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(BASE_DIR, "apps")
if APPS_DIR not in sys.path:
    sys.path.insert(0, APPS_DIR)

# Import the two existing Flask apps
from apps.manga.app import app as manga_app
from apps.text2img.app import app as text2img_app

# Minimal dashboard app
dashboard = Flask("dashboard")

# Add static file serving for mounted apps
@dashboard.route('/static/<path:filename>')
def serve_static(filename):
    # First try text2img static files
    text2img_static_path = os.path.join(APPS_DIR, 'text2img', 'static', filename)
    if os.path.exists(text2img_static_path):
        return send_from_directory(os.path.join(APPS_DIR, 'text2img', 'static'), filename)
    
    # Then try manga static files
    manga_static_path = os.path.join(APPS_DIR, 'manga', 'static', filename)
    if os.path.exists(manga_static_path):
        return send_from_directory(os.path.join(APPS_DIR, 'manga', 'static'), filename)
    
    # If not found, return 404
    return "File not found", 404

# Add specific static file serving for manga app
@dashboard.route('/manga/static/<path:filename>')
def serve_manga_static(filename):
    return send_from_directory(os.path.join(APPS_DIR, 'manga', 'static'), filename)

# Add upload file serving for manga app
@dashboard.route('/manga/uploads/<session_id>/<filename>')
def serve_manga_uploads(session_id, filename):
    if os.path.exists('D:\\'):
        upload_folder = 'D:\\manga_uploads'
    else:
        upload_folder = os.path.join(APPS_DIR, 'manga', 'uploads')
    return send_from_directory(os.path.join(upload_folder, session_id), filename)

@dashboard.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cerberus - Multi-App Platform</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                color: white;
            }
            .container {
                text-align: center;
                max-width: 800px;
                padding: 40px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            }
            .logo {
                font-size: 4rem;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            h1 {
                font-size: 3rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                font-weight: 800;
            }
            .subtitle {
                font-size: 1.2rem;
                margin-bottom: 40px;
                opacity: 0.9;
            }
            .apps-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin-top: 40px;
            }
            .app-card {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 30px;
                text-decoration: none;
                color: white;
                transition: all 0.3s ease;
                border: 2px solid rgba(255, 255, 255, 0.2);
            }
            .app-card:hover {
                transform: translateY(-10px);
                background: rgba(255, 255, 255, 0.25);
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
                border-color: rgba(255, 255, 255, 0.4);
            }
            .app-icon {
                font-size: 3rem;
                margin-bottom: 15px;
            }
            .app-title {
                font-size: 1.5rem;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .app-description {
                opacity: 0.9;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🐕‍🦺🐕‍🦺🐕‍🦺</div>
            <h1>CERBERUS</h1>
            <p class="subtitle">Multi-App Platform - Guardian of Digital Workflows</p>
            
            <div class="apps-grid">
                <a href="/manga/" class="app-card">
                    <div class="app-icon">🎌</div>
                    <div class="app-title">Manga Recap</div>
                    <div class="app-description">Transform manga images into AI-powered recaps with video generation capabilities</div>
                </a>
                
                <a href="/text2img/" class="app-card">
                    <div class="app-icon">🎨</div>
                    <div class="app-title">Text → Images</div>
                    <div class="app-description">Generate stunning images from text descriptions using OpenAI's DALL·E</div>
                </a>
            </div>
        </div>
    </body>
    </html>
    """)

# Mount apps under prefixes
application = DispatcherMiddleware(dashboard, {
    "/manga": manga_app,
    "/text2img": text2img_app
})

if __name__ == "__main__":
    run_simple("0.0.0.0", 5000, application, use_reloader=True, use_debugger=True)
