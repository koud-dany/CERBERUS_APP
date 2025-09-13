import base64
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
)
from openai import OpenAI

# Load env from the correct directory
load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["STATIC_FOLDER"] = "static"
app.config["GENERATED_DIR"] = os.path.join(app.config["STATIC_FOLDER"], "generated")

# Ensure dirs exist
Path(app.config["GENERATED_DIR"]).mkdir(parents=True, exist_ok=True)

# OpenAI client
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)


def save_b64_png(b64_data: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_data))


@app.route("/")
def index():
    # Start with a clean page - don't show existing images on load
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt", "").strip()
    size = request.form.get("size", "1024x1024")
    try:
        n = int(request.form.get("n", "1"))
    except ValueError:
        n = 1
    n = max(1, min(n, 6))  # hard cap 1..6

    if not prompt:
        flash("Please enter a prompt.", "error")
        return redirect(url_for("index"))

    session_id = str(uuid.uuid4())[:8]
    ts = int(time.time())
    run_dir = Path(app.config["GENERATED_DIR"]) / f"{ts}_{session_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size=size,
            n=n,
        )

        saved_files = []
        for i, datum in enumerate(result.data):
            # DALL-E returns URLs, not base64 data
            image_url = datum.url
            # You'll need to download the image from the URL
            import requests
            response = requests.get(image_url)
            out_name = f"img_{i+1:02d}.png"
            out_path = run_dir / out_name
            
            with open(out_path, "wb") as f:
                f.write(response.content)
            saved_files.append(str(out_path).replace("\\", "/"))

        # Generate URLs that work with the main app's static serving
        image_urls = ["/static/" + "/".join(Path(p).parts[3:]) for p in saved_files]  # Skip 'apps/text2img/static'
        return render_template(
            "index.html",
            generated=image_urls,
            prompt=prompt,
            size=size,
            n=n,
            run_dir=str(run_dir).replace("\\", "/"),
        )

    except Exception as e:
        app.logger.exception("Image generation failed")
        flash(f"Generation failed: {e}", "error")
        return redirect(url_for("index"))


@app.route("/gallery")
def gallery():
    # Get all existing images for the gallery
    generated_dir = Path(app.config["GENERATED_DIR"])
    saved_images = []
    
    if generated_dir.exists():
        for run_dir in sorted(generated_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if run_dir.is_dir():
                for img_file in run_dir.glob("*.png"):
                    # Generate URL that works with main app's static serving
                    rel_path = str(img_file.relative_to(Path(app.config["STATIC_FOLDER"])))
                    image_url = "/static/" + rel_path.replace("\\", "/")
                    saved_images.append({
                        'url': image_url,
                        'filename': img_file.name,
                        'folder': run_dir.name,
                        'created': run_dir.stat().st_mtime
                    })
    
    return render_template("gallery.html", images=saved_images)


@app.route("/delete-image", methods=["POST"])
def delete_image():
    try:
        data = request.get_json()
        folder_name = data.get('folder')
        filename = data.get('filename')
        
        if not folder_name or not filename:
            return jsonify({'error': 'Missing folder or filename'}), 400
        
        # Construct the file path
        file_path = Path(app.config["GENERATED_DIR"]) / folder_name / filename
        
        if file_path.exists():
            file_path.unlink()  # Delete the file
            
            # Check if folder is empty and delete it too
            folder_path = file_path.parent
            if folder_path.exists() and not any(folder_path.iterdir()):
                folder_path.rmdir()
            
            return jsonify({'success': True, 'message': 'Image deleted successfully'})
        else:
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to delete image: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
