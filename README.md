# CERBERUS APP

A powerful Flask-based application that combines multiple AI-powered tools:

## Main Components

- **/manga** → Manga processing and video generation app
  - Extract text from manga images using Claude Vision
  - Generate recaps and summaries
  - Create automated videos with narration
  - Multiple video generation methods (Edge TTS, HeyGen AI, OpenAI)

- **/text2img** → Text to Image generator
  - Convert text descriptions to images
  - Gallery view for generated images

The application uses `DispatcherMiddleware` to elegantly integrate multiple Flask apps under a single server.

## Features

- Advanced manga processing capabilities
- Multiple AI integrations (Claude, OpenAI, HeyGen)
- Text-to-Speech with Edge TTS
- Automated video generation
- Image processing and manipulation
- Clean and modern web interface

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
- Copy `.env.example` to `.env` in the root directory
- Set up individual `.env` files in `apps/manga/` and `apps/text2img/` if needed
- Add your API keys for OpenAI, Claude, and HeyGen

4. Run the application:
```bash
python main.py
```

5. Access the application:
- Open http://localhost:5000/
- Manga processing: http://localhost:5000/manga
- Text to Image: http://localhost:5000/text2img

## Requirements

- Python 3.8+
- FFmpeg (for video processing)
- ImageMagick (for image processing)
- Edge TTS (for basic text-to-speech)
- API keys for:
  - OpenAI API
  - Claude API (optional)
  - HeyGen API (optional)

## Project Structure

```
CERBERUS_APP/
├── apps/
│   ├── manga/          # Manga processing application
│   │   ├── static/
│   │   ├── templates/
│   │   └── app.py
│   └── text2img/       # Text to image generator
│       ├── static/
│       ├── templates/
│       └── app.py
├── static/             # Shared static files
├── main.py            # Main application entry point
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## License

MIT License
