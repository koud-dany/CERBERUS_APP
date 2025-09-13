# Manga Recap Generator

A Flask web application that uploads manga images, extracts text using OCR, and generates AI-powered recaps using Claude API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Tesseract OCR:
   - Windows: Download from GitHub
   - Mac: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

3. Run the app:
   ```bash
   python app.py
   ```

4. Open http://localhost:5000

## Usage

1. Upload manga images
2. Extract text using OCR
3. Edit extracted text if needed
4. Enter Claude API key
5. Generate recap

## Requirements

- Python 3.7+
- Tesseract OCR
- Claude API key from console.anthropic.com
