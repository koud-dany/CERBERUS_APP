
# Unified Flask App (Dashboard + Mounted Apps)

This repo integrates two existing Flask apps into a single server with a shared dashboard:

- **/manga** → Manga recap app
- **/text2img** → Text → Image generator

It uses `DispatcherMiddleware` to mount the original apps without invasive refactors.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Configure any .env expected by the sub-apps in their own folders:
# apps/manga/.env and/or apps/text2img/.env

python main.py
```

Then open http://localhost:5000/
