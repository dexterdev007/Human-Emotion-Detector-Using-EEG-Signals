#!/usr/bin/env python3
"""
Run the EEG Emotion Detection UI without Flask.
Serves the static web/ folder on localhost.
"""

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

if not WEB_DIR.exists():
    raise SystemExit("Missing web/ directory. Run scripts/build_model_js.py first.")

# Serve from project root so "/" resolves to index.html which redirects to /web/.
os.chdir(ROOT)

port = int(os.environ.get("PORT", "5500"))
server = ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
print(f"Serving UI at http://127.0.0.1:{port}")
server.serve_forever()
