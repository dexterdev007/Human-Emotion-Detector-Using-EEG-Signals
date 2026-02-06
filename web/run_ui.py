#!/usr/bin/env python3
"""
Run the EEG Emotion Detection UI without Flask.
Serves the static web/ folder on localhost.
"""

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent

if not (WEB_DIR / "index.html").exists():
    raise SystemExit("Missing web/index.html.")

# Serve directly from web/ so / loads index.html
os.chdir(WEB_DIR)

port = int(os.environ.get("PORT", "5500"))
server = ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
print(f"Serving UI at http://127.0.0.1:{port}")
server.serve_forever()
