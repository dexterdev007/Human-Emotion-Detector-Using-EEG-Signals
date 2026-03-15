#!/usr/bin/env python3
"""
Serve the frontend folder for the integrated EEG + Brain Tumor web app.
"""

import os
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

if not (FRONTEND_DIR / "index.html").exists():
    raise SystemExit(f"Missing frontend index at: {FRONTEND_DIR / 'index.html'}")

def find_available_port(start: int = 5500, end: int = 5600) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found in range 5500-5600")


requested_port = int(os.environ.get("PORT", "5500"))
port = find_available_port(start=requested_port, end=5600)

handler = partial(SimpleHTTPRequestHandler, directory=str(FRONTEND_DIR))
server = ThreadingHTTPServer(("127.0.0.1", port), handler)
print(f"Serving frontend: {FRONTEND_DIR}")
print(f"Open: http://127.0.0.1:{port}")
server.serve_forever()
