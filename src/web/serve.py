#!/usr/bin/env python3
"""
Simple HTTP server for the MRI-Crohn Atlas dashboard.

Usage:
    python serve.py
    # Then open http://localhost:8080 in your browser
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"\n🌐 MRI-Crohn Atlas Dashboard")
        print(f"   Serving at: {url}")
        print(f"   Press Ctrl+C to stop\n")

        # Try to open browser
        try:
            webbrowser.open(url)
        except:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
