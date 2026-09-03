#!/usr/bin/env python3
"""Local OCR bridge for the Sentimant web tool.

Wraps the UltraFast-RapidOCR engine and exposes a tiny HTTP endpoint so the
browser-based analyzer can OCR dropped images. Binds to 127.0.0.1 only.

Usage:
    python3 ocr_bridge.py [--port 8765]

Endpoints:
    GET  /health             -> {"ok": true}
    POST /ocr               -> {"ok": true, "text": "...", "lines": [...], "latency_ms": 12.3}
                               raw image bytes in body, Content-Type image/*
    POST /ocr              with invalid image -> HTTP 400 {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make the engine importable when run from this repo OR when run from elsewhere
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hyper_ocr.engine import HyperOCREngine  # noqa: E402

_engine: HyperOCREngine | None = None
_engine_lock = threading.Lock()


def _get_engine() -> HyperOCREngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = HyperOCREngine()
    return _engine


def _decode_image(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def ocr_image(raw: bytes):
    """Return {'ok':True, 'text':..., 'lines':[...], 'latency_ms':...} or raise."""
    img = _decode_image(raw)
    if img is None:
        raise ValueError("Could not decode image")
    results, latency = _get_engine().recognize(img)
    text_lines = []
    for r in results:
        t = (r.get("text") or "").strip()
        if t:
            text_lines.append(t)
    return {
        "ok": True,
        "text": "\n".join(text_lines),
        "lines": [{"text": t, "confidence": r.get("confidence", 0)} for t, r in zip(text_lines, results)],
        "latency_ms": latency,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[ocr-bridge]", fmt % args)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path.split("?", 1)[0] in ("/health", "/health/"):
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.split("?", 1)[0] not in ("/ocr", "/ocr/"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 50 * 1024 * 1024:
                self._send_json(400, {"ok": False, "error": "bad content length"})
                return
            raw = self.rfile.read(length)
            result = ocr_image(raw)
            self._send_json(200, result)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            print("[ocr-bridge] error:", exc)
            self._send_json(500, {"ok": False, "error": f"OCR failed: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentimant local OCR bridge")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[ocr-bridge] listening on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ocr-bridge] stopped")


if __name__ == "__main__":
    main()