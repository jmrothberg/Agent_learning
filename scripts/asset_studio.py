#!/usr/bin/env python3
"""Local Asset Studio — drag/drop PNG helper + Z-Image-Turbo generation.

Serves scripts/asset_studio.html on localhost and exposes a tiny JSON API so
the browser can txt2img, img2img-modify, and save PNGs to a folder you pick.

Usage:
    .venv/bin/python scripts/asset_studio.py
    .venv/bin/python scripts/asset_studio.py --port 8765 --no-browser

Requires the same GPU stack as the agent: ./scripts/install_diffuser.sh
"""

from __future__ import annotations

import argparse
import base64
import json
import platform
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
HTML_PATH = Path(__file__).resolve().parent / "asset_studio.html"

sys.path.insert(0, str(REPO))

import assets  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_gen_lock = threading.Lock()
_generator: assets.ZImageTurboGenerator | None = None
_generator_error: str | None = None
_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None


def _get_generator() -> assets.ZImageTurboGenerator:
    global _generator, _generator_error
    if _generator is None:
        _generator = assets.try_load_image_generator()
        if _generator is None:
            _generator_error = (
                "Could not construct ZImageTurboGenerator — run "
                "./scripts/install_diffuser.sh"
            )
            _generator = assets.ZImageTurboGenerator()
    return _generator


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _decode_b64_png(data_b64: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(data_b64))


def _process_png(src_path: Path, size: tuple[int, int], chroma_key: bool) -> bytes:
    from PIL import Image

    with Image.open(src_path) as src_img:
        src_img.load()
        if size != (src_img.width, src_img.height):
            img = src_img.resize(size, Image.LANCZOS)
        else:
            img = src_img.copy()
        if chroma_key:
            keyed, _ = assets._chroma_key_to_rgba(img)
            out = keyed
        else:
            out = img.convert("RGBA")
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _safe_filename(name: str) -> str:
    cleaned = Path(name).name.strip()
    if not cleaned:
        cleaned = "asset.png"
    if not cleaned.lower().endswith(".png"):
        cleaned += ".png"
    return cleaned


def _pick_file_osascript() -> str | None:
    """Native macOS file picker — safe from HTTP worker threads (no tkinter)."""
    script = (
        'set f to choose file of type {"png", "PNG", "public.png"} '
        'with prompt "Open PNG"\n'
        "return POSIX path of f"
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    path = (r.stdout or "").strip()
    return path or None


def _pick_folder_osascript() -> str | None:
    script = (
        'return POSIX path of (choose folder with prompt "Choose save folder")'
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    path = (r.stdout or "").strip()
    return path or None


def _pick_file_tkinter() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Open PNG",
        filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def _pick_folder_tkinter() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="Choose save folder")
    root.destroy()
    return path or None


def _pick_file_native() -> str | None:
    if platform.system() == "Darwin":
        return _pick_file_osascript()
    return _pick_file_tkinter()


def _pick_folder_native() -> str | None:
    if platform.system() == "Darwin":
        return _pick_folder_osascript()
    return _pick_folder_tkinter()


class AssetStudioHandler(BaseHTTPRequestHandler):
    server_version = "AssetStudio/1.0"

    def end_headers(self) -> None:
        # Allow file:// bookmarks and other local origins to hit the API.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quieter default — the UI has its own log panel.
        sys.stderr.write(f"[asset_studio] {self.address_string()} {fmt % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            if not HTML_PATH.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "asset_studio.html missing")
                return
            body = HTML_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            gen = _get_generator()
            device = getattr(gen, "_device", None)
            err = _generator_error or getattr(gen, "_last_error", None)
            loaded = getattr(gen, "_pipeline", None) is not None
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "device": device,
                    "loaded": loaded,
                    "error": err,
                },
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/pick-file":
            path = _pick_file_native()
            if not path:
                _json_response(self, HTTPStatus.OK, {"path": None})
                return
            p = Path(path).expanduser().resolve()
            if not p.is_file():
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Not a file: {p}"},
                )
                return
            try:
                b64 = _b64_png(p)
            except OSError as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "path": str(p),
                    "parent_dir": str(p.parent),
                    "filename": p.name,
                    "image_b64": b64,
                },
            )
            return

        if route == "/api/pick-folder":
            path = _pick_folder_native()
            _json_response(
                self,
                HTTPStatus.OK,
                {"path": str(Path(path).expanduser().resolve()) if path else None},
            )
            return

        if route == "/api/save":
            data = _read_json(self)
            out_dir = (data.get("dir") or "").strip()
            filename = _safe_filename(str(data.get("filename") or "asset.png"))
            image_b64 = data.get("image_b64") or ""
            if not out_dir:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dir required"})
                return
            if not image_b64:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "image_b64 required"},
                )
                return
            target = Path(out_dir).expanduser().resolve() / filename
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                _decode_b64_png(image_b64, target)
            except OSError as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            _json_response(self, HTTPStatus.OK, {"path": str(target)})
            return

        if route == "/api/generate":
            data = _read_json(self)
            mode = (data.get("mode") or "txt2img").strip().lower()
            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "prompt required"},
                )
                return
            try:
                size_raw = data.get("size") or [512, 512]
                w, h = int(size_raw[0]), int(size_raw[1])
                size = (max(16, w), max(16, h))
            except (TypeError, ValueError, IndexError):
                size = (512, 512)
            chroma_key = bool(data.get("chroma_key", True))
            strength = float(data.get("strength") or 0.55)
            init_b64 = data.get("init_image_b64")

            t0 = time.time()
            note = ""
            with _gen_lock:
                gen = _get_generator()
                temp_path: Path | None = None
                try:
                    if mode == "img2img":
                        if not init_b64:
                            _json_response(
                                self,
                                HTTPStatus.BAD_REQUEST,
                                {"error": "init_image_b64 required for img2img"},
                            )
                            return
                        tmp = REPO / "games" / "_asset_studio_tmp"
                        tmp.mkdir(parents=True, exist_ok=True)
                        init_path = tmp / f"init_{int(time.time() * 1000)}.png"
                        _decode_b64_png(init_b64, init_path)
                        raw = gen.generate_img2img(
                            prompt,
                            str(init_path),
                            strength=strength,
                        )
                        try:
                            init_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    else:
                        if chroma_key and "transparent" not in prompt.lower():
                            prompt = f"{prompt}, transparent background"
                            note = "added transparent background hint"
                        raw = gen.generate(prompt)

                    if not raw:
                        err = getattr(gen, "_last_error", None) or _generator_error
                        _json_response(
                            self,
                            HTTPStatus.BAD_REQUEST,
                            {"error": err or "generation returned None"},
                        )
                        return

                    temp_path = Path(raw)
                    png_bytes = _process_png(temp_path, size, chroma_key)
                finally:
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass

            elapsed = round(time.time() - t0, 2)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "image_b64": base64.b64encode(png_bytes).decode("ascii"),
                    "elapsed_s": elapsed,
                    "note": note,
                },
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def server_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/"


def is_server_up(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/status",
            timeout=0.4,
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    daemon: bool = False,
) -> ThreadingHTTPServer:
    """Bind and start the HTTP server. Idempotent if already running."""
    global _server, _server_thread
    if is_server_up(host, port):
        return _server  # type: ignore[return-value]
    if _server is not None:
        return _server

    if not HTML_PATH.is_file():
        raise FileNotFoundError(f"Missing UI: {HTML_PATH}")

    _server = ThreadingHTTPServer((host, port), AssetStudioHandler)
    if daemon:
        _server_thread = threading.Thread(
            target=_server.serve_forever,
            name="asset-studio-http",
            daemon=True,
        )
        _server_thread.start()
    return _server


def ensure_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    open_browser: bool = False,
) -> str:
    """Start background server if needed; optionally open the UI."""
    url = server_url(host, port)
    if not is_server_up(host, port):
        start_server(host, port, daemon=True)
        # Wait briefly for bind — first call after cold start.
        for _ in range(20):
            if is_server_up(host, port):
                break
            time.sleep(0.05)
    if open_browser:
        webbrowser.open(url)
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description="Asset Studio — local PNG helper")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Start server in background and exit (used by chat.py)",
    )
    args = parser.parse_args()

    if not HTML_PATH.is_file():
        print(f"Missing UI: {HTML_PATH}", file=sys.stderr)
        return 1

    if args.background:
        if is_server_up(args.host, args.port):
            print(server_url(args.host, args.port))
            return 0
        ensure_server(args.host, args.port, open_browser=False)
        print(server_url(args.host, args.port))
        return 0

    if is_server_up(args.host, args.port):
        url = server_url(args.host, args.port)
        print(f"Asset Studio already running at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    server = start_server(args.host, args.port, daemon=False)
    url = server_url(args.host, args.port)
    print(f"Asset Studio at {url}")
    print("Drag PNGs, generate sprites/backgrounds, save into your *_assets/ folder.")
    print("Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
        gen = _generator
        if gen is not None:
            try:
                gen.cleanup()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
