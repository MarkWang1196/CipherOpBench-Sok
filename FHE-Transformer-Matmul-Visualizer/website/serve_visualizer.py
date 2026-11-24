"""Local static server for the visualizer with development caching disabled."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the FHE Transformer matrix multiplication visualizer without browser caching")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    website = Path(__file__).resolve().parent
    handler = partial(NoCacheHandler, directory=str(website))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Open http://localhost:{args.port}/transformer_visualizer.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
