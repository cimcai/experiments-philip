"""Tiny range-aware static file server for the viewer.

Python's stdlib http.server doesn't support HTTP range requests, which means
browsers can't seek into a 35 MB video. This subclass adds range support so
<video> playback works correctly.

Run:  uv run python pipeline/viewer/serve.py
"""

from __future__ import annotations

import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):  # type: ignore[override]
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        rng = self.headers.get("Range")
        size = os.path.getsize(path)
        ctype = self.guess_type(path)

        if rng is None:
            try:
                f = open(path, "rb")
            except OSError:
                self.send_error(404)
                return None
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f

        # Parse "bytes=start-end" or "bytes=start-".
        try:
            unit, _, spec = rng.partition("=")
            if unit.strip() != "bytes":
                raise ValueError("only bytes ranges supported")
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            if start > end or end >= size:
                raise ValueError("range not satisfiable")
        except ValueError:
            self.send_error(416, "Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        self._range_remaining = end - start + 1
        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):  # type: ignore[override]
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        chunk = 64 * 1024
        while remaining > 0:
            data = source.read(min(chunk, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)


def main() -> None:
    here = Path(__file__).parent
    os.chdir(here)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    httpd = ThreadingHTTPServer(("127.0.0.1", port), RangeHandler)
    print(f"serving {here} at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
