#!/usr/bin/env python3
"""Serve a local video gallery with byte ranges and static-asset caching."""

import argparse
import functools
import http.server
import os
from pathlib import Path


class GalleryHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        self._byte_range: tuple[int, int] | None = None
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        suffix = Path(self.path.split("?", 1)[0]).suffix.lower()
        if suffix in {".mp4", ".jpg", ".jpeg", ".webp", ".png"}:
            self.send_header("Cache-Control", "public, max-age=604800, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        if suffix == ".mp4":
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        range_header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not range_header or not os.path.isfile(path):
            self._byte_range = None
            return super().send_head()

        try:
            start, end = self._parse_range(range_header, os.path.getsize(path))
        except ValueError:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        file = open(path, "rb")
        stat = os.fstat(file.fileno())
        self._byte_range = (start, end)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        return file

    @staticmethod
    def _parse_range(value: str, size: int) -> tuple[int, int]:
        if not value.startswith("bytes=") or "," in value or size <= 0:
            raise ValueError(value)
        first, last = value[6:].split("-", 1)
        if first:
            start = int(first)
            end = int(last) if last else size - 1
        else:
            length = int(last)
            if length <= 0:
                raise ValueError(value)
            start = max(0, size - length)
            end = size - 1
        if start < 0 or start >= size or end < start:
            raise ValueError(value)
        return start, min(end, size - 1)

    def copyfile(self, source, outputfile) -> None:
        if self._byte_range is None:
            return super().copyfile(source, outputfile)
        start, end = self._byte_range
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    handler = functools.partial(GalleryHandler, directory=str(args.root.resolve()))
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {args.root.resolve()} at http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
