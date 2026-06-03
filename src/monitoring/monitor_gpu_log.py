#!/usr/bin/env python3
"""Simple read-only web viewer for a growing log file."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def tail_lines(path: str, lines: int) -> str:
    if lines <= 0:
        lines = 200
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except FileNotFoundError:
        return f"[monitor] Log file not found: {path}\n"
    except Exception as exc:  # pragma: no cover
        return f"[monitor] Error reading log file: {exc}\n"


def build_html(default_lines: int, refresh_ms: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GPU Log Monitor</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #111a2b;
      --text: #d7e3ff;
      --muted: #8ea5d6;
      --accent: #4fc3f7;
      --ok: #93e5ab;
      --border: #223250;
    }}
    body {{
      margin: 0;
      font-family: "Menlo", "Consolas", monospace;
      background: linear-gradient(130deg, #0a1020 0%, #11182b 55%, #0f1f2e 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 12px auto;
      padding: 0 12px 12px;
    }}
    .bar {{
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr auto auto auto;
      align-items: center;
      background: rgba(17,26,43,0.95);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      position: sticky;
      top: 8px;
      backdrop-filter: blur(3px);
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    input, button {{
      background: #0b1322;
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
    }}
    input {{ width: 90px; }}
    button {{
      background: #10253b;
      cursor: pointer;
    }}
    pre {{
      margin: 12px 0 0;
      background: rgba(17,26,43,0.92);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      font-size: 13px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 70vh;
    }}
    .ok {{ color: var(--ok); }}
    .accent {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div class="status" id="status">Connecting...</div>
      <input type="number" min="20" max="5000" id="lines" value="{default_lines}" />
      <button id="refresh">Refresh</button>
      <button id="autoscroll">Auto-scroll: on</button>
    </div>
    <pre id="log">Loading log...</pre>
  </div>
  <script>
    const logEl = document.getElementById("log");
    const statusEl = document.getElementById("status");
    const linesEl = document.getElementById("lines");
    const refreshBtn = document.getElementById("refresh");
    const autoBtn = document.getElementById("autoscroll");
    let autoScroll = true;
    let timer = null;

    autoBtn.onclick = () => {{
      autoScroll = !autoScroll;
      autoBtn.textContent = "Auto-scroll: " + (autoScroll ? "on" : "off");
    }};

    async function fetchLog() {{
      const lines = Math.max(20, Math.min(5000, Number(linesEl.value || "{default_lines}")));
      try {{
        const url = `/api/log?lines=${{lines}}&t=${{Date.now()}}`;
        const res = await fetch(url, {{ cache: "no-store" }});
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        logEl.textContent = data.content;
        const mb = (data.size / (1024 * 1024)).toFixed(2);
        statusEl.innerHTML = `<span class="ok">live</span> | updated ${{data.updated}} | size <span class="accent">${{mb}} MB</span>`;
        if (autoScroll) window.scrollTo(0, document.body.scrollHeight);
      }} catch (err) {{
        statusEl.textContent = "Error: " + err.message;
      }}
    }}

    function loop() {{
      fetchLog();
      timer = setTimeout(loop, {refresh_ms});
    }}

    refreshBtn.onclick = fetchLog;
    loop();
  </script>
</body>
</html>
"""


def make_handler(log_file: str, default_lines: int, refresh_ms: int):
    html = build_html(default_lines, refresh_ms).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if parsed.path == "/api/log":
                query = parse_qs(parsed.query)
                lines = default_lines
                if "lines" in query:
                    try:
                        lines = int(query["lines"][0])
                    except Exception:
                        lines = default_lines
                content = tail_lines(log_file, lines)
                st = os.stat(log_file) if os.path.exists(log_file) else None
                payload = {
                    "content": content,
                    "size": (st.st_size if st else 0),
                    "updated": (
                        datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        if st
                        else "log file not found"
                    ),
                }
                out = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a live viewer for a log file.")
    p.add_argument("--log-file", default="gpu_freq_experiment.log")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--lines", type=int, default=300)
    p.add_argument("--refresh-ms", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    handler = make_handler(args.log_file, args.lines, args.refresh_ms)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[monitor] Serving {args.log_file} at http://{args.host}:{args.port}")
    print("[monitor] Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
