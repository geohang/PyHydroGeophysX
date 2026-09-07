#!/usr/bin/env python3
"""Serve the "Usage and Downloads" dashboard locally for a visual check.

The real page lives in docs/source/usage.rst and needs the full Sphinx toolchain
(pydata-sphinx-theme, sphinx-gallery) to build. This script skips that: it
serves docs/source over HTTP and synthesizes a small host page that loads the
same _static/usage-stats.css and _static/usage-stats.js against the same
_static/stats/*.json. What renders here is what renders on the site, minus the
surrounding theme chrome.

The host page is generated in memory, so nothing extra is written into the
repository.

Usage
-----
    python tools/preview_usage.py                  # open http://localhost:8731/
    python tools/preview_usage.py --port 9000
    python tools/preview_usage.py --no-open

Query parameters on the preview page
------------------------------------
    ?visitors=demo   inject synthetic analytics data, so the visitor map, the
                     most-read-pages panel, and the click ranking can be checked
                     before a GoatCounter site is connected
    ?stale=1         replace the committed snapshot with obviously wrong values
                     (1 download, Antarctica only), so a successful live refresh
                     is unmistakable: real numbers on screen means the live
                     ClickHouse query ran
    ?live=off        skip the live refresh and show the committed snapshot alone
    ?traffic=demo    inject GitHub clone and view counts, which only appear when
                     the collector runs with a token. Combine with ?stale=1 to
                     check that a snapshot older than the 14-day window those
                     figures describe drops the tiles instead of mislabelling
                     them
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

DOCS_SOURCE = Path(__file__).resolve().parents[1] / "docs" / "source"

HOST_PAGE = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Usage dashboard preview</title>
<style>
  /* Stand-ins for the pydata-sphinx-theme custom properties the real page has,
     so the light and dark palettes match what the site will show. */
  :root {
    --pst-color-surface: #f6f7f9; --pst-color-border: #d7dbe0;
    --pst-color-text-base: #1a1c1e; --pst-color-text-muted: #6a7076;
    --pst-color-primary: #1f6feb; --pst-color-background: #ffffff;
  }
  html[data-theme="dark"] {
    --pst-color-surface: #21262d; --pst-color-border: #373e47;
    --pst-color-text-base: #e6edf3; --pst-color-text-muted: #8b949e;
    --pst-color-primary: #58a6ff; --pst-color-background: #14181d;
  }
  body {
    margin: 0 auto; padding: 1.5rem 2rem 4rem; max-width: 1100px;
    background: var(--pst-color-background); color: var(--pst-color-text-base);
    font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
  }
  h1 { font-size: 1.9rem; margin: 0.4rem 0 0.2rem; }
  .lede { color: var(--pst-color-text-muted); margin: 0 0 1rem; }
  .bar {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    padding: 0.6rem 0.8rem; margin-bottom: 1.2rem; font-size: 0.82rem;
    background: var(--pst-color-surface); border: 1px solid var(--pst-color-border);
    border-radius: 8px;
  }
  .bar strong { font-weight: 600; }
  .bar a, .bar button {
    font: inherit; font-size: 0.82rem; color: var(--pst-color-primary);
    background: transparent; border: 1px solid var(--pst-color-border);
    border-radius: 6px; padding: 0.2rem 0.6rem; cursor: pointer;
    text-decoration: none;
  }
  .bar span.sep { flex: 1 1 auto; }
</style>
<link rel="stylesheet" href="_static/usage-stats.css">
</head>
<body>

<div class="bar">
  <strong>Preview</strong>
  <button id="themer" type="button">Toggle dark mode</button>
  <a href="/">Real data</a>
  <a href="/?visitors=demo">+ demo visitors</a>
  <a href="/?stale=1">Prove live refresh</a>
  <a href="/?live=off">Snapshot only</a>
  <a href="/?traffic=demo">+ GitHub traffic</a>
  <a href="/?traffic=demo&amp;stale=1">Traffic + stale snapshot</a>
  <span class="sep"></span>
  <span id="mode"></span>
</div>

<h1>Usage and Downloads</h1>
<p class="lede">
  Where PyHydroGeophysX is downloaded, how often, and how the documentation is
  read. Download counts update live on every page load; the rest comes from a
  monthly snapshot.
</p>

<div id="phgx-stats"><p>Loading usage statistics&hellip;</p></div>

<script>
var params = new URLSearchParams(location.search);

document.getElementById('themer').addEventListener('click', function () {
  var root = document.documentElement;
  root.setAttribute('data-theme', root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
});

var notes = [];
if (params.get('visitors') === 'demo') notes.push('synthetic visitor data');
if (params.get('stale') === '1') notes.push('snapshot deliberately wrong');
if (params.get('live') === 'off') notes.push('live refresh disabled');
if (params.get('traffic') === 'demo') notes.push('synthetic GitHub traffic');
document.getElementById('mode').textContent = notes.length
  ? notes.join(' \\u00b7 ')
  : 'real committed snapshot plus live refresh';

if (params.get('live') === 'off') {
  document.getElementById('phgx-stats').setAttribute('data-live', 'off');
}

// Rewrite the snapshot before usage-stats.js reads it, for the two test modes.
if (params.get('stale') === '1' || params.get('visitors') === 'demo' || params.get('traffic') === 'demo') {
  var realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    var url = typeof input === 'string' ? input : input.url;
    if (!/usage_stats\\.json$/.test(url)) return realFetch(input, init);
    return realFetch(input, init)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (params.get('stale') === '1') {
          data.generated_at = '2000-01-01T00:00:00+00:00';
          if (data.pypi) {
            data.pypi.total = 1;
            data.pypi.last_30d = 1;
            data.pypi.countries_reached = 1;
            data.pypi.by_country = { AQ: 1 };
            data.pypi.by_month = { '2000-01': 1 };
          }
          if (data.github) { data.github.stars = 1; data.github.forks = 1; }
        }
        if (params.get('traffic') === 'demo' && data.github) {
          data.github.clones_14d = { total: 37, uniques: 21, daily: {} };
          data.github.views_14d = { total: 412, uniques: 96, daily: {} };
        }
        if (params.get('visitors') === 'demo') {
          data.visitors = {
            window_start: '2025-09-06', window_end: '2026-09-06',
            visitors: 4213, events: 268, countries_reached: 63,
            by_country: { US: 1580, CN: 690, DE: 310, GB: 240, IN: 205, JP: 150,
              FR: 140, CA: 130, BR: 96, IT: 88, AU: 80, KR: 74, NL: 60, ES: 55,
              CH: 44, SE: 40, SG: 36, TR: 30, MX: 28, ZA: 22, NO: 18, KE: 6 },
            top_pages: [
              { path: '/', title: 'PyHydroGeophysX', count: 980, event: false },
              { path: '/installation.html', title: 'Installation', count: 640, event: false },
              { path: '/quickstart.html', title: 'Quickstart', count: 505, event: false },
              { path: '/auto_examples/index.html', title: 'Examples Gallery', count: 430, event: false },
              { path: '/api/index.html', title: 'API Reference', count: 300, event: false },
              { path: '/agents/desktop_studio.html', title: 'Desktop Studio', count: 190, event: false },
              { path: 'click/example-download/Ex2_workflow.py', title: 'example-download', count: 120, event: true },
              { path: 'click/github/PyHydroGeophysX', title: 'github', count: 88, event: true },
              { path: 'click/pypi/pyhydrogeophysx', title: 'pypi', count: 60, event: true }
            ]
          };
        }
        return new Response(JSON.stringify(data), {
          headers: { 'Content-Type': 'application/json' }
        });
      });
  };
}
</script>
<script src="_static/usage-stats.js"></script>
</body>
</html>
"""


class PreviewHandler(http.server.SimpleHTTPRequestHandler):
    """Serve docs/source, with the generated host page at the site root."""

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
        if self.path.split("?", 1)[0] in ("/", "/index.html", "/preview"):
            body = HOST_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt: str, *args: object) -> None:
        # One line per request is noise while clicking around a dashboard.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8731, help="port to listen on")
    parser.add_argument(
        "--no-open", action="store_true", help="do not open a browser window"
    )
    args = parser.parse_args()

    stats = DOCS_SOURCE / "_static" / "stats" / "usage_stats.json"
    if not stats.exists():
        print(
            "No statistics snapshot yet. Run this first:\n"
            "    python tools/fetch_usage_stats.py"
        )
        return 1

    handler = functools.partial(PreviewHandler, directory=str(DOCS_SOURCE))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://localhost:{args.port}/"
        print(f"Serving {DOCS_SOURCE}")
        print(f"Preview at {url}")
        print("  /?visitors=demo   synthetic analytics data")
        print("  /?stale=1         prove the live refresh replaces the snapshot")
        print("  /?live=off        committed snapshot only")
        print("Press Ctrl+C to stop.")
        if not args.no_open:
            threading.Timer(0.4, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
