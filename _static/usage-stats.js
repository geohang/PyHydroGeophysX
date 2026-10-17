/* Renders the "Usage and Downloads" dashboard.
 *
 * Reads two static files that sit next to this script:
 *   stats/usage_stats.json     written by tools/fetch_usage_stats.py
 *   stats/world-110m.geo.json  Natural Earth 110m country outlines (public domain)
 *
 * Draws the world map as inline SVG using interpolated Robinson coefficients.
 * Live refresh also queries ClickHouse unless data-live="off" is set on the
 * container. If GoatCounter is already active, trackClicks records selected
 * link clicks independently of that live-refresh setting.
 */

(function () {
  "use strict";

  var SCRIPT_SRC = (document.currentScript && document.currentScript.src) || "";

  // ---------------------------------------------------------------- colours

  var RAMP_LIGHT = ["#e3eefb", "#a8cbee", "#5f9ede", "#2c6db5", "#123f6d"];
  var RAMP_DARK = ["#1d3448", "#2b5a86", "#3f86c0", "#63aee4", "#9ed3ff"];

  function isDark() {
    return document.documentElement.getAttribute("data-theme") === "dark";
  }

  function ramp() {
    return isDark() ? RAMP_DARK : RAMP_LIGHT;
  }

  function hexToRgb(hex) {
    var n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function rampColor(t) {
    var stops = ramp();
    var clamped = Math.max(0, Math.min(1, t));
    var scaled = clamped * (stops.length - 1);
    var i = Math.min(Math.floor(scaled), stops.length - 2);
    var f = scaled - i;
    var a = hexToRgb(stops[i]);
    var b = hexToRgb(stops[i + 1]);
    return (
      "rgb(" +
      Math.round(a[0] + f * (b[0] - a[0])) + "," +
      Math.round(a[1] + f * (b[1] - a[1])) + "," +
      Math.round(a[2] + f * (b[2] - a[2])) + ")"
    );
  }

  // Use log1p scaling to distinguish small counts when the range is wide;
  // the legend identifies the logarithmic scale.
  function intensity(value, max) {
    if (!value || max <= 0) return 0;
    return Math.log(1 + value) / Math.log(1 + max);
  }

  // ------------------------------------------------------------- projection

  var ROBINSON_X = [
    1.0, 0.9986, 0.9954, 0.99, 0.9822, 0.973, 0.96, 0.9427, 0.9216, 0.8962,
    0.8679, 0.835, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322
  ];
  var ROBINSON_Y = [
    0.0, 0.062, 0.124, 0.186, 0.248, 0.31, 0.372, 0.434, 0.4958, 0.5571,
    0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0
  ];

  var MAP_W = 1000;
  var MAP_SCALE = MAP_W / (2 * 0.8487 * Math.PI);
  var MAP_H = 2 * 1.3523 * MAP_SCALE;

  function project(lon, lat) {
    var abs = Math.min(Math.abs(lat), 90);
    var i = Math.min(Math.floor(abs / 5), 17);
    var f = (abs - i * 5) / 5;
    var xf = ROBINSON_X[i] + f * (ROBINSON_X[i + 1] - ROBINSON_X[i]);
    var yf = ROBINSON_Y[i] + f * (ROBINSON_Y[i + 1] - ROBINSON_Y[i]);
    var x = 0.8487 * xf * ((lon * Math.PI) / 180);
    var y = 1.3523 * yf * (lat < 0 ? -1 : 1);
    return [MAP_W / 2 + x * MAP_SCALE, MAP_H / 2 - y * MAP_SCALE];
  }

  function pathData(geometry) {
    var out = [];
    var polygons =
      geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
    for (var p = 0; p < polygons.length; p++) {
      for (var r = 0; r < polygons[p].length; r++) {
        var ring = polygons[p][r];
        var pts = [];
        for (var k = 0; k < ring.length; k++) {
          var xy = project(ring[k][0], ring[k][1]);
          pts.push(xy[0].toFixed(1) + "," + xy[1].toFixed(1));
        }
        if (pts.length > 2) {
          out.push("M" + pts[0] + "L" + pts.slice(1).join(" ") + "Z");
        }
      }
    }
    return out.join("");
  }

  // ------------------------------------------------------------- formatting

  var regionNames = null;
  try {
    regionNames = new Intl.DisplayNames(["en"], { type: "region" });
  } catch (err) {
    regionNames = null;
  }

  var geoNames = {};

  // Site-specific display labels take precedence over browser and basemap names
  // so the map and ranking list use the same labels.
  var NAME_OVERRIDES = {
    HK: "Hong Kong (China)",
    MO: "Macao (China)"
  };

  // Territory codes folded into another entry, read from the statistics JSON so
  // the collector and this file cannot drift apart. A folded territory gets no
  // row of its own, and its polygon is filled and labelled as the entry it
  // belongs to.
  var MERGES = {};

  function mergeTarget(code) {
    return MERGES[code] || code;
  }

  function foldTerritories(byCode) {
    var out = {};
    for (var code in byCode) {
      var target = mergeTarget(code);
      out[target] = (out[target] || 0) + byCode[code];
    }
    return out;
  }

  /* Name resolution order, most specific first:
   *
   *   1. NAME_OVERRIDES, for the labels this site fixes by hand.
   *   2. Intl.DisplayNames, when the browser resolves the code to a name.
   *   3. The basemap name, when the API is unavailable, throws, or returns
   *      the unresolved code.
   *   4. The bare two-letter code, for anything none of the above resolves.
   */
  function countryName(code) {
    if (NAME_OVERRIDES[code]) return NAME_OVERRIDES[code];
    if (regionNames) {
      try {
        var name = regionNames.of(code);
        if (name && name !== code) return name;
      } catch (err) {
        /* an unassigned code falls through to the basemap name below */
      }
    }
    if (geoNames[code]) return geoNames[code];
    return code;
  }

  function num(value) {
    return (value == null ? 0 : value).toLocaleString("en-US");
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function entriesDesc(obj) {
    return Object.keys(obj || {})
      .map(function (k) {
        return [k, obj[k]];
      })
      .sort(function (a, b) {
        return b[1] - a[1];
      });
  }

  // ------------------------------------------------------------------ tiles

  var reduceMotion = false;
  try {
    reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch (err) {
    reduceMotion = false;
  }

  function countUp(node, target) {
    if (reduceMotion || target < 2) {
      node.textContent = num(target);
      return;
    }
    var started = null;
    var duration = 650;
    function step(now) {
      if (started === null) started = now;
      var t = Math.min(1, (now - started) / duration);
      var eased = 1 - Math.pow(1 - t, 3);
      node.textContent = num(Math.round(target * eased));
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // Tiles that the live refresh can rewrite are registered here by key.
  var TILES = {};

  // Handles the live refresh needs in order to repaint what is already on screen.
  var STATE = {};

  function addTile(parent, value, label, key) {
    if (value == null) return;
    var tile = el("div", "phgx-tile");
    var v = el("div", "phgx-tile-value", "0");
    tile.appendChild(v);
    tile.appendChild(el("div", "phgx-tile-label", label));
    parent.appendChild(tile);
    countUp(v, value);
    if (key) TILES[key] = v;
  }

  function setTile(key, value) {
    if (value == null || !TILES[key]) return;
    TILES[key].textContent = num(value);
  }

  function renderTiles(root, stats) {
    var tiles = el("div", "phgx-tiles");
    var pypi = stats.pypi;
    var github = stats.github;
    var visitors = stats.visitors;

    if (pypi) {
      addTile(tiles, pypi.total, "PyPI downloads, all time", "total");
      addTile(tiles, pypi.last_30d, "PyPI downloads, last 30 days", "last_30d");
      addTile(tiles, pypi.countries_reached, "Countries and territories", "countries");
    }
    if (github) {
      // Stars and forks are collected into the snapshot but deliberately not
      // shown: they measure attention on GitHub rather than use of the package.
      //
      // GitHub's traffic endpoint only ever reports a rolling 14-day window, and
      // it needs a token, so these two figures can only come from the snapshot.
      // The snapshot is rebuilt monthly, so for most of any given month that
      // window has already closed. Naming the end date keeps the tiles both
      // permanent and accurate. Hiding them past some freshness cutoff instead
      // would make them appear for two weeks and then vanish, which a reader
      // would take for a bug.
      var trafficWindow = stats.generated_at
        ? "14 days to " + shortDate(stats.generated_at)
        : "last 14 days";
      if (github.clones_14d) {
        addTile(tiles, github.clones_14d.total, "Repository clones, " + trafficWindow);
      }
      if (github.views_14d) {
        addTile(tiles, github.views_14d.total, "Repository views, " + trafficWindow);
      }
    }
    if (visitors) {
      addTile(tiles, visitors.visitors, "Documentation visits, last year");
      addTile(tiles, visitors.countries_reached, "Visitor countries");
      addTile(tiles, visitors.events, "Tracked link clicks, last year");
    }
    root.appendChild(tiles);
  }

  // -------------------------------------------------------------------- map

  function buildMap(geo, panel) {
    var wrap = el("div", "phgx-map");
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + MAP_W + " " + Math.round(MAP_H));
    svg.setAttribute("role", "img");

    var paths = {};
    for (var i = 0; i < geo.features.length; i++) {
      var feature = geo.features[i];
      var iso = feature.properties.iso;
      geoNames[iso] = feature.properties.name;
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathData(feature.geometry));
      path.setAttribute("data-iso", iso);
      svg.appendChild(path);
      // Natural Earth splits some states across several features; keep them all.
      (paths[iso] = paths[iso] || []).push(path);
    }

    var tooltip = el("div", "phgx-tooltip");
    wrap.appendChild(svg);
    wrap.appendChild(tooltip);
    panel.appendChild(wrap);

    // Antarctica is left out, so the full Robinson frame carries a wide band of
    // empty polar ocean. Crop to what was actually drawn, which also keeps the
    // map correct if the basemap is ever rebuilt at a different extent.
    try {
      var box = svg.getBBox();
      if (box && box.width > 0 && box.height > 0) {
        var pad = 6;
        svg.setAttribute(
          "viewBox",
          [
            (box.x - pad).toFixed(1),
            (box.y - pad).toFixed(1),
            (box.width + 2 * pad).toFixed(1),
            (box.height + 2 * pad).toFixed(1)
          ].join(" ")
        );
      }
    } catch (err) {
      /* keep the full frame if the browser cannot measure a hidden SVG */
    }

    svg.addEventListener("mousemove", function (event) {
      var target = event.target;
      if (target.tagName !== "path" || !target.hasAttribute("data-label")) {
        tooltip.classList.remove("is-visible");
        return;
      }
      var box = wrap.getBoundingClientRect();
      tooltip.textContent = target.getAttribute("data-label");
      tooltip.style.left = event.clientX - box.left + "px";
      tooltip.style.top = event.clientY - box.top + "px";
      tooltip.classList.add("is-visible");
    });
    svg.addEventListener("mouseleave", function () {
      tooltip.classList.remove("is-visible");
    });

    var legend = el("div", "phgx-legend");
    var lowLabel = el("span", null, "1");
    var bar = el("div", "phgx-legend-bar");
    var highLabel = el("span", null, "");
    legend.appendChild(lowLabel);
    legend.appendChild(bar);
    legend.appendChild(highLabel);
    legend.appendChild(el("span", null, "(log scale)"));
    panel.appendChild(legend);

    return {
      svg: svg,
      paths: paths,
      bar: bar,
      highLabel: highLabel,
      paint: function (byCountry, unit) {
        var max = 0;
        for (var code in byCountry) {
          if (byCountry[code] > max) max = byCountry[code];
        }
        var noData = getComputedStyle(document.getElementById("phgx-stats"))
          .getPropertyValue("--phgx-nodata")
          .trim();

        for (var iso in paths) {
          // A folded territory reads the value and the name of its target, so
          // its polygon lights up with the entry it belongs to.
          var key = mergeTarget(iso);
          var value = byCountry[key] || 0;
          var fill = value ? rampColor(intensity(value, max)) : noData;
          var label = value
            ? countryName(key) + ": " + num(value) + " " + unit
            : null;
          for (var j = 0; j < paths[iso].length; j++) {
            var node = paths[iso][j];
            node.setAttribute("fill", fill);
            if (label) {
              node.setAttribute("data-label", label);
              node.setAttribute("data-value", String(value));
            } else {
              node.removeAttribute("data-label");
              node.removeAttribute("data-value");
            }
          }
        }

        var stops = ramp();
        bar.style.background = "linear-gradient(90deg," + stops.join(",") + ")";
        highLabel.textContent = num(max);
        svg.setAttribute(
          "aria-label",
          "World map of " + unit + " by country, highest " + num(max)
        );
      }
    };
  }

  // -------------------------------------------------------------- rankings

  function renderRank(container, entries, unit, limit, labelFn) {
    var label = labelFn || countryName;
    container.textContent = "";
    var list = el("ul", "phgx-rank");
    var top = entries.slice(0, limit || 10);
    var max = top.length ? top[0][1] : 1;
    top.forEach(function (row) {
      var li = document.createElement("li");
      var fill = el("span", "phgx-rank-fill");
      fill.style.width = Math.max(2, (row[1] / max) * 100) + "%";
      li.appendChild(fill);
      li.appendChild(el("span", "phgx-rank-name", label(row[0])));
      li.appendChild(el("span", "phgx-rank-value", num(row[1])));
      li.title = label(row[0]) + ": " + num(row[1]) + " " + unit;
      list.appendChild(li);
    });
    container.appendChild(list);
  }

  // ----------------------------------------------------------- bar chart

  function renderBars(container, monthly) {
    container.textContent = "";
    var keys = Object.keys(monthly).sort().slice(-18);
    if (!keys.length) return;
    var max = 1;
    keys.forEach(function (k) {
      if (monthly[k] > max) max = monthly[k];
    });

    var bars = el("div", "phgx-bars");
    var axis = el("div", "phgx-bar-axis");
    keys.forEach(function (k, index) {
      var bar = el("div", "phgx-bar");
      bar.style.height = Math.max(2, (monthly[k] / max) * 100) + "%";
      bar.title = k + ": " + num(monthly[k]) + " downloads";
      bars.appendChild(bar);
      // Label every third month so the axis stays legible on a narrow screen.
      axis.appendChild(
        el("span", null, index % 3 === 0 || index === keys.length - 1 ? k.slice(2) : "")
      );
    });
    container.appendChild(bars);
    container.appendChild(axis);
  }

  // ------------------------------------------------------------- assembly

  function panel(root, title, note) {
    var box = el("div", "phgx-panel");
    var head = el("div", "phgx-panel-head");
    head.appendChild(el("div", "phgx-panel-title", title));
    if (note) head.appendChild(el("div", "phgx-panel-note", note));
    box.appendChild(head);
    root.appendChild(box);
    return { box: box, head: head };
  }

  function render(root, stats, geo) {
    root.textContent = "";
    MERGES = stats.territory_merges || {};
    renderTiles(root, stats);

    var pypi = stats.pypi;
    var visitors = stats.visitors;
    var hasVisitors = !!(visitors && visitors.by_country && Object.keys(visitors.by_country).length);

    if (pypi && pypi.by_country) {
      var mapPanel = panel(
        root,
        "Where PyHydroGeophysX Is Downloaded",
        pypi.first_day ? "PyPI, " + pypi.first_day + " to " + pypi.last_day : ""
      );

      var layers = [
        {
          key: "downloads",
          label: "Downloads",
          data: pypi.by_country,
          unit: "downloads"
        }
      ];
      if (hasVisitors) {
        layers.push({
          key: "visitors",
          label: "Docs visits",
          data: visitors.by_country,
          unit: "visits"
        });
      }

      if (layers.length > 1) {
        var toggle = el("div", "phgx-toggle");
        layers.forEach(function (layer) {
          var button = el("button", null, layer.label);
          button.type = "button";
          button.setAttribute("aria-pressed", layer.key === "downloads" ? "true" : "false");
          button.addEventListener("click", function () {
            Array.prototype.forEach.call(toggle.children, function (other) {
              other.setAttribute("aria-pressed", "false");
            });
            button.setAttribute("aria-pressed", "true");
            show(layer);
          });
          toggle.appendChild(button);
        });
        mapPanel.head.appendChild(toggle);
      }

      var layout = el("div", "phgx-map-layout");
      var mapCell = el("div");
      var rankCell = el("div");
      layout.appendChild(mapCell);
      layout.appendChild(rankCell);
      mapPanel.box.appendChild(layout);

      var map = buildMap(geo, mapCell);
      var current = layers[0];

      function show(layer) {
        current = layer;
        map.paint(layer.data, layer.unit);
        renderRank(rankCell, entriesDesc(layer.data), layer.unit, 12);
      }
      show(current);

      STATE.show = show;
      STATE.downloads = layers[0];
      STATE.isShowing = function (layer) {
        return current === layer;
      };

      // pydata-sphinx-theme swaps data-theme on the root element; repaint so the
      // choropleth and the legend follow the reader's light or dark setting.
      new MutationObserver(function () {
        map.paint(current.data, current.unit);
      }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    }

    if (pypi && pypi.by_month) {
      var trend = panel(root, "Downloads per Month", "PyPI");
      var chart = el("div");
      trend.box.appendChild(chart);
      renderBars(chart, pypi.by_month);
      STATE.chart = chart;
    }

    if (hasVisitors && visitors.top_pages && visitors.top_pages.length) {
      var read = panel(
        root,
        "Most Read Pages",
        visitors.window_start + " to " + visitors.window_end
      );
      var readCell = el("div");
      read.box.appendChild(readCell);
      var pageRows = visitors.top_pages
        .filter(function (hit) {
          return !hit.event && hit.path;
        })
        .map(function (hit) {
          return [hit.path, hit.count];
        });
      var pageTitles = {};
      visitors.top_pages.forEach(function (hit) {
        if (hit.path) pageTitles[hit.path] = hit.title || hit.path;
      });
      renderRank(readCell, pageRows, "visits", 12, function (path) {
        return pageTitles[path] || path;
      });

      var clickRows = visitors.top_pages
        .filter(function (hit) {
          return hit.event && hit.path;
        })
        .map(function (hit) {
          return [hit.path, hit.count];
        });
      if (clickRows.length) {
        var clicks = panel(root, "Tracked Clicks", "example downloads and outbound links");
        var clickCell = el("div");
        clicks.box.appendChild(clickCell);
        renderRank(clickCell, clickRows, "clicks", 12, function (path) {
          return String(path).replace(/^click\//, "");
        });
      }
    }

    if (!hasVisitors) {
      var setup = panel(root, "Documentation Visits", "not configured");
      var note = el("div", "phgx-setup");
      note.innerHTML =
        "Not connected yet. Setup steps are in " +
        "<code>tools/fetch_usage_stats.py</code>.";
      setup.box.appendChild(note);
    }

    var foot = el("div", "phgx-foot");
    root.appendChild(foot);
    STATE.foot = foot;
    STATE.snapshotAt = stats.generated_at;
    STATE.excludedMirrors = pypi ? pypi.excluded_mirror_downloads : null;
    // Visitor counts cannot be read from the browser, so they always come from
    // the snapshot. Once the live query lands, the footer would otherwise imply
    // that every figure on the page is current.
    STATE.hasSnapshotOnlyFigures = hasVisitors;
    updateFoot(null);
  }

  function shortDate(iso) {
    return String(iso).slice(0, 10);
  }

  function updateFoot(liveAt) {
    if (!STATE.foot) return;
    var lines = [];
    if (liveAt) {
      lines.push("Live from the PyPI download logs, read " + liveAt + ".");
      if (STATE.hasSnapshotOnlyFigures && STATE.snapshotAt) {
        lines.push("Visitor figures from the snapshot of " + shortDate(STATE.snapshotAt) + ".");
      }
    } else if (STATE.snapshotAt) {
      lines.push(
        "Snapshot built " +
          STATE.snapshotAt.replace("T", " ").replace("+00:00", " UTC") +
          "."
      );
    }
    if (STATE.excludedMirrors) {
      // Expose the excluded-request count so readers can interpret the total.
      lines.push(
        num(STATE.excludedMirrors) + " requests from index mirrors and proxy caches excluded."
      );
    }
    STATE.foot.textContent = lines.join(" ");
  }

  // ---------------------------------------------------------- live refresh

  /* Render the stored snapshot first, then query ClickHouse for PyPI counts.
   * A failed refresh leaves the snapshot visible and logs a console warning;
   * its age depends on the last successful collection and docs deployment.
   * GitHub and documentation-visit counts remain at their snapshot values.
   * Add data-live="off" to the container to disable this refresh. This does
   * not disable the separate GoatCounter click handler.
   */

  var CLICKHOUSE = "https://sql-clickhouse.clickhouse.com/";

  function clickhouse(sql) {
    var query = new URLSearchParams({
      user: "demo",
      default_format: "JSON",
      query: sql
    });
    return fetch(CLICKHOUSE + "?" + query.toString(), { mode: "cors" })
      .then(function (response) {
        if (!response.ok) throw new Error("clickhouse " + response.status);
        return response.json();
      })
      .then(function (payload) {
        return payload.data || [];
      });
  }

  function liveRefresh(stats) {
    var project = stats.project;
    var mirrors = (stats.buckets && stats.buckets.mirror_clients) || [];
    if (!project || !mirrors.length) return;

    // Exclude only mirrors, which is exactly what the collector does, so the
    // live number and the snapshot number are the same measurement.
    var table =
      "pypi.pypi_downloads_per_day_by_version_by_installer_by_type_by_country";
    var where =
      "project = '" + project + "' AND installer NOT IN (" +
      mirrors
        .map(function (name) {
          return "'" + name.replace(/'/g, "") + "'";
        })
        .join(",") +
      ")";

    var byCountrySql =
      "SELECT country_code AS k, sum(count) AS v FROM " + table +
      " WHERE " + where + " AND country_code != '' GROUP BY k ORDER BY v DESC";
    var byMonthSql =
      "SELECT formatDateTime(date, '%Y-%m') AS k, sum(count) AS v FROM " + table +
      " WHERE " + where + " GROUP BY k ORDER BY k";
    var summarySql =
      "SELECT sum(count) AS total, sumIf(count, date >= today() - 30) AS last_30d," +
      " toString(max(date)) AS last_day FROM " + table + " WHERE " + where;

    Promise.all([
      clickhouse(byCountrySql),
      clickhouse(byMonthSql),
      clickhouse(summarySql)
    ])
      .then(function (results) {
        var rawCountry = {};
        results[0].forEach(function (row) {
          rawCountry[row.k] = Number(row.v);
        });
        var byCountry = foldTerritories(rawCountry);
        var byMonth = {};
        results[1].forEach(function (row) {
          byMonth[row.k] = Number(row.v);
        });
        var summary = results[2][0] || {};

        if (!Object.keys(byCountry).length) return;

        setTile("total", Number(summary.total));
        setTile("last_30d", Number(summary.last_30d));
        setTile("countries", Object.keys(byCountry).length);

        if (STATE.downloads) {
          STATE.downloads.data = byCountry;
          // Repaint only if the reader is still looking at the download layer.
          if (STATE.show && STATE.isShowing && STATE.isShowing(STATE.downloads)) {
            STATE.show(STATE.downloads);
          }
        }
        if (STATE.chart) renderBars(STATE.chart, byMonth);

        updateFoot(
          new Date().toISOString().replace("T", " ").slice(0, 16) +
            " UTC, covering PyPI through " + (summary.last_day || "")
        );
      })
      .catch(function (error) {
        if (window.console) console.debug("usage-stats: live refresh skipped", error);
      });
  }

  // -------------------------------------------------------- click counting

  // Optional: when GoatCounter is active, record clicks on the links that say
  // something about how the package is used. Nothing happens without it.
  function trackClicks() {
    if (!window.goatcounter || typeof window.goatcounter.count !== "function") return;
    document.addEventListener("click", function (event) {
      var link = event.target.closest && event.target.closest("a[href]");
      if (!link) return;
      var href = link.getAttribute("href") || "";
      var label = null;
      if (link.closest(".sphx-glr-download")) label = "example-download";
      else if (/\.(zip|whl|tar\.gz|exe|dmg|ipynb|py)(\?|$)/i.test(href)) label = "file-download";
      else if (/github\.com/i.test(href)) label = "github";
      else if (/pypi\.org/i.test(href)) label = "pypi";
      if (!label) return;
      window.goatcounter.count({
        path: "click/" + label + "/" + href.split("/").pop(),
        title: label,
        event: true
      });
    });
  }

  function boot() {
    trackClicks();
    var root = document.getElementById("phgx-stats");
    if (!root) return;

    var statsUrl = new URL("stats/usage_stats.json", SCRIPT_SRC).href;
    var geoUrl = new URL("stats/world-110m.geo.json", SCRIPT_SRC).href;

    Promise.all([
      // Revalidate the snapshot, which also supplies the territory merge table
      // used by the live query. Updates depend on collection and deployment.
      fetch(statsUrl, { cache: "no-cache" }).then(function (r) {
        return r.json();
      }),
      // Let normal HTTP caching handle the less frequently updated basemap.
      fetch(geoUrl).then(function (r) {
        return r.json();
      })
    ])
      .then(function (results) {
        render(root, results[0], results[1]);
        if (root.getAttribute("data-live") !== "off") {
          liveRefresh(results[0]);
        }
      })
      .catch(function (error) {
        root.textContent = "";
        var warn = el("div", "phgx-setup");
        warn.textContent =
          "The usage statistics could not be loaded. Run " +
          "python tools/fetch_usage_stats.py to regenerate them.";
        root.appendChild(warn);
        if (window.console) console.warn("usage-stats:", error);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
