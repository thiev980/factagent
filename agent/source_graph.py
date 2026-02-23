"""
FactAgent – Source Graph Visualization
=======================================
Generiert eine interaktive Netzwerk-Visualisierung, die zeigt,
wie Quellen mit Teilaussagen zusammenhängen.

AI-Engineering-Pattern: Explainability / Transparency
- Macht die Entscheidungsgrundlage des Agenten sichtbar
- User kann nachvollziehen, welche Quellen was belegen
- Glaubwürdigkeit und Relevanz werden visuell kodiert

Technologie: vis.js Network (geladen via CDN, kein Build nötig)

Nodes:
  - Rechtecke = Teilaussagen (Farbe = Verdikt)
  - Kreise = Quellen (Farbe = Glaubwürdigkeit, Grösse = Relevanz)

Edges:
  - Quelle → Teilaussage (welche Quelle belegt welche Aussage)
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from agent.models import FactCheckResult, Verdict, Credibility

logger = logging.getLogger(__name__)

# Ausgabe-Verzeichnis für generierte HTML-Dateien
OUTPUT_DIR = Path(__file__).parent.parent / "graphs"


# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------

VERDICT_COLORS = {
    Verdict.TRUE: "#22c55e",           # Grün
    Verdict.FALSE: "#ef4444",          # Rot
    Verdict.PARTIALLY_TRUE: "#eab308", # Gelb
    Verdict.MISLEADING: "#f97316",     # Orange
    Verdict.UNVERIFIABLE: "#9ca3af",   # Grau
}

VERDICT_LABELS = {
    Verdict.TRUE: "Wahr",
    Verdict.FALSE: "Falsch",
    Verdict.PARTIALLY_TRUE: "Teilw. wahr",
    Verdict.MISLEADING: "Irreführend",
    Verdict.UNVERIFIABLE: "Unklar",
}

CREDIBILITY_COLORS = {
    Credibility.HIGH: "#22c55e",    # Grün
    Credibility.MEDIUM: "#eab308",  # Gelb
    Credibility.LOW: "#ef4444",     # Rot
}

CREDIBILITY_LABELS = {
    Credibility.HIGH: "Hoch",
    Credibility.MEDIUM: "Mittel",
    Credibility.LOW: "Niedrig",
}


def _source_id(url: str) -> str:
    """Erstellt eine kurze, deterministische ID aus einer URL."""
    return "src_" + hashlib.md5(url.encode()).hexdigest()[:8]


def _truncate(text: str, max_len: int = 50) -> str:
    """Kürzt Text auf max_len Zeichen mit Ellipse."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Graph-Daten aus FactCheckResult extrahieren
# ---------------------------------------------------------------------------

def build_graph_data(result: FactCheckResult) -> dict:
    """
    Extrahiert Nodes und Edges aus einem FactCheckResult.
    
    Returns:
        Dict mit "nodes" und "edges" Listen für vis.js
    """
    nodes = []
    edges = []
    seen_sources = {}  # url → node_id

    # 1) Sub-Claim Nodes (Rechtecke)
    for i, sv in enumerate(result.sub_verdicts):
        claim_id = f"claim_{i}"
        color = VERDICT_COLORS.get(sv.verdict, "#9ca3af")
        verdict_label = VERDICT_LABELS.get(sv.verdict, "?")

        nodes.append({
            "id": claim_id,
            "label": _truncate(sv.claim, 60),
            "title": (
                f"<b>{sv.claim}</b><br>"
                f"Verdikt: {verdict_label}<br>"
                f"Konfidenz: {sv.confidence:.0%}<br><br>"
                f"<i>{sv.reasoning}</i>"
            ),
            "shape": "box",
            "color": {
                "background": color,
                "border": color,
                "highlight": {"background": color, "border": "#000"},
            },
            "font": {"color": "#fff", "size": 14, "face": "Arial"},
            "margin": 12,
            "group": "claims",
        })

        # 2) Source Nodes (Kreise) + Edges zu Sub-Claims
        for source in sv.evidence:
            src_id = _source_id(source.url)

            # Source-Node nur einmal erstellen
            if source.url not in seen_sources:
                seen_sources[source.url] = src_id
                cred_color = CREDIBILITY_COLORS.get(source.credibility, "#9ca3af")
                cred_label = CREDIBILITY_LABELS.get(source.credibility, "?")

                # Grösse basierend auf Relevanz (15-40px)
                size = 15 + int(source.relevance_score * 25)

                domain = source.url.split("/")[2] if "/" in source.url else source.url

                nodes.append({
                    "id": src_id,
                    "label": _truncate(source.title, 35),
                    "title": (
                        f"<b>{source.title}</b><br>"
                        f"🌐 {domain}<br>"
                        f"Glaubwürdigkeit: {cred_label}<br>"
                        f"Relevanz: {source.relevance_score:.0%}<br><br>"
                        f"<i>{source.snippet[:200]}</i><br><br>"
                        f"<a href='{source.url}' target='_blank'>Quelle öffnen →</a>"
                    ),
                    "shape": "dot",
                    "size": size,
                    "color": {
                        "background": cred_color,
                        "border": cred_color,
                        "highlight": {"background": cred_color, "border": "#000"},
                    },
                    "font": {"size": 11, "face": "Arial"},
                    "group": "sources",
                    "url": source.url,
                })

            # Edge: Source → Sub-Claim
            edges.append({
                "from": src_id,
                "to": claim_id,
                "color": {"color": "#999", "highlight": "#333"},
                "width": max(1, source.relevance_score * 3),
                "smooth": {"type": "curvedCW", "roundness": 0.2},
            })

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# HTML generieren
# ---------------------------------------------------------------------------

def generate_graph_html(result: FactCheckResult, claim: str = "") -> str:
    """
    Generiert eine komplette HTML-Datei mit vis.js Network-Graph.
    
    Args:
        result: Das FactCheckResult
        claim: Die ursprüngliche Behauptung (für den Titel)
    
    Returns:
        Pfad zur generierten HTML-Datei
    """
    graph_data = build_graph_data(result)
    nodes_json = json.dumps(graph_data["nodes"], ensure_ascii=False)
    edges_json = json.dumps(graph_data["edges"], ensure_ascii=False)

    # Verdikt-Info für Header
    verdict_label = VERDICT_LABELS.get(result.overall_verdict, "?")
    verdict_color = VERDICT_COLORS.get(result.overall_verdict, "#999")

    # Stats
    n_sources = len([n for n in graph_data["nodes"] if n.get("group") == "sources"])
    n_claims = len([n for n in graph_data["nodes"] if n.get("group") == "claims"])

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FactAgent – Source Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
        }}
        #header {{
            padding: 20px 30px;
            background: #1e293b;
            border-bottom: 2px solid {verdict_color};
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        #header h1 {{
            font-size: 18px;
            font-weight: 600;
        }}
        #header .claim {{
            font-size: 14px;
            color: #94a3b8;
            max-width: 500px;
        }}
        #header .verdict {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            color: #fff;
            background: {verdict_color};
        }}
        #header .stats {{
            font-size: 13px;
            color: #94a3b8;
        }}
        #graph-container {{
            width: 100%;
            height: calc(100vh - 140px);
        }}
        #legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px;
            font-size: 12px;
            z-index: 10;
            max-width: 240px;
        }}
        #legend h3 {{
            font-size: 13px;
            margin-bottom: 10px;
            color: #cbd5e1;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 6px 0;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-box {{
            width: 16px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        .legend-divider {{
            height: 1px;
            background: #334155;
            margin: 10px 0;
        }}
        #tooltip {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px;
            font-size: 13px;
            z-index: 10;
            max-width: 350px;
            display: none;
        }}
        #tooltip h3 {{
            font-size: 14px;
            margin-bottom: 8px;
        }}
        #tooltip a {{
            color: #60a5fa;
            text-decoration: none;
        }}
        #tooltip a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h1>🔍 FactAgent – Source Graph</h1>
            <div class="claim">«{_truncate(claim or result.original_claim, 120)}»</div>
        </div>
        <div style="text-align: right;">
            <span class="verdict">{verdict_label} ({result.confidence:.0%})</span>
            <div class="stats">{n_claims} Teilaussagen · {n_sources} Quellen · {len(graph_data['edges'])} Verbindungen</div>
        </div>
    </div>

    <div id="graph-container"></div>

    <div id="legend">
        <h3>Legende</h3>
        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 8px;">Teilaussagen (Rechtecke)</div>
        <div class="legend-item"><span class="legend-box" style="background:#22c55e"></span> Wahr</div>
        <div class="legend-item"><span class="legend-box" style="background:#eab308"></span> Teilweise wahr</div>
        <div class="legend-item"><span class="legend-box" style="background:#f97316"></span> Irreführend</div>
        <div class="legend-item"><span class="legend-box" style="background:#ef4444"></span> Falsch</div>
        <div class="legend-item"><span class="legend-box" style="background:#9ca3af"></span> Nicht überprüfbar</div>
        <div class="legend-divider"></div>
        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 8px;">Quellen (Kreise)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#22c55e"></span> Hohe Glaubwürdigkeit</div>
        <div class="legend-item"><span class="legend-dot" style="background:#eab308"></span> Mittlere Glaubwürdigkeit</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ef4444"></span> Niedrige Glaubwürdigkeit</div>
        <div class="legend-divider"></div>
        <div style="font-size: 11px; color: #94a3b8;">Kreisgrösse = Relevanz<br>Linienstärke = Relevanz</div>
    </div>

    <div id="tooltip">
        <h3 id="tooltip-title"></h3>
        <div id="tooltip-content"></div>
    </div>

    <script>
        // Graph-Daten (von Python generiert)
        const nodesData = {nodes_json};
        const edgesData = {edges_json};

        // vis.js Network erstellen
        const container = document.getElementById('graph-container');
        const data = {{
            nodes: new vis.DataSet(nodesData),
            edges: new vis.DataSet(edgesData),
        }};

        const options = {{
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -80,
                    centralGravity: 0.01,
                    springLength: 180,
                    springConstant: 0.06,
                    damping: 0.4,
                    avoidOverlap: 0.5,
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 200,
                    fit: true,
                }},
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
            }},
            layout: {{
                improvedLayout: true,
            }},
        }};

        const network = new vis.Network(container, data, options);

        // Tooltip bei Hover
        const tooltip = document.getElementById('tooltip');
        const tooltipTitle = document.getElementById('tooltip-title');
        const tooltipContent = document.getElementById('tooltip-content');

        network.on('hoverNode', function(params) {{
            const node = nodesData.find(n => n.id === params.node);
            if (node && node.title) {{
                tooltipTitle.textContent = node.label;
                tooltipContent.innerHTML = node.title;
                tooltip.style.display = 'block';
            }}
        }});

        network.on('blurNode', function() {{
            tooltip.style.display = 'none';
        }});

        // Klick auf Quelle → URL öffnen
        network.on('doubleClick', function(params) {{
            if (params.nodes.length > 0) {{
                const node = nodesData.find(n => n.id === params.nodes[0]);
                if (node && node.url) {{
                    window.open(node.url, '_blank');
                }}
            }}
        }});

        // Fit nach Stabilisierung
        network.on('stabilizationIterationsDone', function() {{
            network.fit({{ animation: {{ duration: 500 }} }});
        }});
    </script>
</body>
</html>"""

    # Datei speichern
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = f"source_graph_{hashlib.md5(claim.encode()).hexdigest()[:8]}.html"
    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"📊 Source Graph generiert: {filepath}")
    return str(filepath)
