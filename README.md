# 🔍 FactAgent – Interaktiver Faktencheck-Assistent

Ein AI-powered Faktencheck-Tool, das Behauptungen automatisch zerlegt, im Web recherchiert, Quellen bewertet und ein strukturiertes Verdikt liefert.

**[Live Demo →](#)** · **[Portfolio →](#)**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Claude API](https://img.shields.io/badge/LLM-Claude%20API-orange)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green)
![Chainlit](https://img.shields.io/badge/Frontend-Chainlit-purple)

---

## Was macht FactAgent?

Du gibst eine Behauptung ein – FactAgent überprüft sie:

```
Input:  "Die Schweiz hat die höchste Einwanderungsrate in Europa."
Output: 🟡 Teilweise wahr (78% Konfidenz)
        Die Schweiz hat eine der höchsten Einwanderungsraten in Europa,
        liegt aber hinter Luxemburg und Malta...
```

## AI-Engineering-Patterns

Dieses Projekt demonstriert die folgenden AI-Engineering-Konzepte:

### 1. Agentic Workflow (ReAct Pattern)
Der Agent durchläuft autonom 4 Schritte – nicht ein einzelner API-Call, sondern eine orchestrierte Pipeline:

```
Claim Decomposer → Evidence Retriever → Source Evaluator → Verdict Synthesizer
```

### 2. RAG (Retrieval Augmented Generation)
Das LLM recherchiert aktiv im Web, statt sich auf Trainingsdaten zu verlassen. Jede Aussage wird mit aktuellen, externen Quellen abgeglichen.

### 3. Structured Output
Jeder LLM-Call liefert validiertes JSON (via Pydantic Models) – keine Freitext-Interpretation nötig. Das ist essenziell für produktionsreife AI-Anwendungen.

### 4. Prompt Engineering
Jeder Agent-Schritt hat einen spezialisierten Prompt mit Rollenanweisung, Beispielen und Constraints. Prompts sind dokumentiert und versioniert.

### 5. Evaluation
Ein Eval-Set mit 15 Behauptungen (bekannte Verdikt) ermöglicht systematisches Testen und Accuracy-Tracking.

## Architektur

```
User Input (Behauptung)
        │
        ▼
┌─────────────────────┐
│  Claim Decomposer   │  LLM-Call: Zerlegt in Teilaussagen
│  (Claude Sonnet)     │  + generiert Suchanfragen
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Evidence Retriever  │  Tavily API: Web Search + Extraktion
│  (Tavily Search)     │  Deduplizierung, Ranking
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Source Evaluator     │  LLM-Call: Relevanz, Glaubwürdigkeit,
│  (Claude Sonnet)     │  Verdikt pro Teilaussage
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Verdict Synthesizer │  LLM-Call: Gesamtbewertung mit
│  (Claude Sonnet)     │  Zusammenfassung und Quellen
└────────┬────────────┘
         │
         ▼
   Chat UI (Chainlit)
```

## Tech Stack

| Komponente | Technologie | Warum? |
|---|---|---|
| LLM | Claude API (Sonnet) | Starke Reasoning-Fähigkeiten, gutes Preis-Leistungs-Verhältnis |
| Orchestrierung | LangGraph | State-Management, Routing, Error Handling |
| Web Search | Tavily API | Speziell für AI-Agents gebaut, hohe Qualität |
| Structured Output | Pydantic v2 | Schema-Validierung, Typsicherheit |
| Frontend | Chainlit | Chat-UI mit Step-Visualisierung |

## Setup

### Voraussetzungen
- Python 3.11+
- Anthropic API Key ([console.anthropic.com](https://console.anthropic.com/))
- Tavily API Key ([tavily.com](https://tavily.com/) – kostenloser Tier)

### Installation

```bash
# Repository klonen
git clone https://github.com/thiev980/factagent.git
cd factagent

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # macOS/Linux
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# API-Keys konfigurieren
cp .env.example .env
# → .env editieren und Keys eintragen
```

### Starten

```bash
# Web App starten
chainlit run app.py

# → Öffnet http://localhost:8000
```

### Evaluation ausführen

```bash
# Alle 15 Test-Claims
python -m eval.run_eval

# Nur die ersten 5
python -m eval.run_eval --limit 5
```

## Projektstruktur

```
factagent/
├── app.py                  # Chainlit Web App (Frontend)
├── requirements.txt        # Python Dependencies
├── .env.example            # API-Key Template
├── agent/
│   ├── __init__.py
│   ├── models.py           # Pydantic Models (Structured Output)
│   ├── prompts.py          # Alle Prompts (Prompt Engineering)
│   ├── tools.py            # Tavily Search (RAG)
│   ├── nodes.py            # Agent-Schritte (Agentic Workflow)
│   └── graph.py            # LangGraph Workflow (Orchestrierung)
├── eval/
│   ├── eval_set.json       # Test-Behauptungen mit erwarteten Verdikten
│   └── run_eval.py         # Evaluations-Script
└── README.md
```

## Verdikt-Kategorien

| Verdikt | Emoji | Bedeutung |
|---|---|---|
| `true` | ✅ | Durch Evidenz bestätigt |
| `false` | ❌ | Durch Evidenz widerlegt |
| `partially_true` | 🟡 | Teilweise korrekt, mit Einschränkungen |
| `misleading` | ⚠️ | Technisch korrekt, aber irreführend |
| `unverifiable` | ❓ | Nicht genügend verlässliche Quellen |

## Mögliche Erweiterungen

- **Streaming**: LLM-Antworten live streamen (Chainlit unterstützt das)
- **Caching**: Suchergebnisse cachen, um Kosten zu senken
- **Mehrsprachigkeit**: Automatische Spracherkennung und -anpassung
- **Source Graph**: Visualisierung der Quellen-Netzwerke
- **Human-in-the-Loop**: Nutzer können Verdikt korrigieren → Feedback-Loop
- **Historische Claims**: Datenbank bereits geprüfter Behauptungen

## Kosten

Pro Faktencheck fallen ca. 3-4 Claude API Calls an (Sonnet):
- Claim Decomposition: ~500 Input + 500 Output Tokens
- Evidence Evaluation: ~2000 Input + 500 Output Tokens (pro Teilaussage)
- Verdict Synthesis: ~2000 Input + 500 Output Tokens
- Tavily: 1000 kostenlose Suchen/Monat

**Geschätzt: ~$0.01-0.03 pro Faktencheck** (mit Claude Sonnet)

## Über mich

Data Analyst mit Hintergrund in Soziologie – fokussiert auf die Schnittstelle von Gesellschaft, Medien und KI. Dieses Projekt entstand als Portfolio-Showcase für AI Engineering.

---

*Built with Claude API, LangGraph, Tavily, and Chainlit*
