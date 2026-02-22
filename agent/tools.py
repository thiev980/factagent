"""
FactAgent – Search Tools (Evidence Retrieval)
=============================================
Hier passiert die RAG-Komponente: Der Agent holt sich aktiv
externes Wissen, anstatt sich auf das LLM-Training zu verlassen.

AI-Engineering-Pattern: RAG (Retrieval Augmented Generation)
- LLM generiert Suchanfragen
- External Tool führt die Suche durch
- Ergebnisse fliessen zurück in den LLM-Kontext
"""

import os
import logging
from tavily import TavilyClient

logger = logging.getLogger(__name__)


def get_tavily_client() -> TavilyClient:
    """Erstellt einen Tavily-Client mit API-Key aus der Umgebung."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY nicht gesetzt. "
            "Bitte in .env eintragen (kostenlos auf https://tavily.com/)."
        )
    return TavilyClient(api_key=api_key)


def search_evidence(queries: list[str], max_results_per_query: int = 5) -> list[dict]:
    """
    Führt mehrere Suchanfragen aus und gibt deduplizierte Ergebnisse zurück.
    
    Args:
        queries: Liste von Suchanfragen (z.B. 2-3 pro Teilaussage)
        max_results_per_query: Max. Ergebnisse pro Anfrage
    
    Returns:
        Liste von Suchergebnis-Dicts mit url, title, content
    """
    client = get_tavily_client()
    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            logger.info(f"🔍 Suche: {query}")
            response = client.search(
                query=query,
                max_results=max_results_per_query,
                search_depth="advanced",       # Tiefere Suche für bessere Ergebnisse
                include_raw_content=False,      # Spart Tokens
                include_answer=False,           # Wir wollen die Rohdaten
            )

            for result in response.get("results", []):
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "content": result.get("content", "")[:500],  # Auf 500 Zeichen begrenzen
                        "score": result.get("score", 0.0),
                        "query": query,
                    })

        except Exception as e:
            logger.warning(f"⚠️ Suche fehlgeschlagen für '{query}': {e}")
            continue

    # Sortiere nach Tavily-Score (höchste Relevanz zuerst)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    logger.info(f"📊 {len(all_results)} eindeutige Quellen gefunden")
    return all_results


def format_search_results_for_prompt(results: list[dict]) -> str:
    """
    Formatiert Suchergebnisse als lesbaren Text für den LLM-Prompt.
    
    Wichtig: Wir geben dem LLM die Rohdaten, damit es selbst
    bewerten kann, was relevant und glaubwürdig ist.
    """
    if not results:
        return "Keine Suchergebnisse gefunden."

    formatted_parts = []
    for i, result in enumerate(results, 1):
        formatted_parts.append(
            f"### Quelle {i}\n"
            f"- **URL**: {result['url']}\n"
            f"- **Titel**: {result['title']}\n"
            f"- **Inhalt**: {result['content']}\n"
        )

    return "\n".join(formatted_parts)
