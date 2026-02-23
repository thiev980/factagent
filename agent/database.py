"""
FactAgent – Database (Historische Claims)
==========================================
Speichert abgeschlossene Faktenchecks in einer SQLite-Datenbank.
Ermöglicht Similarity-Search über FTS5 (Full-Text Search), 
das direkt in SQLite eingebaut ist – keine Extra-Dependencies.

AI-Engineering-Pattern: Caching / Knowledge Base
- Vermeidet redundante (teure) API-Calls
- Baut über Zeit eine Wissensbasis auf
- FTS5 ermöglicht unscharfe Suche ("Schweiz Einwanderung" findet 
  auch "Einwanderungsrate der Schweiz")

Datenbank-Datei: factagent.db (wird automatisch erstellt)
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.models import FactCheckResult, Verdict

logger = logging.getLogger(__name__)

# Datenbank-Pfad (im Projektverzeichnis)
DB_PATH = Path(__file__).parent.parent / "factagent.db"


def get_connection() -> sqlite3.Connection:
    """Erstellt eine SQLite-Verbindung mit WAL-Modus für bessere Concurrency."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Dict-ähnlicher Zugriff auf Spalten
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """
    Erstellt die Tabellen, falls sie nicht existieren.
    Wird beim App-Start aufgerufen.
    """
    conn = get_connection()
    try:
        # Haupttabelle: Faktenchecks
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                claim_normalized TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                result_json TEXT NOT NULL,
                human_reviewed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                check_duration_seconds REAL
            )
        """)

        # FTS5 Virtual Table für Full-Text-Suche
        # Indexiert die normalisierte Behauptung + das Summary
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fact_checks_fts 
            USING fts5(
                claim_normalized, 
                summary,
                content='fact_checks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            )
        """)

        # Trigger: FTS-Index automatisch aktualisieren
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fact_checks_ai AFTER INSERT ON fact_checks BEGIN
                INSERT INTO fact_checks_fts(rowid, claim_normalized, summary)
                VALUES (new.id, new.claim_normalized, 
                        json_extract(new.result_json, '$.summary'));
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fact_checks_ad AFTER DELETE ON fact_checks BEGIN
                INSERT INTO fact_checks_fts(fact_checks_fts, rowid, claim_normalized, summary)
                VALUES ('delete', old.id, old.claim_normalized,
                        json_extract(old.result_json, '$.summary'));
            END
        """)

        conn.commit()
        logger.info(f"✅ Datenbank initialisiert: {DB_PATH}")

    finally:
        conn.close()


def _normalize_claim(claim: str) -> str:
    """
    Normalisiert eine Behauptung für den Vergleich.
    - Kleinbuchstaben
    - Whitespace normalisieren
    - Satzzeichen am Ende entfernen
    """
    import re
    text = claim.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.!?;:')
    return text


def store_fact_check(
    claim: str,
    result: FactCheckResult,
    human_reviewed: bool = False,
    duration_seconds: float | None = None,
) -> int:
    """
    Speichert einen abgeschlossenen Faktencheck in der Datenbank.
    
    Args:
        claim: Die ursprüngliche Behauptung
        result: Das FactCheckResult-Objekt
        human_reviewed: Wurde der Check vom User überprüft?
        duration_seconds: Dauer des Checks in Sekunden
    
    Returns:
        Die ID des neuen Eintrags
    """
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO fact_checks 
            (claim, claim_normalized, verdict, confidence, result_json, 
             human_reviewed, created_at, check_duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            claim,
            _normalize_claim(claim),
            result.overall_verdict.value,
            result.confidence,
            result.model_dump_json(),
            1 if human_reviewed else 0,
            datetime.now(timezone.utc).isoformat(),
            duration_seconds,
        ))
        conn.commit()
        row_id = cursor.lastrowid
        logger.info(f"💾 Faktencheck gespeichert (ID: {row_id})")
        return row_id
    finally:
        conn.close()


def find_similar_claims(
    claim: str,
    limit: int = 5,
    min_rank: float = -10.0,
) -> list[dict]:
    """
    Sucht nach ähnlichen, bereits geprüften Behauptungen.
    
    Nutzt SQLite FTS5 mit BM25-Ranking.
    
    Args:
        claim: Die neue Behauptung
        limit: Max. Anzahl Ergebnisse
        min_rank: Min. BM25-Score (negativer = besser bei FTS5)
    
    Returns:
        Liste von Dicts mit id, claim, verdict, confidence, 
        created_at, rank, result (FactCheckResult)
    """
    conn = get_connection()
    try:
        normalized = _normalize_claim(claim)

        # FTS5-Suche mit BM25-Ranking
        rows = conn.execute("""
            SELECT 
                fc.id,
                fc.claim,
                fc.verdict,
                fc.confidence,
                fc.human_reviewed,
                fc.created_at,
                fc.result_json,
                rank
            FROM fact_checks_fts 
            JOIN fact_checks fc ON fact_checks_fts.rowid = fc.id
            WHERE fact_checks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (normalized, limit)).fetchall()

        results = []
        for row in rows:
            if row["rank"] <= min_rank:
                try:
                    result = FactCheckResult.model_validate_json(row["result_json"])
                except Exception:
                    result = None

                results.append({
                    "id": row["id"],
                    "claim": row["claim"],
                    "verdict": row["verdict"],
                    "confidence": row["confidence"],
                    "human_reviewed": bool(row["human_reviewed"]),
                    "created_at": row["created_at"],
                    "rank": row["rank"],
                    "result": result,
                })

        logger.info(f"🔎 {len(results)} ähnliche Claims gefunden für: {claim[:50]}...")
        return results

    except Exception as e:
        # FTS-Suche kann bei bestimmten Suchbegriffen fehlschlagen
        logger.warning(f"⚠️ FTS-Suche fehlgeschlagen: {e}")
        return []
    finally:
        conn.close()


def find_exact_claim(claim: str) -> Optional[dict]:
    """
    Sucht nach einem exakten Match (normalisiert).
    
    Returns:
        Dict mit Ergebnis oder None
    """
    conn = get_connection()
    try:
        normalized = _normalize_claim(claim)
        row = conn.execute("""
            SELECT id, claim, verdict, confidence, human_reviewed, 
                   created_at, result_json
            FROM fact_checks
            WHERE claim_normalized = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (normalized,)).fetchone()

        if row:
            try:
                result = FactCheckResult.model_validate_json(row["result_json"])
            except Exception:
                result = None

            return {
                "id": row["id"],
                "claim": row["claim"],
                "verdict": row["verdict"],
                "confidence": row["confidence"],
                "human_reviewed": bool(row["human_reviewed"]),
                "created_at": row["created_at"],
                "result": result,
            }
        return None
    finally:
        conn.close()


def get_recent_checks(limit: int = 10) -> list[dict]:
    """Gibt die letzten N Faktenchecks zurück."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT id, claim, verdict, confidence, human_reviewed, 
                   created_at, check_duration_seconds
            FROM fact_checks
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_stats() -> dict:
    """Gibt Statistiken über die Datenbank zurück."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM fact_checks").fetchone()[0]
        by_verdict = conn.execute("""
            SELECT verdict, COUNT(*) as count 
            FROM fact_checks 
            GROUP BY verdict
        """).fetchall()
        reviewed = conn.execute(
            "SELECT COUNT(*) FROM fact_checks WHERE human_reviewed = 1"
        ).fetchone()[0]

        return {
            "total_checks": total,
            "human_reviewed": reviewed,
            "by_verdict": {row["verdict"]: row["count"] for row in by_verdict},
        }
    finally:
        conn.close()
