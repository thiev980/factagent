"""
FactAgent – Pydantic Models (Structured Output)
================================================
Diese Models definieren die Datenstrukturen, die der Agent zwischen
seinen Schritten weitergibt. Sie sind gleichzeitig das Schema für
die strukturierte Ausgabe der LLM-Calls.

AI-Engineering-Pattern: Structured Output
- LLMs liefern JSON statt Freitext
- Pydantic validiert die Ausgabe automatisch
- Typsicherheit durch die gesamte Pipeline
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums für klare Kategorien
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Typ der Behauptung – bestimmt, wie der Agent vorgeht."""
    FACTUAL = "factual"           # Überprüfbare Faktenaussage
    OPINION = "opinion"           # Meinungsäusserung (nicht checkbar)
    MIXED = "mixed"               # Enthält beides
    PREDICTION = "prediction"     # Zukunftsaussage (schwer checkbar)


class Verdict(str, Enum):
    """Bewertung einer (Teil-)Aussage."""
    TRUE = "true"
    FALSE = "false"
    PARTIALLY_TRUE = "partially_true"
    MISLEADING = "misleading"       # Technisch korrekt, aber irreführend
    UNVERIFIABLE = "unverifiable"   # Kann nicht überprüft werden


class Credibility(str, Enum):
    """Einschätzung der Quellenglaubwürdigkeit."""
    HIGH = "high"       # Offizielle Statistikämter, peer-reviewed, etc.
    MEDIUM = "medium"   # Etablierte Medien, Wikipedia, etc.
    LOW = "low"         # Blogs, Social Media, unbekannte Quellen


# ---------------------------------------------------------------------------
# Claim Decomposition (Schritt 1)
# ---------------------------------------------------------------------------

class SubClaim(BaseModel):
    """Eine einzelne, überprüfbare Teilaussage."""
    claim: str = Field(description="Die Teilaussage in einem klaren Satz")
    search_queries: list[str] = Field(
        description="2-3 gezielte Suchanfragen, um diese Teilaussage zu überprüfen",
        min_length=1,
        max_length=3,
    )


class ClaimDecomposition(BaseModel):
    """Ergebnis der Zerlegung einer Behauptung in überprüfbare Teile."""
    original_claim: str = Field(description="Die ursprüngliche Behauptung")
    claim_type: ClaimType = Field(description="Typ der Behauptung")
    language: str = Field(
        description="Erkannte Sprache der Behauptung (z.B. 'de', 'en', 'fr')",
        default="de",
    )
    sub_claims: list[SubClaim] = Field(
        description="Liste der überprüfbaren Teilaussagen",
        min_length=1,
        max_length=5,
    )


# ---------------------------------------------------------------------------
# Evidence (Schritt 2 + 3)
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """Eine einzelne Quelle mit Bewertung."""
    url: str = Field(description="URL der Quelle")
    title: str = Field(description="Titel der Seite/des Artikels")
    snippet: str = Field(description="Relevanter Textausschnitt (max 200 Zeichen)")
    relevance_score: float = Field(
        description="Wie relevant ist diese Quelle für die Teilaussage (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    credibility: Credibility = Field(
        description="Einschätzung der Quellenglaubwürdigkeit"
    )


class SubClaimVerdict(BaseModel):
    """Bewertung einer einzelnen Teilaussage mit Evidenz."""
    claim: str = Field(description="Die überprüfte Teilaussage")
    verdict: Verdict = Field(description="Bewertung")
    confidence: float = Field(
        description="Konfidenz der Bewertung (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    evidence: list[Source] = Field(
        description="Quellen, die zur Bewertung herangezogen wurden"
    )
    reasoning: str = Field(
        description="Kurze Begründung der Bewertung (2-3 Sätze)"
    )


# ---------------------------------------------------------------------------
# Final Result (Schritt 4)
# ---------------------------------------------------------------------------

class FactCheckResult(BaseModel):
    """Das Gesamtergebnis des Faktenchecks."""
    original_claim: str = Field(description="Die ursprüngliche Behauptung")
    overall_verdict: Verdict = Field(description="Gesamtbewertung")
    confidence: float = Field(
        description="Gesamtkonfidenz (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    sub_verdicts: list[SubClaimVerdict] = Field(
        description="Bewertungen der einzelnen Teilaussagen"
    )
    summary: str = Field(
        description="Zusammenfassende Erklärung in 3-5 Sätzen, "
        "in der Sprache der ursprünglichen Behauptung"
    )
    key_sources: list[Source] = Field(
        description="Die wichtigsten 3-5 Quellen insgesamt"
    )


# ---------------------------------------------------------------------------
# Human-in-the-Loop Feedback
# ---------------------------------------------------------------------------

class SubClaimFeedback(BaseModel):
    """Feedback des Users zu einer einzelnen Teilaussage."""
    claim: str = Field(description="Die betroffene Teilaussage")
    corrected_verdict: Optional[Verdict] = Field(
        default=None,
        description="Vom User korrigiertes Verdikt (None = AI-Verdikt akzeptiert)",
    )
    user_comment: Optional[str] = Field(
        default=None,
        description="Optionaler Kommentar/Kontext des Users",
    )


class HumanFeedback(BaseModel):
    """
    Gesammeltes Feedback des Users nach der Evidenzbewertung.
    
    AI-Engineering-Pattern: Human-in-the-Loop
    - Der Mensch kann AI-Entscheidungen korrigieren
    - Feedback fliesst in die finale Synthese ein
    - Transparenz: AI-Verdikt vs. User-Korrektur wird dokumentiert
    """
    reviewed: bool = Field(
        default=False,
        description="Hat der User die Bewertungen aktiv überprüft?",
    )
    sub_claim_feedback: list[SubClaimFeedback] = Field(
        default_factory=list,
        description="Feedback pro Teilaussage",
    )
    general_comment: Optional[str] = Field(
        default=None,
        description="Allgemeiner Kommentar des Users zur Behauptung",
    )


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """
    Der Zustand, der durch den LangGraph-Workflow fliesst.
    Jeder Node liest und schreibt Teile dieses States.
    """
    claim: str = ""
    decomposition: Optional[ClaimDecomposition] = None
    search_results: dict[str, list[dict]] = Field(default_factory=dict)
    sub_verdicts: list[SubClaimVerdict] = Field(default_factory=list)
    human_feedback: Optional[HumanFeedback] = None
    final_result: Optional[FactCheckResult] = None
    error: Optional[str] = None
