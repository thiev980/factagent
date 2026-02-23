"""
FactAgent – Rate Limiter
=========================
Schützt die API-Keys vor Missbrauch, wenn die App öffentlich
auf Hugging Face Spaces läuft.

Einfaches In-Memory Rate Limiting pro Session:
- Max. N Checks pro Session
- Min. Wartezeit zwischen Checks
- Kein externer Redis/DB nötig

Für eine Demo-App völlig ausreichend.
"""

import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Konfiguration
MAX_CHECKS_PER_SESSION = 10       # Max. Checks pro Chat-Session
MIN_SECONDS_BETWEEN_CHECKS = 30   # Min. Sekunden zwischen zwei Checks
MAX_CLAIM_LENGTH = 500            # Max. Zeichenlänge einer Behauptung


class RateLimiter:
    """In-Memory Rate Limiter pro Session-ID."""

    def __init__(self):
        self._session_counts: dict[str, int] = defaultdict(int)
        self._session_last_check: dict[str, float] = defaultdict(float)

    def check(self, session_id: str) -> tuple[bool, str]:
        """
        Prüft, ob ein Check erlaubt ist.
        
        Returns:
            (allowed: bool, message: str)
        """
        now = time.time()

        # Session-Limit
        if self._session_counts[session_id] >= MAX_CHECKS_PER_SESSION:
            return False, (
                f"⚠️ Du hast das Limit von {MAX_CHECKS_PER_SESSION} Checks "
                f"pro Session erreicht. Starte eine neue Session, um "
                f"weitere Behauptungen zu prüfen."
            )

        # Cooldown
        elapsed = now - self._session_last_check[session_id]
        if self._session_last_check[session_id] > 0 and elapsed < MIN_SECONDS_BETWEEN_CHECKS:
            wait = int(MIN_SECONDS_BETWEEN_CHECKS - elapsed)
            return False, (
                f"⏳ Bitte warte noch {wait} Sekunden vor dem nächsten Check."
            )

        return True, ""

    def record(self, session_id: str):
        """Registriert einen durchgeführten Check."""
        self._session_counts[session_id] += 1
        self._session_last_check[session_id] = time.time()
        logger.info(
            f"📊 Rate Limit: Session {session_id[:8]}... "
            f"hat {self._session_counts[session_id]}/{MAX_CHECKS_PER_SESSION} Checks"
        )

    def reset(self, session_id: str):
        """Setzt den Counter für eine Session zurück."""
        self._session_counts.pop(session_id, None)
        self._session_last_check.pop(session_id, None)


def validate_claim(claim: str) -> tuple[bool, str]:
    """
    Validiert die Behauptung vor dem Check.
    
    Returns:
        (valid: bool, message: str)
    """
    if len(claim) > MAX_CLAIM_LENGTH:
        return False, (
            f"⚠️ Behauptung ist zu lang ({len(claim)} Zeichen). "
            f"Maximum: {MAX_CLAIM_LENGTH} Zeichen."
        )

    if len(claim) < 10:
        return False, "⚠️ Behauptung ist zu kurz. Bitte formuliere einen vollständigen Satz."

    return True, ""


# Globale Instanz
rate_limiter = RateLimiter()
