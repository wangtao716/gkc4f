"""Per-match JSONL records for reproducible classroom experiments.

The recorder stores only session metadata and completed game events.  It does
not inspect the user's account, machine, files, or input outside the game.
Filesystem failures deliberately propagate to the app so it can tell the user
that the current session was not saved.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


_RECORDED_EVENTS = frozenset((
    "decision", "observation", "attack", "dash", "phase",
    "player_hit", "boss_hit", "boss_block", "end",
))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionRecorder:
    """Write one exclusively created, UTF-8 JSONL file per match.

    Every accepted event is flushed before ``record`` returns.  Events keep
    their original fields and values; session metadata occupies separate
    records.  ``close`` is idempotent, while recording a game event after
    closing is a programming error and raises ``ValueError``.
    """

    def __init__(self, directory: Path | str, mode: str, seed: int,
                 input_source: str = "keyboard") -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        started = _utc_now()
        timestamp = started.strftime("%Y%m%dT%H%M%S.%fZ")
        # Exclusive creation prevents a collision from ever truncating a
        # previous experiment, including simultaneous processes or launches.
        for attempt in range(8):
            self.path = target / f"session_{timestamp}_{uuid4().hex[:8]}.jsonl"
            try:
                self._stream = self.path.open("x", encoding="utf-8", newline="\n")
                break
            except FileExistsError:
                if attempt == 7:
                    raise
        self._closed = False
        try:
            self._write({
                "type": "session_start",
                "schema_version": 3,
                "mode": mode,
                "seed": seed,
                "input_source": input_source,
                "started_at": started.isoformat(timespec="microseconds"),
            })
        except BaseException:
            self._closed = True
            self._stream.close()
            raise

    def _write(self, payload: dict) -> None:
        # Serialize first: invalid values must not leave a partial JSON line.
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()

    def record(self, event: dict) -> None:
        """Persist learning and combat outcomes; ignore non-game events."""
        if event.get("type") not in _RECORDED_EVENTS:
            return
        if self._closed:
            raise ValueError("Cannot record a game event after the session is closed")
        self._write(event)

    def close(self, summary: dict | None = None) -> None:
        """Flush and close once, preserving the caller's summary as-is."""
        if self._closed:
            return
        try:
            self._write({
                "type": "session_end",
                "ended_at": _utc_now().isoformat(timespec="microseconds"),
                "summary": summary,
            })
        finally:
            # Even after an I/O failure, release the handle and avoid a second
            # session_end on a later cleanup call.  OSError is not swallowed.
            self._closed = True
            self._stream.close()
