"""One-time-per-peer guard for pre-Honcho memory-file migration.

Replaces the previous ``session_strategy == "per-session"`` heuristic, which
only suppressed re-migration for per-run sessions. Per-conversation / per-repo /
global strategies — and cron runs, whose session key is unique per run —
defeated it: a brand-new empty Honcho session was created every conversation/run,
so ``not session.messages`` stayed true and MEMORY.md/USER.md were re-uploaded
each time. The deriver then thrashed on near-duplicate prior-memory observations.

This marker makes migration idempotent per ``(user peer, memory-file content)``
regardless of session strategy. The fingerprint lets a materially changed
MEMORY.md/USER.md migrate exactly once more.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MARKER_FILENAME = ".honcho-migrated.json"

# Files migrate_memory_files() uploads, in a stable order.
_MIGRATED_FILES = ("MEMORY.md", "USER.md")


def memory_files_fingerprint(memory_dir: Path) -> str | None:
    """SHA-256 over the migratable memory files' contents.

    Returns ``None`` when none of the files exist with non-empty content (so
    there is nothing to migrate and no marker should be written).
    """
    h = hashlib.sha256()
    found = False
    for name in _MIGRATED_FILES:
        path = memory_dir / name
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        found = True
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(content.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest() if found else None


def _load(marker_path: Path) -> dict[str, str]:
    if marker_path.exists():
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def migration_needed(marker_path: Path, peer_id: str, fingerprint: str | None) -> bool:
    """True iff this peer has not yet migrated this exact memory-file content."""
    if not fingerprint:
        return False
    return _load(marker_path).get(peer_id) != fingerprint


def record_migration(marker_path: Path, peer_id: str, fingerprint: str) -> None:
    """Persist that ``peer_id`` migrated content ``fingerprint``."""
    migrated = _load(marker_path)
    migrated[peer_id] = fingerprint
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(migrated), encoding="utf-8")
