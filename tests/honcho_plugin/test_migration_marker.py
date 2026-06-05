"""Regression tests for the one-time-per-peer memory-file migration guard.

Bug: under non-per-session strategies (and cron, whose session key is unique per
run) every new conversation/run created a fresh empty Honcho session, so the old
``not session.messages`` check re-migrated MEMORY.md/USER.md each time and flooded
the deriver with duplicate prior-memory observations.
"""
from plugins.memory.honcho.migration_marker import (
    MARKER_FILENAME,
    memory_files_fingerprint,
    migration_needed,
    record_migration,
)


def _seed_memories(tmp_path):
    mem = tmp_path / "memories"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("chanwoo prefers concise answers.\n", encoding="utf-8")
    (mem / "USER.md").write_text("chanwoo is the host operator.\n", encoding="utf-8")
    return mem


def test_migration_runs_once_per_peer_across_new_sessions(tmp_path):
    mem = _seed_memories(tmp_path)
    marker = mem / MARKER_FILENAME
    fp = memory_files_fingerprint(mem)
    assert fp is not None

    # First brand-new session for this peer -> migrate.
    assert migration_needed(marker, "chanwoo", fp) is True
    record_migration(marker, "chanwoo", fp)

    # A second brand-new session for the SAME peer must NOT re-migrate.
    # This is the flooding bug the marker fixes.
    assert migration_needed(marker, "chanwoo", fp) is False


def test_other_peer_still_migrates_once(tmp_path):
    mem = _seed_memories(tmp_path)
    marker = mem / MARKER_FILENAME
    fp = memory_files_fingerprint(mem)
    record_migration(marker, "chanwoo", fp)

    # A different peer (e.g. a cron peer) is independent and still migrates once.
    assert migration_needed(marker, "reviewer-cron", fp) is True
    record_migration(marker, "reviewer-cron", fp)
    assert migration_needed(marker, "reviewer-cron", fp) is False


def test_material_content_change_triggers_one_remigration(tmp_path):
    mem = _seed_memories(tmp_path)
    marker = mem / MARKER_FILENAME
    fp1 = memory_files_fingerprint(mem)
    record_migration(marker, "chanwoo", fp1)
    assert migration_needed(marker, "chanwoo", fp1) is False

    (mem / "MEMORY.md").write_text(
        "chanwoo prefers concise answers.\nchanwoo now also runs the dev fleet.\n",
        encoding="utf-8",
    )
    fp2 = memory_files_fingerprint(mem)
    assert fp2 != fp1
    assert migration_needed(marker, "chanwoo", fp2) is True


def test_no_files_means_no_migration_and_no_marker(tmp_path):
    mem = tmp_path / "memories"
    mem.mkdir()
    assert memory_files_fingerprint(mem) is None
    # Nothing to migrate -> never "needed", so no marker is ever written.
    assert migration_needed(mem / MARKER_FILENAME, "chanwoo", None) is False
