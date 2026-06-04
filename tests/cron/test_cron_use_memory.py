"""Tests for cron jobs opting into Honcho external memory (``use_memory``).

Cron jobs run with ``skip_memory=True`` by default — a deliberate guard so a
cron persona/system-prompt never pollutes the human's Honcho user
representation. This feature adds an *opt-in* per-job ``use_memory`` flag.
When set, the cron run:

* gets memory recall + write-back (``skip_memory=False``), and
* writes are scoped to a dedicated **cron peer** (``<identity>-cron``) so the
  human user peer is never observed — honoring the original guard's intent.

Covered as Given/When/Then scenarios:

* Data layer — ``create_job(use_memory=...)`` stores/normalizes the flag,
  defaults to False, back-compat for legacy jobs, ``update_job`` toggles it.
* Scheduler — ``run_job`` threads ``skip_memory``/``cron_memory`` into
  ``AIAgent`` based on the flag.
* Honcho provider — the cron guard yields to ``cron_memory=True``; the cron
  peer name is derived from the agent identity.
"""

from __future__ import annotations

import sys
import types

import pytest


# Stub optional heavy deps pulled in transitively (mirrors sibling cron tests).
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so jobs/scripts don't leak between tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# Data layer: create_job / update_job / normalization
# ---------------------------------------------------------------------------


def test_create_job_defaults_use_memory_false(hermes_env):
    """Given no use_memory arg, When a job is created, Then it defaults False."""
    from cron.jobs import create_job

    job = create_job(prompt="ping", schedule="every 5m")
    assert job["use_memory"] is False


def test_create_job_stores_use_memory_true(hermes_env):
    """Given use_memory=True, When a job is created, Then the flag persists."""
    from cron.jobs import create_job, load_jobs

    job = create_job(prompt="ping", schedule="every 5m", use_memory=True)
    assert job["use_memory"] is True

    # Survives a round-trip through jobs.json storage.
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert stored["use_memory"] is True


def test_normalize_legacy_job_without_use_memory(hermes_env):
    """Given a legacy job lacking the field, When read, Then it normalizes False."""
    from cron.jobs import _normalize_job_record

    legacy = {"id": "abc", "prompt": "hi", "schedule": {"kind": "cron", "expr": "0 5 * * *"}}
    normalized = _normalize_job_record(legacy)
    assert normalized["use_memory"] is False


def test_update_job_toggles_use_memory(hermes_env):
    """Given an existing job, When updated, Then use_memory flips and persists."""
    from cron.jobs import create_job, update_job

    job = create_job(prompt="ping", schedule="every 5m")
    assert job["use_memory"] is False

    updated = update_job(job["id"], {"use_memory": True})
    assert updated["use_memory"] is True

    re_disabled = update_job(job["id"], {"use_memory": False})
    assert re_disabled["use_memory"] is False


# ---------------------------------------------------------------------------
# Scheduler: run_job threads skip_memory / cron_memory into AIAgent
# ---------------------------------------------------------------------------


class _RecordingAgent:
    """Captures the kwargs run_job constructs AIAgent with, then no-ops."""

    last_init: dict = {}

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        return {"final_response": "ok", "completed": True, "failed": False}


def _run_job_capturing(monkeypatch, job):
    import run_agent
    import cron.scheduler as cron_scheduler

    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
            "provider": "openai",
            "api_mode": "responses",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )

    _RecordingAgent.last_init = {}
    cron_scheduler.run_job(job)
    return _RecordingAgent.last_init


def test_run_job_default_skips_memory(hermes_env, monkeypatch):
    """Given a normal job, When run, Then skip_memory=True and cron_memory=False."""
    init = _run_job_capturing(
        monkeypatch, {"id": "j1", "name": "n", "prompt": "ping", "model": "gpt-5.5"}
    )
    assert init.get("skip_memory") is True
    assert init.get("cron_memory") is False


def test_run_job_use_memory_enables_honcho(hermes_env, monkeypatch):
    """Given use_memory=True, When run, Then skip_memory=False and cron_memory=True."""
    init = _run_job_capturing(
        monkeypatch,
        {"id": "j2", "name": "n", "prompt": "ping", "model": "gpt-5.5", "use_memory": True},
    )
    assert init.get("skip_memory") is False
    assert init.get("cron_memory") is True


# ---------------------------------------------------------------------------
# Honcho provider: cron guard yields to cron_memory; cron peer derivation
# ---------------------------------------------------------------------------


def test_honcho_cron_guard_skips_without_cron_memory():
    """Given platform=cron without cron_memory, Then the provider stays inert."""
    from plugins.memory.honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    provider.initialize("sess-1", platform="cron")
    assert provider._cron_skipped is True


def test_honcho_cron_guard_yields_to_cron_memory():
    """Given cron_memory=True, Then the cron guard does NOT mark the run skipped."""
    from plugins.memory.honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    # Honcho is not configured in the test env, so initialize() falls through to
    # the config check and returns WITHOUT taking the cron-skip branch.
    provider.initialize("sess-1", platform="cron", cron_memory=True)
    assert provider._cron_skipped is False


def test_cron_peer_name_derives_from_identity():
    """The cron write peer is a dedicated, sanitized '<identity>-cron' peer."""
    from plugins.memory.honcho import _cron_peer_name

    assert _cron_peer_name("backend-senior") == "backend-senior-cron"
    # Honcho peer IDs must match ^[a-zA-Z0-9_-]+$ — dots/spaces get sanitized.
    assert _cron_peer_name("host.admin agent") == "host-admin-agent-cron"
    # Missing identity falls back to a stable default, never an empty peer.
    assert _cron_peer_name("") == "cron"


# ---------------------------------------------------------------------------
# Model-facing tool: cronjob(action=create/update, use_memory=...)
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_with_use_memory(hermes_env, monkeypatch):
    """Given cronjob(create, use_memory=True), Then the stored job has the flag."""
    import json
    from tools.cronjob_tools import cronjob

    res = json.loads(
        cronjob(action="create", prompt="digest", schedule="every 1h", use_memory=True)
    )
    assert res["success"] is True

    from cron.jobs import load_jobs
    stored = next(j for j in load_jobs() if j["id"] == res["job_id"])
    assert stored["use_memory"] is True


def test_cronjob_tool_update_toggles_use_memory(hermes_env, monkeypatch):
    """Given an existing job, When cronjob(update, use_memory=True), Then it flips."""
    import json
    from tools.cronjob_tools import cronjob
    from cron.jobs import load_jobs

    created = json.loads(cronjob(action="create", prompt="ping", schedule="every 1h"))
    job_id = created["job_id"]
    assert next(j for j in load_jobs() if j["id"] == job_id)["use_memory"] is False

    cronjob(action="update", job_id=job_id, use_memory=True)
    assert next(j for j in load_jobs() if j["id"] == job_id)["use_memory"] is True
