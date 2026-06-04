"""Tests for per-role memory on delegated sub-agents (`agent_name`).

Delegated sub-agents run `skip_memory=True` by default — they are ephemeral
and would otherwise write into the parent/user peer. But when a lead agent
assigns a *named* role to a delegation (`agent_name="collector"`), that
sub-agent should accumulate its OWN experience, scoped to a stable per-role
peer `<parent_identity>-<agent_name>` (never the human user peer).

Scenarios:
- Given a delegation WITHOUT agent_name, the child stays memory-isolated
  (skip_memory=True) — unchanged legacy behavior.
- Given a delegation WITH agent_name, the child runs memory-on
  (skip_memory=False, cron_memory=True) scoped to `<parent>-<agent_name>`.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cron"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


class _RecordingAgent:
    """Captures the kwargs _build_child_agent constructs AIAgent with."""

    last_init: dict = {}

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)

    # _build_child_agent sets several attrs on the constructed child.
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


def _build(monkeypatch, **extra):
    import run_agent
    from tools import delegate_tool

    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "framework-scout"
    )
    _RecordingAgent.last_init = {}
    delegate_tool._build_child_agent(
        0, "collect upstream issues", None, None, None, 8, 1, _make_mock_parent(), **extra
    )
    return _RecordingAgent.last_init


def test_delegation_without_agent_name_is_memory_isolated(monkeypatch):
    init = _build(monkeypatch)
    assert init.get("skip_memory") is True
    assert init.get("cron_memory") in (False, None)


def test_delegation_with_agent_name_enables_scoped_memory(monkeypatch):
    init = _build(monkeypatch, agent_name="collector")
    assert init.get("skip_memory") is False
    assert init.get("cron_memory") is True
    # The child writes to its own stable role peer, derived from the lead.
    assert init.get("memory_identity") == "framework-scout-collector"
