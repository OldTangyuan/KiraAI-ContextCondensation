"""Unit tests for the context_condensation plugin.

Run EITHER way, from the KiraAI repo root:

    python data/plugins/context_condensation/tests/self_test.py   # standalone
    python -m pytest data/plugins/context_condensation/tests/ -v # pytest

The tests exercise the pure-Python logic (message filtering, cache
persistence/robustness, summary length cap) with a FakeLLM. They import the
plugin modules directly under the ``context_condensation`` package namespace
without booting the full application.

pytest is used when available (the standalone runner falls back to plain
assertions so the suite also runs in environments without it).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path

try:  # pytest optional for standalone runs
    import pytest
except ImportError:
    pytest = None


# ---------------------------------------------------------------------------
# Stub the framework modules the plugin imports, so importing the plugin
# modules does not require a full KiraAI boot (same pattern as the
# hippocampus plugin tests).
# ---------------------------------------------------------------------------


def _install_stubs():
    if "core.provider" not in sys.modules:
        provider_stub = types.ModuleType("core.provider")

        @dataclass
        class _LLMRequest:
            messages: list = field(default_factory=list)
            user_prompt: list = field(default_factory=list)

        class _LLMModelClient:
            pass

        class _LLMResponse:
            def __init__(self, text_response="", reasoning_content=""):
                self.text_response = text_response
                self.reasoning_content = reasoning_content

        provider_stub.LLMRequest = _LLMRequest
        provider_stub.LLMModelClient = _LLMModelClient
        provider_stub.LLMResponse = _LLMResponse
        sys.modules["core.provider"] = provider_stub

    if "core.prompt_manager" not in sys.modules:
        pm_stub = types.ModuleType("core.prompt_manager")

        class _Prompt:
            def __init__(self, content="", name=None, source=None, persist=True, **kw):
                self.content = content
                self.name = name
                self.source = source
                self.persist = persist
                self.kwargs = kw

            def to_string(self):
                return self.content

        pm_stub.Prompt = _Prompt
        sys.modules["core.prompt_manager"] = pm_stub

    if "core.agent.message" not in sys.modules:
        am_stub = types.ModuleType("core.agent.message")

        class _OpenAIMessage:
            def __init__(self, role="user", content=None, **kw):
                self.role = role
                self.content = content
                for k, v in kw.items():
                    setattr(self, k, v)

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        am_stub.OpenAIMessage = _OpenAIMessage
        sys.modules["core.agent.message"] = am_stub

    if "core.plugin" not in sys.modules:
        plugin_stub = types.ModuleType("core.plugin")
        import logging

        plugin_stub.BasePlugin = object
        plugin_stub.get_logger = lambda name, color: logging.getLogger(name)
        plugin_stub.on = types.SimpleNamespace(
            llm_request=lambda **kw: (lambda fn: fn),
            step_result=lambda **kw: (lambda fn: fn),
        )
        plugin_stub.Priority = types.SimpleNamespace(LOW=0, HIGH=100)
        sys.modules["core.plugin"] = plugin_stub

    if "core.utils.path_utils" not in sys.modules:
        pu_stub = types.ModuleType("core.utils.path_utils")
        pu_stub.get_data_path = lambda: Path(tempfile.gettempdir())
        sys.modules["core.utils.path_utils"] = pu_stub

    # plugin package namespace so `from context_condensation import ...` works
    if "context_condensation" not in sys.modules:
        pkg = types.ModuleType("context_condensation")
        pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["context_condensation"] = pkg


_install_stubs()

from context_condensation.context_cache import (  # noqa: E402
    CachedRound,
    CompressionGroup,
    ContextCache,
    is_prompt,
)
from context_condensation.compressor import CompressionEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, text):
        self.text_response = text
        self.reasoning_content = ""


class FakeLLM:
    """Scripted LLM: returns queued responses, then a fallback."""

    def __init__(self, scripted=None, fallback="FAKE"):
        self.scripted = list(scripted or [])
        self.idx = 0
        self.fallback = fallback
        self.calls = []

    async def chat(self, request):
        text = self.scripted[self.idx] if self.idx < len(self.scripted) else self.fallback
        self.idx += 1
        self.calls.append(text[:40])
        return _FakeResp(text)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _new_cache(tmp_path, anchor=2):
    return ContextCache(Path(tmp_path), anchor_size=anchor)


def _sync_rounds(cache, n, prefix="msg"):
    """Sync n user/assistant rounds into the cache."""
    for i in range(n):
        cache.sync(
            [
                {"role": "user", "content": f"{prefix}{i} question"},
                {"role": "assistant", "content": f"{prefix}{i} answer"},
            ]
        )


# ---------------------------------------------------------------------------
# is_prompt / message filtering
# ---------------------------------------------------------------------------


def test_is_prompt_detects_framework_prompts():
    assert not is_prompt({"role": "user", "content": "hi"})
    assert not is_prompt({"role": "assistant", "content": "yo"})

    from core.prompt_manager import Prompt

    assert is_prompt(Prompt("<system_reminder>", name="dynamic_context_start", persist=False))
    assert is_prompt(Prompt("relocated", name="time", persist=False))


def test_parse_into_rounds_skips_prompts():
    from core.prompt_manager import Prompt

    rounds = ContextCache._parse_into_rounds(
        [
            {"role": "user", "content": "hello"},
            Prompt("<system_reminder>", persist=False),
            {"role": "assistant", "content": "hi back"},
        ]
    )
    assert len(rounds) == 1
    roles = [m["role"] for m in rounds[0].messages]
    assert roles == ["user", "assistant"]
    texts = [str(m.get("content", "")) for m in rounds[0].messages]
    assert "<system_reminder>" not in "\n".join(texts)


def test_sync_ignores_prompt_objects(tmp_path):
    from core.prompt_manager import Prompt

    cache = _new_cache(tmp_path)
    added, present = cache.sync(
        [
            {"role": "user", "content": "real user"},
            {"role": "assistant", "content": "real asst"},
            Prompt("scaffold", persist=False),
        ]
    )
    assert len(added) == 1
    assert len(present) == 1
    all_text = "\n".join(
        str(m.get("content", "")) for r in present for m in r.messages
    )
    assert "scaffold" not in all_text


# ---------------------------------------------------------------------------
# Persistence / robustness
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    cache = _new_cache(tmp_path)
    _sync_rounds(cache, 3, prefix="rt")
    cache.save("sid_x")

    cache2 = _new_cache(tmp_path)
    assert cache2.load("sid_x") is True
    assert len(cache2.rounds) == 3
    assert cache2._next_round_index == cache._next_round_index
    assert len(cache2._fingerprint_index) == 3


def test_load_corrupt_file_starts_fresh(tmp_path):
    cache_dir = Path(tmp_path) / "caches"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "bad.json").write_text("{ not valid json", encoding="utf-8")

    cache = _new_cache(tmp_path)
    assert cache.load("bad") is False
    assert len(cache.rounds) == 0


def test_load_empty_file_starts_fresh(tmp_path):
    cache_dir = Path(tmp_path) / "caches"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "empty.json").write_text("", encoding="utf-8")

    cache = _new_cache(tmp_path)
    assert cache.load("empty") is False
    assert len(cache.rounds) == 0


def test_save_with_unserializable_message_does_not_crash(tmp_path):
    cache = _new_cache(tmp_path)

    class Weird:
        def to_dict(self):
            raise TypeError("boom")

    cache._rounds.append(
        CachedRound(round_index=0, messages=[Weird()], status="active")
    )
    cache.save("sid_weird")  # must not raise


def test_gc_orphan_groups_removes_sealed_empty(tmp_path):
    cache = _new_cache(tmp_path)
    cache.add_group(CompressionGroup(group_id="g_sealed", layer=1, compressed=True))
    cache.add_group(
        CompressionGroup(
            group_id="g_live", layer=1, summary_text="content", compressed=True
        )
    )
    cache.add_group(
        CompressionGroup(group_id="final", layer=99, summary_text="f", compressed=True)
    )
    dropped = cache.gc_orphan_groups()
    assert dropped == 1
    remaining = {g.group_id for g in cache.groups}
    assert "g_sealed" not in remaining
    assert {"g_live", "final"} <= remaining


def test_gc_runs_during_sync(tmp_path):
    cache = _new_cache(tmp_path)
    cache.add_group(CompressionGroup(group_id="dead", layer=1, compressed=True))
    _sync_rounds(cache, 1)
    assert "dead" not in {g.group_id for g in cache.groups}


# ---------------------------------------------------------------------------
# Summary length cap
# ---------------------------------------------------------------------------


def test_final_summary_within_cap(tmp_path):
    cache = _new_cache(tmp_path)
    _sync_rounds(cache, 4, prefix="cap")
    engine = CompressionEngine(FakeLLM(["A", "B"]), summary_max_chars=50)
    cache.add_group(CompressionGroup(group_id="g0", layer=1, source_rounds=[0, 1]))
    cache.add_group(CompressionGroup(group_id="g1", layer=1, source_rounds=[2, 3]))
    ok = _run(engine.compress_all_pending(cache))
    assert ok
    summary = _run(engine.build_final_summary(cache, covered_ids={0, 1, 2, 3}))
    assert len(summary) <= 50 * 1.1 + 1


def test_final_summary_self_compress_when_over_cap(tmp_path):
    cache = _new_cache(tmp_path)
    _sync_rounds(cache, 2, prefix="sc")
    engine = CompressionEngine(
        FakeLLM(["A" * 200, "B" * 200, "SHORT_SUMMARY"]), summary_max_chars=50
    )
    cache.add_group(
        CompressionGroup(
            group_id="g0", layer=1, source_rounds=[0], summary_text="A" * 200, compressed=True
        )
    )
    cache.add_group(
        CompressionGroup(
            group_id="g1", layer=1, source_rounds=[1], summary_text="B" * 200, compressed=True
        )
    )
    summary = _run(engine.build_final_summary(cache, covered_ids={0, 1}))
    assert len(summary) <= 50 * 1.1 + 1


def test_final_summary_hard_truncate_when_llm_still_long(tmp_path):
    """Even a self-compress that returns too-long text stays bounded."""
    cache = _new_cache(tmp_path)
    _sync_rounds(cache, 2, prefix="ht")
    engine = CompressionEngine(
        FakeLLM(["A" * 200, "B" * 200, "C" * 200]), summary_max_chars=50
    )
    cache.add_group(
        CompressionGroup(
            group_id="g0", layer=1, source_rounds=[0], summary_text="A" * 200, compressed=True
        )
    )
    cache.add_group(
        CompressionGroup(
            group_id="g1", layer=1, source_rounds=[1], summary_text="B" * 200, compressed=True
        )
    )
    summary = _run(engine.build_final_summary(cache, covered_ids={0, 1}))
    assert len(summary) <= 55  # 50 * 1.1 hard cap


# ---------------------------------------------------------------------------
# Plugin helper: current user text (persist filter)
# ---------------------------------------------------------------------------


def test_current_user_text_excludes_persist_false(tmp_path):
    from context_condensation.main import ContextCondensationPlugin
    from core.prompt_manager import Prompt

    plugin = ContextCondensationPlugin.__new__(ContextCondensationPlugin)

    class Req:
        user_prompt = [
            Prompt("<system_reminder>", persist=False),
            Prompt("relocated time", persist=False),
            Prompt("真实用户消息", persist=True),
        ]

    text = plugin._current_user_text(Req())
    assert "system_reminder" not in text
    assert "relocated" not in text
    assert "真实用户消息" in text


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

_ALL_TESTS = [
    test_is_prompt_detects_framework_prompts,
    test_parse_into_rounds_skips_prompts,
    test_sync_ignores_prompt_objects,
    test_save_load_roundtrip,
    test_load_corrupt_file_starts_fresh,
    test_load_empty_file_starts_fresh,
    test_save_with_unserializable_message_does_not_crash,
    test_gc_orphan_groups_removes_sealed_empty,
    test_gc_runs_during_sync,
    test_final_summary_within_cap,
    test_final_summary_self_compress_when_over_cap,
    test_final_summary_hard_truncate_when_llm_still_long,
    test_current_user_text_excludes_persist_false,
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="context_condensation self-test")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    passed = 0
    failed = 0
    for fn in _ALL_TESTS:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = Path(tempfile.mkdtemp(prefix="cc_test_"))
                fn(tmp)
            else:
                fn()  # no tmp_path needed
            passed += 1
            if args.verbose:
                print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
