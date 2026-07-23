"""
Context Condensation - layered context compression plugin.

Per-session strategy (DESIGN §3/§4):
  1. Mirror req.messages into a length-unlimited ContextCache, matching rounds
     by content fingerprint (position-independent).
  2. While the conversation grows, pre-compress growth-zone rounds in a
     background task: preprocess oversized tool results / image descriptions
     (cache only), group by char count, compress groups, then merge summaries
     pairwise per layer. Nothing is removed from the live context yet.
  3. When uncompressed rounds reach the framework's max_memory_length, wait
     for background compression, build the final summary (merging the previous
     cycle's summary, §4.4), then rebuild req.messages as
     [summary][anchor zone originals]. Compressed rounds are stripped from
     every subsequent request and the summary is re-injected each round.
  4. Bailout: if uncompressed rounds pile up to 2x max_memory_length (e.g.
     the compression LLM keeps failing), pause the normal cycle and run a
     one-shot emergency collapse instead (§4.5.2).

KV-cache design (§7):
  - system_prompt is never touched.
  - The summary block sits at a fixed position (right after system, before
    history) and stays byte-identical within a cycle.
  - Growth only appends at the tail -> high prefix cache hit rate.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.plugin import BasePlugin, Priority, logger, on
from core.provider import LLMRequest
from core.agent.message import OpenAIMessage

from .context_cache import (
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_COMPRESSED,
    CachedRound,
    ContextCache,
    msg_get,
    msg_text,
)
from .compressor import CompressionEngine
from .preprocessor import preprocess_round

# Max seconds to wait for a running background compression before injecting (§9.3)
_COMPRESS_WAIT_TIMEOUT = 30.0
_COMPRESS_WAIT_INTERVAL = 0.5


class ContextCondensationPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        basic = cfg.get("section_basic", {})
        comp = cfg.get("section_compression", {})
        adv = cfg.get("section_advanced", {})

        self._enabled: bool = basic.get("enabled", True)
        self._anchor_size: int = max(1, int(basic.get("anchor_size", 5)))
        self._max_chars_per_group: int = max(200, int(basic.get("max_chars_per_group", 800)))

        self._compression_model_id: str = comp.get("compression_model", "") or ""
        self._use_persona: bool = comp.get("use_persona_in_compression", False)
        self._preprocess_tools: bool = comp.get("preprocess_tool_results", True)
        self._tool_max_chars: int = max(200, int(comp.get("tool_result_max_chars", 2000)))
        self._summary_max_chars: int = max(200, int(comp.get("summary_max_chars", 1500)))
        self._summary_prefix: str = comp.get(
            "summary_prefix",
            "[对话历史摘要 - 以下是你与用户之前对话的关键信息，不是用户当前说的话]\n",
        )

        self._async_compression: bool = adv.get("async_compression", True)
        self._inject_on_mismatch: bool = adv.get("inject_on_mismatch", False)
        self._debug: bool = adv.get("debug_log", False)

        # Trigger threshold auto-read from the framework config
        try:
            self._max_context_rounds: int = int(
                self.ctx.config.get_config("bot_config.bot.max_memory_length", 10)
            )
        except (TypeError, ValueError, AttributeError):
            self._max_context_rounds = 10
        # Anchor must stay below the trigger threshold, otherwise the growth
        # zone is always empty and a cycle would fire on every round.
        if self._anchor_size >= self._max_context_rounds:
            self._anchor_size = max(1, self._max_context_rounds - 2)

        self._data_dir: Path = Path(".")
        self._caches: Dict[str, ContextCache] = {}
        self._background_tasks: Dict[str, asyncio.Task] = {}
        # Per-session lock: serializes hook executions of the same sid so two
        # concurrent requests can never run competing compression cycles.
        self._locks: Dict[str, asyncio.Lock] = {}

    # ---- lifecycle -------------------------------------------------------------

    async def initialize(self):
        self._data_dir = Path(self.ctx.get_plugin_data_dir())
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[context_condensation] Initialized "
            f"(anchor={self._anchor_size}, threshold={self._max_context_rounds}, "
            f"bailout={2 * self._max_context_rounds})"
        )

    async def terminate(self):
        # Re-entrant: cancel background tasks, persist caches, drop in-memory state
        for task in list(self._background_tasks.values()):
            if not task.done():
                task.cancel()
        for task in list(self._background_tasks.values()):
            if not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        self._background_tasks.clear()
        for sid, cache in list(self._caches.items()):
            cache.save(sid)
        self._caches.clear()
        self._locks.clear()

    # ---- helpers -----------------------------------------------------------------

    def _get_cache(self, sid: str) -> ContextCache:
        if sid not in self._caches:
            cache = ContextCache(self._data_dir, self._anchor_size)
            cache.load(sid)
            self._caches[sid] = cache
        return self._caches[sid]

    def _get_compression_llm(self):
        if self._compression_model_id:
            client = self.ctx.get_llm_client(self._compression_model_id)
            if client is not None:
                return client
            logger.warning(
                f"[context_condensation] Configured model "
                f"'{self._compression_model_id}' unavailable; using default fast LLM"
            )
        return self.ctx.get_default_fast_llm_client()

    async def _make_engine(self) -> Optional[CompressionEngine]:
        llm = self._get_compression_llm()
        if llm is None:
            logger.warning("[context_condensation] No compression LLM available")
            return None
        persona = ""
        if self._use_persona:
            persona_mgr = getattr(self.ctx, "persona_mgr", None)
            if persona_mgr is not None:
                try:
                    active = await persona_mgr.get_active_persona()
                    if active is not None and getattr(active, "content", None):
                        persona = active.content.strip()
                except Exception:
                    logger.warning("[context_condensation] Failed to fetch active persona")
        return CompressionEngine(
            llm,
            persona=persona,
            max_chars_per_group=self._max_chars_per_group,
            summary_max_chars=self._summary_max_chars,
        )

    def _log_debug(self, msg: str) -> None:
        if self._debug:
            logger.info(f"[context_condensation] {msg}")

    def _get_lock(self, sid: str) -> asyncio.Lock:
        if sid not in self._locks:
            self._locks[sid] = asyncio.Lock()
        return self._locks[sid]

    @staticmethod
    async def _save(cache: ContextCache, sid: str) -> None:
        """Persist without blocking the event loop on disk I/O."""
        try:
            await asyncio.to_thread(cache.save, sid)
        except Exception:
            logger.warning(f"[context_condensation] Failed to persist cache for {sid}")

    async def _cancel_task(self, sid: str) -> None:
        """Cancel and await any running background/emergency task for a sid."""
        task = self._background_tasks.pop(sid, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def _rebuild_messages(
        self,
        req: LLMRequest,
        summary_text: str,
        kept_rounds: List[CachedRound],
    ) -> None:
        """Rebuild req.messages as [summary][kept rounds' original messages]."""
        summary_msg = OpenAIMessage(
            role="user", content=f"{self._summary_prefix}{summary_text}"
        )
        req.messages = [summary_msg] + [m for r in kept_rounds for m in r.messages]

    # ---- main hook (§3.2) -----------------------------------------------------------

    @on.llm_request(priority=Priority.LOW - 1)
    async def inject_compressed_context(self, event, req: LLMRequest, *_):
        """Sync cache, re-inject summary, and drive the compression cycle.

        Runs at LOW-1: after all other plugins finished their prompt injection,
        right before assemble_prompt().
        """
        if not self._enabled:
            return
        sid = getattr(event, "sid", None) or getattr(getattr(event, "session", None), "sid", None)
        if not sid:
            return

        # Failsafe: snapshot the original context. On ANY plugin error the
        # framework must see req.messages exactly as if we never ran.
        # (Our rebuilds rebind req.messages to a new list instead of mutating
        # it in place, so this reference always stays valid.)
        original_messages = req.messages
        try:
            async with self._get_lock(sid):
                await self._process(sid, req)
        except Exception:
            logger.exception(
                f"[context_condensation] sid={sid} hook failed; "
                f"context passed through untouched"
            )
            req.messages = original_messages
            return
        # Defensive invariant: never turn a non-empty context into an empty one
        if original_messages and not req.messages:
            logger.warning(
                f"[context_condensation] sid={sid} rebuild produced empty messages; restored"
            )
            req.messages = original_messages

    async def _process(self, sid: str, req: LLMRequest) -> None:
        cache = self._get_cache(sid)
        added, present = cache.sync(req.messages)

        # Severe mismatch (history cleared / framework reset) — §9.1
        if cache.detect_mismatch(present):
            await self._handle_mismatch(sid, cache, present, req)
            return

        vanished = cache.archive_absent(present)

        # Re-inject: strip rounds covered by the final summary, prepend it (§9.5)
        self._apply_summary(req, cache, present)

        uncompressed = cache.uncompressed_of(present)
        self._log_debug(
            f"sid={sid} total={cache.total_rounds} present={len(present)} "
            f"uncompressed={len(uncompressed)} new={len(added)} vanished={len(vanished)}"
        )

        # Bailout threshold: 2x max_memory_length — pause normal cycle (§4.5.2)
        if len(uncompressed) >= 2 * self._max_context_rounds:
            self._start_emergency(sid)
            return

        # Injection threshold reached — run a compression cycle (§4.1).
        # Vanished-but-cached rounds join the compression set so their content
        # is absorbed into the summary instead of being lost.
        if len(uncompressed) >= self._max_context_rounds:
            await self._run_injection_cycle(sid, cache, uncompressed, req, vanished)
            return

        # Growth phase: pre-compress in the background (§4.3)
        if len(uncompressed) > self._anchor_size:
            self._start_background(sid)

    # ---- summary re-injection ---------------------------------------------------

    def _apply_summary(self, req: LLMRequest, cache: ContextCache, present: List[CachedRound]) -> None:
        covered = cache.covered_of(present)
        summary_text = cache.final_summary_text()
        if not summary_text:
            if not covered:
                return
            # Covered rounds but no summary (corrupted state): resurrect them
            # rather than silently dropping context. Reset their group link so
            # they can be re-grouped and re-compressed.
            for r in covered:
                r.status = STATUS_ACTIVE
                r.compression_group = None
            # Old group summaries must not be merged into a future final
            # summary again — the rounds are back in the live context.
            cache.drop_non_final_groups()
            logger.warning(
                "[context_condensation] Covered rounds without a final summary; resurrected"
            )
            return
        # The summary covers ALL compressed/archived history — including rounds
        # the framework has already truncated out of req.messages. Injection
        # must therefore depend on the CACHE having covered rounds, not on
        # covered rounds still being present; otherwise that history would
        # silently vanish from the model's context (§9.5).
        if not covered and not cache.has_covered_rounds():
            return
        kept = cache.uncompressed_of(present)
        self._rebuild_messages(req, summary_text, kept)
        self._log_debug(
            f"Re-injected summary ({len(summary_text)} chars), "
            f"stripped {len(covered)} covered rounds"
        )

    # ---- injection cycle (§4.1) -----------------------------------------------------

    async def _run_injection_cycle(
        self,
        sid: str,
        cache: ContextCache,
        uncompressed: List[CachedRound],
        req: LLMRequest,
        vanished: Optional[List[CachedRound]] = None,
    ) -> None:
        growth = cache.growth_of(uncompressed)
        # Merge in vanished-but-cached rounds (chronological order)
        if vanished:
            merged = list(growth)
            seen = {r.round_index for r in merged}
            merged.extend(r for r in vanished if r.round_index not in seen)
            merged.sort(key=lambda r: r.round_index)
            growth = merged
        if not growth:
            return  # nothing beyond the anchor zone yet

        # Wait for any running background compression (§9.3)
        await self._wait_for_background(sid)

        engine = await self._make_engine()
        if engine is None:
            return

        # Preprocess growth rounds that have not been preprocessed yet (cache only)
        if self._preprocess_tools:
            await self._preprocess_rounds(growth, engine)

        # Make sure every growth round is grouped and every REQUIRED group is
        # compressed. Rounds already done by the background pass are skipped;
        # unrelated pending groups (e.g. anchor rounds grouped early) do not
        # block this cycle.
        engine.plan_layer1_groups(cache, growth)
        growth_ids = {r.round_index for r in growth}
        if not await engine.compress_all_pending(cache, only_rounds=growth_ids):
            # Compression keeps failing: do not inject a partial summary.
            # The context keeps growing until the bailout threshold takes over.
            logger.warning(
                f"[context_condensation] sid={sid} compression incomplete; "
                f"postponing injection"
            )
            await self._save(cache, sid)
            return

        await engine.merge_available_pairs(cache)
        summary_text = await engine.build_final_summary(cache)
        if not summary_text:
            logger.warning(f"[context_condensation] sid={sid} no summary produced")
            await self._save(cache, sid)
            return

        # Status transitions (§5.2): previous cycle's compressed -> archived,
        # this cycle's growth -> compressed. Anchor stays active.
        old_covered = [
            r.round_index for r in cache.rounds if r.status == STATUS_COMPRESSED
        ]
        cache.mark_rounds_status(old_covered, STATUS_ARCHIVED)
        cache.mark_rounds_status(
            [r.round_index for r in growth], STATUS_COMPRESSED
        )

        anchor = cache.anchor_of(cache.uncompressed_of(uncompressed))
        self._rebuild_messages(req, summary_text, anchor)
        await self._save(cache, sid)

        self._log_debug(
            f"sid={sid} cycle done: summary={len(summary_text)} chars, "
            f"compressed={len(growth)} rounds, anchor={len(anchor)} rounds, "
            f"trace={cache.trace_summary_to_rounds('final')}"
        )

    async def _preprocess_rounds(self, rounds: List[CachedRound], engine: CompressionEngine) -> None:
        """Produce condensed copies for compression input (§6).

        Results go to ``r.condensed_messages``; ``r.messages`` always keeps
        the original framework content so the anchor zone stays verbatim.
        """
        for r in rounds:
            if r.is_preprocessed or not r.messages:
                continue
            condensed = r.messages
            try:
                condensed = await preprocess_round(r.messages, self._tool_max_chars, engine.llm)
            except Exception as e:
                logger.warning(
                    f"[context_condensation] Preprocess failed for round {r.round_index}: {e}"
                )
            r.condensed_messages = condensed
            # Grouping uses the condensed size (§6.4); sync() will no longer
            # overwrite total_chars for preprocessed rounds.
            condensed_chars = sum(len(msg_text(msg_get(m, "content", ""))) for m in condensed)
            r.total_chars = condensed_chars or r.total_chars
            r.is_preprocessed = True

    # ---- background compression (§4.3) --------------------------------------------

    def _start_background(self, sid: str) -> None:
        task = self._background_tasks.get(sid)
        if task is not None and not task.done():
            return
        if not self._async_compression:
            # Sync mode is handled lazily inside the injection cycle
            return
        self._background_tasks[sid] = asyncio.create_task(self._background_pass(sid))

    async def _background_pass(self, sid: str) -> None:
        try:
            cache = self._caches.get(sid)
            if cache is None:
                return
            tracked = cache.tracked_rounds()
            growth = cache.growth_of(tracked)
            if not growth:
                return
            engine = await self._make_engine()
            if engine is None:
                return
            if self._preprocess_tools:
                await self._preprocess_rounds(growth, engine)
            # Group only once enough ungrouped content has accumulated;
            # otherwise rounds arriving one at a time would each form their
            # own tiny group (more LLM calls, worse summaries). The injection
            # cycle always groups whatever is left.
            ungrouped_chars = sum(
                r.total_chars for r in growth if r.compression_group is None
            )
            if ungrouped_chars >= self._max_chars_per_group:
                engine.plan_layer1_groups(cache, growth)
            await engine.compress_all_pending(cache)
            await engine.merge_available_pairs(cache)
            await self._save(cache, sid)
            self._log_debug(f"sid={sid} background pass done")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[context_condensation] sid={sid} background pass error")

    async def _wait_for_background(self, sid: str) -> None:
        task = self._background_tasks.get(sid)
        if task is None or task.done():
            return
        self._log_debug(f"sid={sid} waiting for background compression...")
        waited = 0.0
        while not task.done() and waited < _COMPRESS_WAIT_TIMEOUT:
            await asyncio.sleep(_COMPRESS_WAIT_INTERVAL)
            waited += _COMPRESS_WAIT_INTERVAL
        if not task.done():
            # Timeout: proceed with partial results; the cycle completes next round (§9.3)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            logger.warning(
                f"[context_condensation] sid={sid} background wait timed out; using partial results"
            )

    # ---- emergency collapse (§4.5.2) ------------------------------------------------

    def _start_emergency(self, sid: str) -> None:
        task = self._background_tasks.get(sid)
        if task is not None and not task.done():
            return
        logger.warning(
            f"[context_condensation] sid={sid} cache piled up beyond "
            f"{2 * self._max_context_rounds} rounds; starting emergency collapse"
        )
        self._background_tasks[sid] = asyncio.create_task(self._emergency_pass(sid))

    async def _emergency_pass(self, sid: str) -> None:
        try:
            cache = self._caches.get(sid)
            if cache is None:
                return
            tracked = cache.tracked_rounds()
            growth = cache.growth_of(tracked)
            if not growth:
                return
            engine = await self._make_engine()
            if engine is None:
                return
            # Preprocess first (consistent with the background pass): a raw
            # oversized tool result could otherwise single-handedly exceed the
            # emergency input cap.
            if self._preprocess_tools:
                await self._preprocess_rounds(growth, engine)
            text, included = await engine.emergency_collapse(cache, growth)
            if not text:
                logger.warning(
                    f"[context_condensation] sid={sid} emergency collapse failed; retry next round"
                )
                return
            included_set = set(included)
            old_covered = [
                r.round_index for r in cache.rounds if r.status == STATUS_COMPRESSED
            ]
            cache.mark_rounds_status(old_covered, STATUS_ARCHIVED)
            cache.mark_rounds_status(included, STATUS_COMPRESSED)
            # Rounds excluded by the input cap (oldest) were NOT summarized;
            # archive them explicitly instead of pretending they are covered.
            excluded = [r.round_index for r in growth if r.round_index not in included_set]
            if excluded:
                cache.mark_rounds_status(excluded, STATUS_ARCHIVED)
                logger.warning(
                    f"[context_condensation] sid={sid} {len(excluded)} oldest rounds "
                    f"exceeded the emergency input cap and were archived unsummarized"
                )
            await self._save(cache, sid)
            logger.info(
                f"[context_condensation] sid={sid} emergency collapse done: "
                f"{len(included)} rounds -> {len(text)} chars"
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[context_condensation] sid={sid} emergency collapse error")

    # ---- mismatch handling (§9.1) ------------------------------------------------------

    async def _handle_mismatch(
        self,
        sid: str,
        cache: ContextCache,
        present: List[CachedRound],
        req: LLMRequest,
    ) -> None:
        # A background/emergency task may still be running against this cache;
        # cancel it first so it cannot resurrect stale state after we clear or
        # rebuild (the task holds a reference to the same cache object).
        await self._cancel_task(sid)

        if not self._inject_on_mismatch:
            # Default: wipe the cache and start fresh — safest option
            logger.warning(
                f"[context_condensation] sid={sid} severe context mismatch; clearing cache"
            )
            cache.clear()
            cache.sync(req.messages)
            await self._save(cache, sid)
            return

        # Compress all vanished uncompressed rounds into the summary, keep the
        # rounds still present as the new active starting point.
        logger.warning(
            f"[context_condensation] sid={sid} severe context mismatch; "
            f"compressing cached history before injection"
        )
        present_ids = {r.round_index for r in present}
        vanished = [
            r for r in cache.tracked_rounds() if r.round_index not in present_ids
        ]
        engine = await self._make_engine()
        if engine is not None and vanished:
            text, _included = await engine.emergency_collapse(cache, vanished)
            if text:
                # All vanished rounds become history (summarized or not)
                cache.mark_rounds_status(
                    [r.round_index for r in vanished], STATUS_ARCHIVED
                )
        await self._save(cache, sid)

        summary_text = cache.final_summary_text()
        if summary_text:
            self._rebuild_messages(req, summary_text, cache.uncompressed_of(present))
