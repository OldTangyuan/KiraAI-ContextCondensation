"""
Context Condensation - layered context compression plugin (write-through).

Per-session strategy:
  1. Mirror framework memory into a length-unlimited ContextCache, matching
     rounds by content fingerprint (position-independent). Sync happens both
     at ON_LLM_REQUEST and ~1s after every reply (ON_STEP_RESULT), so the
     cache tracks framework memory in near real time.
  2. A background pipeline keeps itself fully caught up after every turn:
     preprocess oversized tool results / image descriptions (condensed copies
     only), group growth rounds by accumulated chars, compress each group,
     then merge same-layer summaries pairwise (a lone summary is never
     re-compressed by itself — it waits for a partner). The pass loops until
     no progress is made, so reaching the threshold never means starting
     compression from scratch.
  3. When uncompressed rounds reach the framework's max_memory_length, only
     the final merge remains (~1 LLM call): the previous final summary plus
     the ready top summaries become the new final summary (§4.4), and the
     result is WRITTEN BACK to framework memory via
     SessionManager.write_memory as [summary chunk][anchor chunks].
     Afterwards the framework itself loads the compressed context; no
     per-round stripping is needed.
  4. If the framework later truncates the summary chunk before the next
     cycle fires, the cached final summary is prepended in-memory as a
     fallback bridge (§9.5).
  5. Bailout: at 2x max_memory_length uncompressed rounds, pause the normal
     cycle and run a one-shot emergency collapse instead (§4.5.2).
  6. Rounds covered by the final summary have their stored content cleared
     immediately (memory + disk hygiene).

KV-cache design (§7):
  - system_prompt is never touched.
  - The summary chunk sits at a fixed position (right after system) and stays
    byte-identical within a cycle; growth only appends at the tail.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.plugin import BasePlugin, Priority, get_logger, on
from core.provider import LLMRequest
from core.agent.message import OpenAIMessage

from .context_cache import (
    CachedRound,
    ContextCache,
    is_prompt,
    msg_get,
    msg_text,
)
from .compressor import CompressionEngine
from .preprocessor import preprocess_round

# Delay between a finished reply and the real-time cache re-sync
_POST_REPLY_SYNC_DELAY = 1.0
# Max iterations of one background pass (bounds retries when the LLM is down)
_BACKGROUND_MAX_LOOPS = 4
# Minimum seconds between background passes after a failed (no-progress) pass,
# so a down LLM cannot trigger a retry storm every turn.
_BACKGROUND_RETRY_DELAY = 15.0

logger = get_logger('context_condensation.main', 'blue')

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
        self._tool_max_chars: int = max(200, int(comp.get("tool_result_max_chars", 1500)))
        self._summary_max_chars: int = max(200, int(comp.get("summary_max_chars", 1500)))
        self._summary_prefix: str = comp.get(
            "summary_prefix",
            "[对话历史摘要 - 以下是你与用户之前对话的关键信息，不是用户当前说的话，同时摘要可能有损，不一定准确]\n",
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
        self._post_reply_tasks: Dict[str, asyncio.Task] = {}
        # Per-session lock: serializes hook executions of the same sid so two
        # concurrent requests can never run competing compression cycles.
        self._locks: Dict[str, asyncio.Lock] = {}
        # Per-session monotonic timestamp of the last background pass that made
        # no progress (LLM failures) — used to throttle retries.
        self._last_background_fail: Dict[str, float] = {}

    # ---- lifecycle -------------------------------------------------------------

    async def initialize(self):
        # get_plugin_data_dir() resolves via the caller's module to the active
        # plugin_id; it can return None when plugin_mgr is unset or the module
        # is not registered. Fall back to a stable path in that case.
        base = None
        try:
            base = self.ctx.get_plugin_data_dir()
        except Exception:
            base = None
        if base is None:
            from core.utils.path_utils import get_data_path
            base = get_data_path() / "plugin_data" / "context_condensation"
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)

        # Cache continuity: plugin_id changed from "context_condensation" to
        # "KiraAI-ContextCondensation" at some point, so the active plugin dir
        # may be a fresh empty directory while real caches still live under the
        # old name. Migrate once (only when the active dir has no caches yet).
        active_caches = base / "caches"
        legacy_caches = base.parent / "context_condensation" / "caches"
        if (
            legacy_caches != active_caches  # guard the fallback-path self-move case
            and not list(active_caches.glob("*.json"))
            and legacy_caches.exists()
        ):
            try:
                active_caches.mkdir(parents=True, exist_ok=True)
                for f in legacy_caches.glob("*.json"):
                    try:
                        f.replace(active_caches / f.name)
                    except OSError as e:
                        logger.warning(
                            f"[context_condensation] Failed to migrate cache "
                            f"{f.name}: {e}"
                        )
                migrated = len(list(active_caches.glob("*.json")))
                logger.info(
                    f"[context_condensation] Migrated {migrated} cache file(s) "
                    f"from legacy plugin dir"
                )
            except OSError as e:
                logger.warning(f"[context_condensation] Cache migration failed: {e}")

        self._data_dir = base
        logger.info(
            f"[context_condensation] Initialized "
            f"(anchor={self._anchor_size}, threshold={self._max_context_rounds}, "
            f"bailout={2 * self._max_context_rounds}, write-through)"
        )

    async def terminate(self):
        # Re-entrant: cancel all tasks, persist caches, drop in-memory state
        for tasks in (self._background_tasks, self._post_reply_tasks):
            for task in list(tasks.values()):
                if not task.done():
                    task.cancel()
            for task in list(tasks.values()):
                if not task.done():
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            tasks.clear()
        for sid, cache in list(self._caches.items()):
            try:
                cache.save(sid)
            except Exception:
                pass  # best-effort persist during shutdown
        self._caches.clear()
        self._locks.clear()
        self._last_background_fail.clear()

    # ---- helpers -----------------------------------------------------------------

    def _get_cache(self, sid: str) -> ContextCache:
        if sid not in self._caches:
            cache = ContextCache(self._data_dir, self._anchor_size)
            cache.load(sid)
            self._caches[sid] = cache
        return self._caches[sid]

    def _get_compression_llm(self):
        """Resolve the compression LLM with a full fallback chain.

        get_default_fast_llm_client()/get_default_llm_client() RAISE (e.g.
        ValueError when unconfigured) rather than return None — every getter
        must be guarded, or the whole pipeline dies silently.
        """
        candidates = []
        if self._compression_model_id:
            candidates.append(
                (f"configured '{self._compression_model_id}'",
                 lambda: self.ctx.get_llm_client(self._compression_model_id))
            )
        candidates.append(("fast LLM", self.ctx.get_default_fast_llm_client))
        candidates.append(("default LLM", self.ctx.get_default_llm_client))
        for label, getter in candidates:
            try:
                client = getter()
            except Exception:
                client = None
            if client is not None:
                if label.startswith("configured") or label != "fast LLM":
                    self._log_debug(f"Compression LLM resolved via {label}")
                return client
        logger.warning(
            "[context_condensation] No compression LLM available "
            "(configure a fast/default LLM or compression_model)"
        )
        return None

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
        # debug_log=true is the user-facing "verbose" switch: emit at INFO so
        # the pipeline detail is actually visible with the default global level
        # (DEBUG would be filtered by the console handler at INFO). Guarded by
        # self._debug so default installs stay quiet.
        if self._debug:
            logger.info(f"[context_condensation] {msg}")

    @staticmethod
    def _current_user_text(req: LLMRequest) -> str:
        """Join the current incoming user input from persist=True prompts.

        The framework marks dynamic blocks (sessions/chat_env/time relocated
        with ``dynamic_prompt_position="latest_user"`` and the
        ``<system_reminder>`` wrapper) as ``persist=False`` — they are prompt
        scaffolding, not the user's actual message, and must be excluded.
        """
        parts: List[str] = []
        for p in getattr(req, "user_prompt", []) or []:
            if not isinstance(p, object):
                continue
            if getattr(p, "persist", True) is False:
                continue
            text = (getattr(p, "content", "") or "").strip()
            if text:
                parts.append(text)
        return "\n".join(parts)

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

    async def _cancel_task(self, tasks: Dict[str, asyncio.Task], sid: str) -> None:
        """Cancel and await any running task for a sid."""
        task = tasks.pop(sid, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def _summary_chunk_content(self, summary_text: str) -> str:
        return f"{self._summary_prefix}{summary_text}"

    def _strip_summary_chunk(
        self, messages: List[Any], cache: ContextCache
    ) -> Tuple[List[Any], bool]:
        """Detect our summary chunk at the head of framework memory.

        Returns (messages_without_chunk, chunk_found). A chunk is only honored
        when it matches the cached final summary (or the cache has none, in
        which case the chunk is adopted); otherwise it is treated as ordinary
        user content.
        """
        if not messages:
            return messages, False
        first = messages[0]
        if msg_get(first, "role") != "user":
            return messages, False
        text = msg_text(msg_get(first, "content", ""))
        if not text.startswith(self._summary_prefix):
            return messages, False
        summary = text[len(self._summary_prefix):].strip()
        final = cache.final_summary_text().strip()
        if final and summary != final:
            return messages, False
        if not final and summary:
            cache.adopt_final_summary(summary)
            self._log_debug("Adopted summary chunk from framework memory")
        return list(messages[1:]), True

    def _sync_from_framework(
        self, cache: ContextCache, messages: List[Any]
    ) -> Tuple[List[CachedRound], List[CachedRound], bool]:
        """Strip the summary chunk (if any) and sync the rest into the cache."""
        remaining, summary_present = self._strip_summary_chunk(messages, cache)
        added, present = cache.sync(remaining)
        return added, present, summary_present

    def _rebuild_messages(
        self,
        req: LLMRequest,
        summary_text: str,
        kept_rounds: List[CachedRound],
    ) -> None:
        """Rebuild req.messages as [summary][kept rounds' original messages]."""
        summary_msg = OpenAIMessage(
            role="user", content=self._summary_chunk_content(summary_text)
        )
        req.messages = [summary_msg] + [m for r in kept_rounds for m in r.messages]

    async def _write_framework_memory(
        self, sid: str, summary_text: str, anchor_rounds: List[CachedRound]
    ) -> None:
        """Write the compressed context back to framework memory (write-through).

        The summary message is FUSED into the first anchor chunk as
        [summary, user, assistant, ...] instead of being a standalone chunk,
        so framework memory ends up with exactly ``len(anchor_rounds)``
        chunks and the summary never takes an extra window slot.
        """
        chunks: List[List[dict]] = []
        summary_msg = {"role": "user", "content": self._summary_chunk_content(summary_text)}
        if anchor_rounds:
            chunks.append(
                [summary_msg] + ContextCache._serialize_messages(anchor_rounds[0].messages)
            )
            for r in anchor_rounds[1:]:
                chunks.append(ContextCache._serialize_messages(r.messages))
        else:
            chunks.append([summary_msg])
        try:
            await asyncio.to_thread(
                self.ctx.session_mgr.write_memory, sid, chunks
            )
        except Exception as e:
            # In-memory rebuild still happened; next cycle will retry the write
            logger.error(
                f"[context_condensation] sid={sid} write_memory failed: {e}"
            )

    # ---- main hook: ON_LLM_REQUEST ------------------------------------------------

    @on.llm_request(priority=Priority.LOW - 1)
    async def inject_compressed_context(self, event, req: LLMRequest, *_):
        """Sync cache, fall back to summary injection if needed, drive the cycle.

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
        original_messages = req.messages
        try:
            async with self._get_lock(sid):
                await self._process(sid, req)
        except Exception as e:
            logger.error(
                f"[context_condensation] sid={sid} hook failed: {e}; "
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
        # Normalize: ephemeral Prompt scaffolding (persist=False dynamic blocks
        # and any other plugin-injected Prompt) is not conversation history and
        # must never enter the cache or be rebuilt into req.messages.
        history = [m for m in req.messages if not is_prompt(m)]
        added, present, summary_present = self._sync_from_framework(cache, history)

        # Severe mismatch (history cleared / framework reset) — §9.1
        if cache.detect_mismatch(present):
            await self._handle_mismatch(sid, cache, present, req)
            return

        vanished = cache.archive_absent(present)

        # Fallback bridge (§9.5): the summary chunk is gone from framework
        # memory (truncated before the next cycle fired) but the cache still
        # holds the final summary — prepend it in-memory for this turn.
        summary_text = cache.final_summary_text()
        if not summary_present and summary_text:
            kept = cache.uncompressed_of(present)
            self._rebuild_messages(req, summary_text, kept)
            self._log_debug("Fallback summary injection (chunk trimmed by framework)")

        uncompressed = cache.uncompressed_of(present)
        self._log_debug(
            f"sid={sid} total={cache.total_rounds} present={len(present)} "
            f"uncompressed={len(uncompressed)} new={len(added)} vanished={len(vanished)}"
        )
        if self._debug:
            user_text = self._current_user_text(req)
            if user_text:
                self._log_debug(f"sid={sid} current user input: {user_text[:120]!r}")
        # Real-time persistence: the cache file tracks every new round
        if added:
            await self._save(cache, sid)

        # Bailout threshold: 2x max_memory_length — pause normal cycle (§4.5.2)
        if len(uncompressed) >= 2 * self._max_context_rounds:
            self._start_emergency(sid)
            return

        # Injection threshold reached — run a compression cycle (§4.1)
        if len(uncompressed) >= self._max_context_rounds:
            await self._run_injection_cycle(sid, cache, uncompressed, req, vanished)
            return

        # Growth phase: keep the pipeline fully caught up in the background
        if self._has_pending_work(cache):
            self._start_background(sid)

    # ---- real-time hook: ON_STEP_RESULT -----------------------------------------

    @on.step_result(priority=Priority.LOW)
    async def post_reply_sync(self, event, *_):
        """Re-sync the cache from framework memory right after a reply.

        The framework appends the new round to chat_memory.json immediately
        after the agent loop; a short debounced delay lets that write land,
        then the cache is updated and the compression pipeline is driven —
        without waiting for the user's next message.
        """
        if not self._enabled:
            return
        sid = getattr(event, "sid", None) or getattr(getattr(event, "session", None), "sid", None)
        if not sid:
            return
        task = self._post_reply_tasks.get(sid)
        if task is not None and not task.done():
            return  # debounce: one pending sync per session
        self._post_reply_tasks[sid] = asyncio.create_task(self._post_reply_sync(sid))

    async def _post_reply_sync(self, sid: str) -> None:
        try:
            await asyncio.sleep(_POST_REPLY_SYNC_DELAY)
            session_mgr = getattr(self.ctx, "session_mgr", None)
            if session_mgr is None:
                return
            # Avoid recreating a deleted session via fetch_memory's ensure logic
            try:
                if session_mgr.get_memory_count(sid) <= 0:
                    return
            except Exception:
                # Session was deleted in the meantime — nothing to sync.
                return
            async with self._get_lock(sid):
                cache = self._get_cache(sid)
                added, present, _ = self._sync_from_framework(
                    cache, session_mgr.fetch_memory(sid)
                )
                cache.archive_absent(present)
                if added:
                    await self._save(cache, sid)
                    self._log_debug(
                        f"sid={sid} post-reply sync: {len(added)} new round(s) cached"
                    )
                # Drive the pipeline after EVERY reply (even without new rounds):
                # a full-but-uncompressed window must keep pre-compressing so
                # the next cycle finds everything ready.
                if self._has_pending_work(cache):
                    self._start_background(sid)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A deleted session or transient session-manager error must not
            # surface as a traceback flood.
            logger.debug(f"[context_condensation] sid={sid} post-reply sync error: {e}")

    # ---- injection cycle (write-through) -----------------------------------------

    async def _run_injection_cycle(
        self,
        sid: str,
        cache: ContextCache,
        uncompressed: List[CachedRound],
        req: LLMRequest,
        vanished: Optional[List[CachedRound]] = None,
    ) -> None:
        """Compress the whole growth zone in ONE shot and inject the final summary.

        Timing design: at the injection threshold the cycle is SELF-CONTAINED
        and does a SINGLE LLM call for the final summary. It cancels any
        in-flight background pass (so it cannot race our cache mutation), then
        builds the final summary from [old final summary] + [every ready top
        summary] + [raw content of the remaining growth rounds] in one merge
        call. Nothing is pre-compressed group-by-group here — that is the
        background pipeline's job. So final latency is exactly ONE LLM call,
        whether or not the background caught up.

        The entire growth zone is replaced by the summary (it never defers a
        whole turn). If the LLM is down, nothing is replaced this turn — the
        window passes through and the next cycle retries.
        """
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

        # Stop any running pipeline pass BEFORE touching group state: both this
        # cycle and the pass would otherwise mutate the same cache concurrently.
        await self._cancel_task(self._background_tasks, sid)

        engine = await self._make_engine()
        if engine is None:
            return

        # Preprocess oversized tool results / image descriptions in the CACHE
        # (never the live context) so the serialized input stays compact. This
        # is optional and best-effort — a failure must not block the cycle.
        # Do NOT preprocess here: pre-processing is a background-pipeline
        # optimization (it shrinks oversized tool results before they enter
        # compression). At the injection threshold the goal is ONE shot — we
        # fold the growth content into the final merge directly. Running
        # per-round pre-processing here would fire an LLM call per oversized
        # round, reintroducing the exact stall we are removing.
        # Everything in the growth zone is replaced by the summary.
        covered_ids = {r.round_index for r in growth}

        # ONE-shot final summary: old final + ready tops + raw growth content.
        summary_text = await engine.build_final_summary(cache, covered_ids=covered_ids)
        if not summary_text:
            self._start_background(sid)
            await self._save(cache, sid)
            logger.warning(
                f"[context_condensation] sid={sid} cycle: final summary failed "
                f"(LLM unavailable); passing through uncompressed"
            )
            return

        # Kept = the anchor tail only (chronological)
        kept = [r for r in uncompressed if r.round_index not in covered_ids]

        # Write-through: framework memory becomes [summary fused into first
        # kept chunk] + kept chunks
        await self._write_framework_memory(sid, summary_text, kept)
        # The current request uses the compressed context immediately
        self._rebuild_messages(req, summary_text, kept)

        # Cleanup: all growth rounds are DELETED — their content lives in the
        # final summary now, keeping them would only bloat the cache.
        cache.delete_rounds(list(covered_ids))
        await self._save(cache, sid)

        logger.info(
            f"[context_condensation] sid={sid} cycle done: compressed "
            f"{len(covered_ids)} growth rounds in one pass -> {len(summary_text)} chars, "
            f"kept={len(kept)} rounds"
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
            # Keep the condensed copy only when it actually differs — storing
            # an identical duplicate would just bloat the cache file.
            r.condensed_messages = condensed if condensed != r.messages else None
            # Grouping uses the condensed size (§6.4); sync() will no longer
            # overwrite total_chars for preprocessed rounds.
            condensed_chars = sum(len(msg_text(msg_get(m, "content", ""))) for m in condensed)
            r.total_chars = condensed_chars or r.total_chars
            r.is_preprocessed = True

    # ---- continuous background pipeline -----------------------------------------

    def _has_pending_work(self, cache: ContextCache) -> bool:
        """Whether the pipeline has anything compressible right now.

        Eager: ANY ungrouped growth-zone round (or pending/mergeable group)
        triggers a background pass. Compression happens in the CACHE only and
        never touches the live context, so the earlier it runs the better —
        the injection cycle then finds ready groups instead of stalling on a
        pile of LLM calls when the window is full. ``_background_pass`` still
        batches short rounds (never compressing a single round by itself), so
        this is frequent without being per-round.
        """
        growth = cache.growth_of(cache.tracked_rounds())
        if any(r.compression_group is None for r in growth):
            return True
        if any(g.layer == 1 and not g.compressed for g in cache.groups):
            return True
        tops = [g for g in cache.unmerged_tops() if g.summary_text]
        return len(tops) >= 2

    def _start_background(self, sid: str) -> None:
        task = self._background_tasks.get(sid)
        if task is not None and not task.done():
            return
        if not self._async_compression:
            # Sync mode is handled lazily inside the injection cycle
            return
        # Throttle: if the last pass made no progress (LLM down), wait out the
        # backoff window before retrying so a dead provider cannot trigger a
        # retry storm every single turn.
        last_fail = self._last_background_fail.get(sid, 0.0)
        if last_fail and (time.monotonic() - last_fail) < _BACKGROUND_RETRY_DELAY:
            return
        self._background_tasks[sid] = asyncio.create_task(self._background_pass(sid))

    async def _background_pass(self, sid: str) -> None:
        """Run the pipeline until no more progress can be made.

        Each iteration: preprocess -> group growth rounds -> compress every
        pending group -> merge every pairable same-layer summary. The loop
        stops when an iteration produces nothing (all done, or the LLM keeps
        failing — the next turn retries).

        Grouping batches short rounds together (complete_only=True keeps a
        partial tail ungrouped so future rounds join it). A lone short round
        stays ungrouped to wait for a partner; full char-batches and single
        over-long rounds seal immediately. The injection cycle seals EVERYTHING
        (complete_only=False), so at final time nothing is left ungrouped.
        """
        try:
            cache = self._caches.get(sid)
            if cache is None:
                return
            engine = await self._make_engine()
            if engine is None:
                return
            for _ in range(_BACKGROUND_MAX_LOOPS):
                progress = 0
                growth = cache.growth_of(cache.tracked_rounds())
                if growth and self._preprocess_tools:
                    await self._preprocess_rounds(growth, engine)
                # Group with complete_only=True: short rounds batch together and
                # a lone short round stays ungrouped to wait for a partner. Full
                # char-batches (and single over-long rounds) seal immediately.
                # The injection cycle seals EVERYTHING (complete_only=False),
                # so at final time nothing is left ungrouped.
                if growth:
                    progress += len(
                        engine.plan_layer1_groups(cache, growth, complete_only=True)
                    )
                # Was there anything that would trigger an LLM call? If not, the
                # no-progress exit below is "waiting for a partner", NOT an LLM
                # failure — it must not arm the backoff throttle.
                had_llm_work = any(
                    g.layer == 1 and not g.compressed for g in cache.groups
                ) or any(
                    g.summary_text for g in cache.unmerged_tops()
                )
                pending_before = sum(
                    1 for g in cache.groups if g.layer == 1 and g.compressed
                )
                await engine.compress_all_pending(cache)
                progress += sum(
                    1 for g in cache.groups if g.layer == 1 and g.compressed
                ) - pending_before
                progress += await engine.merge_available_pairs(cache)
                if progress:
                    await self._save(cache, sid)
                else:
                    if had_llm_work:
                        # LLM calls were attempted and made no progress: the LLM
                        # is likely down. Arm the backoff so the next turn does
                        # not immediately retry.
                        self._last_background_fail[sid] = time.monotonic()
                    break
            self._log_debug(f"sid={sid} background pipeline caught up")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Single-line error: a transient failure (LLM down, provider error)
            # must not flood the log with a full traceback on every turn.
            self._last_background_fail[sid] = time.monotonic()
            logger.error(f"[context_condensation] sid={sid} background pass error: {e}")

    # ---- emergency collapse (§4.5.2) ------------------------------------------------

    def _start_emergency(self, sid: str) -> None:
        task = self._background_tasks.get(sid)
        if task is not None and not task.done():
            return
        # Throttle repeated emergency attempts just like background passes.
        last_fail = self._last_background_fail.get(sid, 0.0)
        if last_fail and (time.monotonic() - last_fail) < _BACKGROUND_RETRY_DELAY:
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
            excluded = [r.round_index for r in growth if r.round_index not in included_set]
            if excluded:
                logger.warning(
                    f"[context_condensation] sid={sid} {len(excluded)} oldest rounds "
                    f"exceeded the emergency input cap and were dropped unsummarized"
                )
            anchor = cache.anchor_of(tracked)
            # Write the collapsed context back, then delete all covered rounds
            await self._write_framework_memory(sid, text, anchor)
            cache.delete_rounds([r.round_index for r in growth])
            await self._save(cache, sid)
            logger.info(
                f"[context_condensation] sid={sid} emergency collapse done: "
                f"{len(included)} rounds -> {len(text)} chars"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._last_background_fail[sid] = time.monotonic()
            logger.error(
                f"[context_condensation] sid={sid} emergency collapse error: {e}"
            )

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
        await self._cancel_task(self._background_tasks, sid)

        if not self._inject_on_mismatch:
            # Default: wipe the cache and start fresh — safest option
            logger.warning(
                f"[context_condensation] sid={sid} severe context mismatch; clearing cache"
            )
            cache.clear()
            cache.sync([m for m in req.messages if not is_prompt(m)])
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
                # All vanished rounds become history — delete them outright
                cache.delete_rounds([r.round_index for r in vanished])
        await self._save(cache, sid)

        summary_text = cache.final_summary_text()
        if summary_text:
            kept = cache.uncompressed_of(present)
            await self._write_framework_memory(sid, summary_text, kept)
            self._rebuild_messages(req, summary_text, kept)
