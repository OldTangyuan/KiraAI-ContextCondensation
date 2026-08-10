"""
Layered compression pipeline (DESIGN §4).

Core rules:
  - Group by accumulated character count, never a fixed pair size.
  - Same-layer contents are merged pairwise; raw rounds and summaries are
    never mixed (the only sanctioned exception is the per-cycle merge of the
    previous final summary with newly finished top summaries, §4.4).
  - A failed LLM call leaves the group uncompressed so it is retried later;
    failure never pollutes the cache (§4.5.1).
  - Emergency collapse (§4.5.2) compresses everything in one shot when the
    cache piles up beyond the bailout threshold.

The engine operates directly on ContextCache group records, so the pipeline
state survives plugin restarts via the cache file.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from core.plugin import get_logger
from core.provider import LLMRequest
from core.agent.message import OpenAIMessage

from .context_cache import (
    FINAL_GROUP_ID,
    STATUS_COMPRESSING,
    CachedRound,
    CompressionGroup,
    ContextCache,
    msg_get,
    msg_text,
)

logger = get_logger('context_condensation.compressor', 'blue')
# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

COMPRESS_GROUP_PROMPT = """请简洁地总结以下对话轮次。
要求：
- 保留关键事实、决定和结论。
- 保留提到的人名、数字、日期。
- 保留对话的时间顺序。
- 保留任何未完成的任务或悬而未决的话题。
- 去除闲聊、冗余措辞。
- 使用第三人称。

只输出摘要文本，不要加任何前言。

对话内容：
{round_content}
"""

COMPRESS_PAIR_PROMPT = """你正在合并一段对话历史中两个连续片段的摘要。
将它们合并为一个简洁的摘要。保留：
- 关键事实、决定和结论。
- 提到的人名、数字、日期。
- 对话的时间顺序流程。
- 任何未完成的任务或悬而未决的话题。

去除闲聊、冗余措辞。
使用第三人称。

只输出合并后的摘要文本，不要加任何前言。

---
片段 A:
{a_content}

片段 B:
{b_content}
---

合并后的摘要："""

SELF_COMPRESS_PROMPT = """以下对话摘要内容过长，需要进一步压缩。
压缩时保留所有关键信息：
- 关键事实、决定和结论。
- 人名、数字、日期。
- 时间线流程。
- 未完成的任务。

只输出压缩后的摘要。

当前摘要：
{summary}
"""

FINAL_MERGE_PROMPT = """以下是一段对话历史的压缩片段（可能是先前生成的摘要，也可能是尚未压缩的原始对话内容）。
请将它们合并/压缩为一条完整、简洁、信息密度高的最终摘要。
保留：关键事实、决定、人名、数字、日期、时间线、未完成的任务。
以较新的信息为主，但不要丢失重要的旧信息。
去除冗余措辞，使用第三人称。

只输出合并后的摘要文本，不要加任何前言。

内容：
{content}
"""

EMERGENCY_PROMPT = """以下是一段过长且未经整理的对话历史。由于上下文即将溢出，
请一次性将其压缩为一条高密度摘要。
保留：关键事实、决定、人名、数字、日期、时间线、未完成的任务。
可以牺牲措辞细节，优先保证信息不丢失，且以较新的信息为主。

只输出摘要文本。

对话历史：
{content}
"""

PERSONA_SYSTEM_PROMPT = (
    "在总结时，请套用以下角色的语气、词汇风格和表述习惯进行总结：\n{persona}"
)

# Hard cap for the one-shot emergency compression input
EMERGENCY_INPUT_MAX_CHARS = 30000
# Hard cap for a single group serialization fed to the LLM
GROUP_INPUT_MAX_CHARS = 12000
# Per-call timeout: a hung LLM becomes a retriable failure instead of
# blocking the chat reply forever.
LLM_CALL_TIMEOUT = 120.0
# How much the final summary may overshoot summary_max_chars before the hard
# truncation kicks in (LLM summaries are approximate; a little slack avoids
# chopping a perfectly good summary over a handful of characters).
SUMMARY_HARD_CAP_FACTOR = 1.1


# ---------------------------------------------------------------------------
# Compression engine
# ---------------------------------------------------------------------------

class CompressionEngine:
    """Stateless-per-call engine; all durable state lives in the ContextCache."""

    def __init__(
        self,
        llm,
        persona: str = "",
        max_chars_per_group: int = 800,
        summary_max_chars: int = 1500,
    ):
        self._llm = llm
        self._persona = (persona or "").strip()
        self._max_chars_per_group = max_chars_per_group
        self._summary_max_chars = summary_max_chars

    @property
    def llm(self):
        return self._llm

    # ---- LLM call -------------------------------------------------------------

    async def _llm_chat(self, prompt: str) -> str:
        """Send a single compression prompt; raises on failure (caller handles)."""
        messages: List[Any] = []
        if self._persona:
            messages.append(OpenAIMessage(
                role="system",
                content=PERSONA_SYSTEM_PROMPT.format(persona=self._persona),
            ))
        messages.append(OpenAIMessage(role="user", content=prompt))
        request = LLMRequest(messages=messages)
        response = await asyncio.wait_for(self._llm.chat(request), timeout=LLM_CALL_TIMEOUT)
        return (response.text_response or "").strip()

    # ---- serialization ---------------------------------------------------------

    @staticmethod
    def serialize_round(round_: CachedRound) -> str:
        """Serialize a round as `[role]: content` lines (§9.6).

        Uses the preprocessed (condensed) copy when available — that is the
        whole point of preprocessing. reasoning_content is deliberately
        excluded: chain-of-thought must not leak into summaries.
        """
        lines: List[str] = []
        for m in (round_.condensed_messages or round_.messages):
            role = msg_get(m, "role", "")
            text = msg_text(msg_get(m, "content", ""))
            if not text and role == "assistant":
                tool_calls = msg_get(m, "tool_calls") or []
                names = [
                    str(tc.get("function", {}).get("name", ""))
                    for tc in tool_calls if isinstance(tc, dict)
                ]
                names = [n for n in names if n]
                if names:
                    text = f"(调用工具: {', '.join(names)})"
            if text:
                lines.append(f"[{role}]: {text}")
        return "\n".join(lines)

    def serialize_group(self, cache: ContextCache, group: CompressionGroup) -> str:
        parts: List[str] = []
        for ri in group.source_rounds:
            r = cache.get_round_by_index(ri)
            if r is not None and r.messages:
                parts.append(self.serialize_round(r))
        return "\n".join(parts)

    # ---- layer 0: grouping + compression ----------------------------------------

    def plan_layer1_groups(
        self,
        cache: ContextCache,
        rounds: List[CachedRound],
        complete_only: bool = False,
    ) -> List[CompressionGroup]:
        """Assign ungrouped rounds to layer-1 groups by accumulated chars.

        Short rounds are batched together; only an over-long round forms a
        singleton group. With ``complete_only`` (background mode) a partial
        trailing group is left ungrouped so future rounds can still batch
        into it; the injection cycle passes complete_only=False to group
        whatever remains.
        """
        ungrouped = [r for r in rounds if r.compression_group is None]
        if not ungrouped:
            return []

        created: List[CompressionGroup] = []
        current: List[CachedRound] = []
        current_chars = 0

        def flush() -> None:
            nonlocal current, current_chars
            if not current:
                return
            gid = cache.new_group_id(layer=1)
            group = CompressionGroup(
                group_id=gid,
                layer=1,
                source_rounds=[r.round_index for r in current],
            )
            cache.add_group(group)
            for r in current:
                r.compression_group = gid
                r.status = STATUS_COMPRESSING
            created.append(group)
            current = []
            current_chars = 0

        for r in ungrouped:
            if current and current_chars + r.total_chars > self._max_chars_per_group:
                flush()
            current.append(r)
            current_chars += r.total_chars
            # A full group (or a single over-long round) is sealed immediately
            if complete_only and current_chars >= self._max_chars_per_group:
                flush()
        if not complete_only:
            flush()
        return created

    async def compress_group(self, cache: ContextCache, group: CompressionGroup) -> bool:
        """Compress one layer-1 group. Failure leaves it uncompressed (§4.5.1)."""
        if group.compressed:
            return True
        content = self.serialize_group(cache, group)
        if not content:
            # Round contents unavailable (e.g. lost after restart): seal the
            # group so it does not block the pipeline forever.
            group.summary_text = ""
            group.compressed = True
            logger.debug(
                f"[context_condensation] Group {group.group_id} has no content; sealed"
            )
            return True
        try:
            text = await self._llm_chat(
                COMPRESS_GROUP_PROMPT.format(round_content=content[:GROUP_INPUT_MAX_CHARS])
            )
        except Exception as e:
            logger.warning(
                f"[context_condensation] Group {group.group_id} compression failed: {e}"
            )
            return False
        if not text:
            return False
        group.summary_text = text
        group.compressed = True
        return True

    async def compress_all_pending(
        self, cache: ContextCache, only_rounds: Optional[set] = None
    ) -> bool:
        """Compress uncompressed layer-1 groups. Returns True if all required done.

        With ``only_rounds``, only groups covering those round indices are
        required to succeed; unrelated pending groups (e.g. anchor rounds
        grouped early) are still attempted but do not block the result.
        """
        ok = True
        for g in cache.groups:
            if g.layer != 1 or g.compressed:
                continue
            success = await self.compress_group(cache, g)
            if not success and only_rounds is not None:
                if not any(ri in only_rounds for ri in g.source_rounds):
                    continue  # unrelated to this cycle; not a blocker
            if not success:
                ok = False
        return ok

    # ---- layer 2+: pairwise same-layer merging -----------------------------------

    async def _merge_texts(self, a: str, b: str) -> Optional[str]:
        try:
            # Trim oversized inputs so a bloated old summary cannot blow up the
            # merge call or the model's context window.
            a = a[:GROUP_INPUT_MAX_CHARS]
            b = b[:GROUP_INPUT_MAX_CHARS]
            text = await self._llm_chat(
                COMPRESS_PAIR_PROMPT.format(a_content=a, b_content=b)
            )
            return text or None
        except Exception as e:
            logger.warning(f"[context_condensation] Pair merge failed: {e}")
            return None

    async def _cap_summary(self, text: str) -> str:
        """Bound a final summary to ~``summary_max_chars``.

        Three layers, in order:
          1. Accept as-is when within the soft cap (summary_max_chars).
          2. Self-compress via the LLM when over the soft cap.
          3. HARD-truncate when still over the cap (self-compress failed, or
             the LLM returned something still too long) — the summary can never
             grow unbounded, even if the LLM is completely down.

        Truncation keeps the head and marks it, since summaries are head-heavy
        (recency is preserved by the head-anchored injection order).
        """
        hard_cap = int(self._summary_max_chars * SUMMARY_HARD_CAP_FACTOR)
        if len(text) <= hard_cap:
            return text
        if len(text) > self._summary_max_chars:
            try:
                compressed = await self._llm_chat(SELF_COMPRESS_PROMPT.format(summary=text))
                if compressed:
                    text = compressed
            except Exception as e:
                logger.warning(f"[context_condensation] Self-compression failed: {e}")
        if len(text) <= hard_cap:
            return text
        if len(text) > self._summary_max_chars:
            text = text[: self._summary_max_chars].rstrip() + "…"
        return text

    async def merge_available_pairs(self, cache: ContextCache) -> int:
        """Pair-merge unmerged top summaries, same layer only (§4.2).

        An odd one out stays unmerged and waits for a future partner.
        Returns the number of parent groups created.
        """
        created = 0
        while True:
            # Skip empty summaries (sealed content-less groups): merging two
            # empty texts wastes an LLM call and produces nothing.
            tops = [g for g in cache.unmerged_tops() if g.summary_text]
            if len(tops) < 2:
                break
            merged_any = False
            # Advance through ALL layers ascending: a lone leftover at one
            # layer must not block pair-ready summaries at higher layers.
            for layer in sorted({g.layer for g in tops}):
                level = [g for g in tops if g.layer == layer]
                for i in range(0, len(level) - 1, 2):
                    a, b = level[i], level[i + 1]
                    text = await self._merge_texts(a.summary_text, b.summary_text)
                    if text is None:
                        continue  # retry next pass
                    parent = CompressionGroup(
                        group_id=cache.new_group_id(layer=layer + 1),
                        layer=layer + 1,
                        source_groups=[a.group_id, b.group_id],
                        summary_text=text,
                        compressed=True,
                    )
                    cache.add_group(parent)
                    created += 1
                    merged_any = True
                if merged_any:
                    break  # tops changed; recompute before continuing
            if not merged_any:
                break
        return created

    # ---- final summary -------------------------------------------------------------

    async def build_final_summary(self, cache: ContextCache, covered_ids: set) -> str:
        """Produce the next final summary in ONE LLM call (§4.4).

        Inputs, in chronological order:
          - the previous final summary (if any),
          - every compressed top group whose traced rounds are all covered by
            this cycle (or already deleted),
          - the raw content of covered rounds that have NO compressed group yet
            (they were not pre-compressed in time — they are folded directly
            into the final summary so nothing in the freed window is lost).

        All three are joined into a single ``FINAL_MERGE_PROMPT`` call. This is
        the timing fix: the final summary no longer re-compresses stale groups
        one by one (which caused a long stall when the window was full of
        un-pre-compressed rounds). Total final cost = ONE LLM call, regardless
        of how much the background pipeline had or had not caught up.
        """
        tops = cache.unmerged_tops()
        consumable: List[CompressionGroup] = []
        for g in tops:
            if not g.summary_text:
                continue
            traced = cache.trace_summary_to_rounds(g.group_id)
            if traced and all(
                ri in covered_ids or cache.get_round_by_index(ri) is None
                for ri in traced
            ):
                consumable.append(g)

        parts: List[str] = []

        old_final = cache.find_group(FINAL_GROUP_ID)
        if old_final is not None and old_final.summary_text:
            parts.append(old_final.summary_text)
        for g in consumable:  # cache order = creation order = chronological
            parts.append(g.summary_text)

        # Fold in covered rounds that have NO compressed summary yet. They are
        # serialized inline so a pre-compression gap cannot lose them — and
        # this keeps the window freed in one pass even if the background
        # pipeline never caught up.
        for ri in sorted(covered_ids):
            r = cache.get_round_by_index(ri)
            if r is None:
                continue
            if r.compression_group is not None:
                # Its group was already handled above (compressed) or failed
                # (uncompressed group) — only fold it in if the group truly
                # produced no summary.
                g = cache.find_group(r.compression_group)
                if g is not None and g.summary_text:
                    continue
            serialized = self.serialize_round(r)
            if serialized:
                parts.append(serialized)

        if not parts:
            return ""

        # Single-shot merge: one LLM call produces the next final summary.
        content = "\n".join(parts)[:GROUP_INPUT_MAX_CHARS]
        if not content.strip():
            return ""
        try:
            text = await self._llm_chat(FINAL_MERGE_PROMPT.format(content=content))
        except Exception as e:
            logger.warning(f"[context_condensation] Final merge failed: {e}")
            text = ""
        if not text:
            # Merge produced nothing (LLM down / empty reply): fall back to the
            # concatenated parts so the cycle still injects the ready summaries
            # rather than deferring the whole window.
            text = parts[0] if len(parts) == 1 else "\n".join(parts)
            if not text:
                return ""

        # Bound the final summary length: self-compress when over the soft cap,
        # hard-truncate as a fallback so it can never grow unbounded.
        text = await self._cap_summary(text)

        if old_final is None:
            old_final = CompressionGroup(group_id=FINAL_GROUP_ID, layer=99)
            cache.add_group(old_final)
        # Covered rounds are DELETED right after the cycle, so their indices in
        # source_rounds would point at nothing — keeping them only makes the
        # cache file grow without bound. Drop the trace list entirely.
        old_final.source_rounds = []
        old_final.source_groups = []
        old_final.summary_text = text
        old_final.compressed = True
        # Consumed groups (and their subtrees) are deleted — replacement done
        for g in consumable:
            cache.delete_group_tree(g.group_id)
        return text

    # ---- emergency collapse (§4.5.2) -------------------------------------------------

    async def emergency_collapse(
        self, cache: ContextCache, growth_rounds: List[CachedRound]
    ) -> tuple:
        """One-shot compression of uncompressed growth rounds.

        Skips layering and grouping entirely; priority is getting the cache
        back under control, not summary quality.

        When the input exceeds the hard cap, the NEWEST rounds are kept and
        the oldest ones are excluded — and only the included rounds are
        reported as covered, so nothing is silently marked as summarized when
        it was not. Groups fully covered by the included rounds are consumed
        into the final summary's source chain (prevents double-merging later).

        Returns (summary_text, included_round_indices). ("", []) on failure.
        """
        serialized = [
            (r.round_index, self.serialize_round(r))
            for r in growth_rounds if r.messages
        ]
        if not serialized:
            return "", []

        # Newest-first accumulation under the cap, then restore chronology
        included: List[int] = []
        parts: List[str] = []
        total = 0
        for ri, text in reversed(serialized):
            if parts and total + len(text) > EMERGENCY_INPUT_MAX_CHARS:
                break
            parts.insert(0, text)
            included.insert(0, ri)
            total += len(text)
        content = "\n".join(parts)
        if not content.strip():
            return "", []

        try:
            text = await self._llm_chat(EMERGENCY_PROMPT.format(content=content))
        except Exception as e:
            logger.warning(f"[context_condensation] Emergency collapse failed: {e}")
            return "", []
        if not text:
            return "", []

        old_final_text = cache.final_summary_text()
        if old_final_text:
            merged = await self._merge_texts(old_final_text, text)
            if merged:
                text = merged

        # Bound the merged result so the emergency path cannot grow unbounded
        # either (self-compress + hard truncate).
        text = await self._cap_summary(text)

        old_final = cache.find_group(FINAL_GROUP_ID)
        if old_final is None:
            old_final = CompressionGroup(group_id=FINAL_GROUP_ID, layer=99)
            cache.add_group(old_final)
        # Same as the cycle path: collapsed rounds are deleted afterwards, so
        # their indices would dangle — drop the trace list to keep the cache
        # file bounded.
        old_final.source_rounds = []
        old_final.source_groups = []
        # Delete groups fully covered by this collapse (subtrees included) so
        # they are never merged into the final summary a second time. Rounds
        # no longer cached count as covered (same boundary rule as cycles).
        included_set = set(included)
        for g in list(cache.groups):
            if g.group_id == FINAL_GROUP_ID:
                continue
            traced = cache.trace_summary_to_rounds(g.group_id)
            if traced and all(
                ri in included_set or cache.get_round_by_index(ri) is None
                for ri in traced
            ):
                cache.delete_group_tree(g.group_id)
        old_final.summary_text = text
        old_final.compressed = True
        return text, included
