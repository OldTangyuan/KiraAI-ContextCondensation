"""
Context cache module.

Maintains an independent, length-unlimited store of conversation rounds,
kept in sync with the framework-provided ``req.messages`` on every request.

Key properties (per DESIGN.md):
  - Rounds are matched by content fingerprint (SHA-256 of the first user
    message's first 200 chars), never by position, so framework truncation
    and compression lag cannot corrupt tracking.
  - Every round has a status state machine:
    active -> compressing -> compressed -> archived
  - Any summary group can be traced back to its original rounds via
    source_rounds / source_groups recursion.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.plugin import get_logger

logger = get_logger('context_condensation.context_cache', 'blue')

# Round status state machine
STATUS_ACTIVE = "active"            # In req.messages, original text kept
STATUS_COMPRESSING = "compressing"  # Assigned to a compression group, still in req.messages
STATUS_COMPRESSED = "compressed"    # Covered by the injected final summary, stripped from req.messages
STATUS_ARCHIVED = "archived"        # Historical record, never in req.messages

FINAL_GROUP_ID = "final"

# Bumped whenever preprocessing logic changes (e.g. regex fixes). On load, a
# version mismatch resets is_preprocessed so rounds are re-processed with the
# current logic instead of being stuck with a stale "no change" marker.
PREPROCESS_VERSION = 2


# ---------------------------------------------------------------------------
# Message access helpers (framework memory yields dicts; agent loop may append
# OpenAIMessage objects -- handle both transparently).
# ---------------------------------------------------------------------------

def msg_get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def msg_text(content: Any) -> str:
    """Extract plain text from a message content field (str / multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def is_prompt(msg: Any) -> bool:
    """True when ``msg`` is a framework Prompt object rather than a real message.

    Prompt objects (core.prompt_manager.Prompt) carry no ``role`` field, so
    ``msg_get`` returns None for them; real messages (dict / OpenAIMessage)
    always have a non-None role. Such objects must never enter the cache:
    they are ephemeral prompt scaffolding (e.g. ``persist=False`` dynamic
    blocks), not conversation history.
    """
    return msg_get(msg, "role", None) is None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CachedRound:
    """One conversation round (user + assistant + optional tool chain).

    ``messages`` always holds the ORIGINAL framework content (used to rebuild
    req.messages for the anchor zone). Preprocessing output lives separately
    in ``condensed_messages`` and is only used as compression input, so the
    anchor zone always stays verbatim (DESIGN §3.1/§6.4).
    """
    round_index: int                              # Global, immutable once assigned
    messages: List[Any] = field(default_factory=list)
    total_chars: int = 0
    is_preprocessed: bool = False
    condensed_messages: Optional[List[Any]] = None  # Preprocessed copy (compression input only)
    compression_group: Optional[str] = None       # Layer-1 group id, None = ungrouped
    status: str = STATUS_ACTIVE
    fingerprint: str = ""

    def calc_chars(self) -> int:
        self.total_chars = sum(len(msg_text(msg_get(m, "content", ""))) for m in self.messages)
        return self.total_chars

    def first_user_text(self) -> str:
        for m in self.messages:
            if msg_get(m, "role") == "user":
                text = msg_text(msg_get(m, "content", ""))
                if text:
                    return text
        return ""

    def make_fingerprint(self) -> str:
        """Fingerprint from the first user message.

        KiraAI user messages embed a unique [message_id: XXX] marker, so
        fingerprints do not collide across rounds.
        """
        text = self.first_user_text()
        if not text:
            # Fallback for rounds without user text: hash the whole round so
            # it does not get re-added as a new round on every sync.
            text = "\n".join(msg_text(msg_get(m, "content", "")) for m in self.messages)
        if text:
            self.fingerprint = hashlib.sha256(text[:200].encode("utf-8")).hexdigest()[:16]
        else:
            self.fingerprint = ""
        return self.fingerprint


@dataclass
class CompressionGroup:
    """A compression group record (layer 1 = round group, 2+ = summary merge)."""
    group_id: str
    layer: int                                     # 1 = round group, 2+ = merged summaries
    source_rounds: List[int] = field(default_factory=list)
    source_groups: List[str] = field(default_factory=list)
    summary_text: str = ""
    compressed: bool = False


# ---------------------------------------------------------------------------
# Context cache
# ---------------------------------------------------------------------------

class ContextCache:
    """Per-session round cache with fingerprint matching and persistence."""

    def __init__(self, data_dir: Path, anchor_size: int = 5):
        self._data_dir = data_dir
        self._anchor_size = anchor_size
        self._rounds: List[CachedRound] = []
        self._groups: List[CompressionGroup] = []
        self._next_round_index: int = 0
        self._next_group_index: int = 0
        self._fingerprint_index: Dict[str, int] = {}
        # Guards concurrent save() calls (they run in worker threads via
        # asyncio.to_thread and could otherwise corrupt the shared .tmp file)
        self._save_lock = threading.Lock()

    # ---- persistence --------------------------------------------------------

    def _cache_path(self, sid: str) -> Path:
        safe_sid = sid.replace(":", "_").replace("/", "_")
        return self._data_dir / "caches" / f"{safe_sid}.json"

    @staticmethod
    def _serialize_messages(messages: List[Any]) -> List[dict]:
        out: List[dict] = []
        for m in messages:
            if isinstance(m, dict):
                out.append(m)
            elif hasattr(m, "to_dict"):
                out.append(m.to_dict())
            else:
                out.append({"role": msg_get(m, "role", "user"),
                            "content": msg_text(msg_get(m, "content", ""))})
        return out

    def save(self, sid: str) -> None:
        path = self._cache_path(sid)
        with self._save_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "anchor_size": self._anchor_size,
                    "preprocess_version": PREPROCESS_VERSION,
                    "next_round_index": self._next_round_index,
                    "next_group_index": self._next_group_index,
                    "rounds": [
                        {
                            "i": r.round_index,
                            "c": r.total_chars,
                            "p": r.is_preprocessed,
                            "g": r.compression_group,
                            "s": r.status,
                            "fp": r.fingerprint,
                            # Persist message content ONLY for rounds that still
                            # need it (not yet compressed). Once a round is covered
                            # by the final summary its content is redundant.
                            **(
                                {"m": self._serialize_messages(r.messages)}
                                if r.status in (STATUS_ACTIVE, STATUS_COMPRESSING) and r.messages
                                else {}
                            ),
                            **(
                                {"cm": self._serialize_messages(r.condensed_messages)}
                                if r.condensed_messages
                                else {}
                            ),
                        }
                        for r in self._rounds
                    ],
                    "groups": [
                        {
                            "id": g.group_id,
                            "l": g.layer,
                            "s": g.source_rounds,
                            "sg": g.source_groups,
                            "t": g.summary_text,
                            "c": g.compressed,
                        }
                        for g in self._groups
                    ],
                }
                # Atomic write: never leave a half-written JSON behind on crash
                tmp_path = path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp_path.replace(path)
            except Exception:
                # Broad guard: serialization errors (e.g. an unserializable
                # message object) must never crash the event loop.
                logger.warning(
                    f"[context_condensation] Failed to save cache for {sid}"
                )

    def load(self, sid: str) -> bool:
        path = self._cache_path(sid)
        if not path.exists():
            return False
        # Read under the same lock save() uses: save runs in a worker thread
        # (asyncio.to_thread) and could otherwise be mid-write while we read,
        # surfacing a half-written file.
        with self._save_lock:
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning(
                    f"[context_condensation] Cannot read cache for {sid}; starting fresh"
                )
                return False
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Corrupt/empty file: refuse to crash, rebuild from scratch.
            logger.warning(
                f"[context_condensation] Corrupt cache for {sid}; starting fresh"
            )
            return False

        try:
            self._deserialize(data)
        except Exception:
            logger.warning(
                f"[context_condensation] Invalid cache content for {sid}; starting fresh"
            )
            self.clear()
            return False
        return True

    def _deserialize(self, data: dict) -> None:
        """Populate cache state from a parsed cache dict (shared by load)."""
        # anchor_size comes from the live config, not the file: the user may
        # have changed it in WebUI since the cache was written.
        self._next_round_index = data.get("next_round_index", 0)
        preprocess_stale = data.get("preprocess_version", 1) != PREPROCESS_VERSION
        self._next_group_index = data.get("next_group_index", 0)

        self._rounds = []
        self._fingerprint_index.clear()
        for rd in data.get("rounds", []):
            if rd.get("s") in (STATUS_COMPRESSED, STATUS_ARCHIVED):
                # Migration: covered rounds carry no content, only metadata —
                # drop them. Traceability lives in the final group's flat
                # source_rounds list.
                continue
            r = CachedRound(
                round_index=rd["i"],
                total_chars=rd.get("c", 0),
                is_preprocessed=rd.get("p", False),
                compression_group=rd.get("g"),
                status=rd.get("s", STATUS_ACTIVE),
                fingerprint=rd.get("fp", ""),
                # Uncompressed rounds keep their messages across restarts so
                # framework truncation can never destroy un-summarized content.
                messages=list(rd.get("m", [])),
                condensed_messages=(list(rd["cm"]) if rd.get("cm") else None),
            )
            # Hygiene: never keep an identical duplicate of messages in
            # condensed_messages (preprocessing that changed nothing)
            if r.condensed_messages == r.messages:
                r.condensed_messages = None
            if preprocess_stale:
                # Preprocessing logic changed since this cache was written:
                # re-process with the current version (fixes e.g. image
                # descriptions stuck with a stale no-change marker).
                r.is_preprocessed = False
                r.condensed_messages = None
            self._rounds.append(r)
            if r.fingerprint:
                self._fingerprint_index[r.fingerprint] = len(self._rounds) - 1
            if r.round_index >= self._next_round_index:
                self._next_round_index = r.round_index + 1

        self._groups = []
        max_gid = -1
        for gd in data.get("groups", []):
            self._groups.append(
                CompressionGroup(
                    group_id=gd["id"],
                    layer=gd.get("l", 1),
                    source_rounds=gd.get("s", []),
                    source_groups=gd.get("sg", []),
                    summary_text=gd.get("t", ""),
                    compressed=gd.get("c", False),
                )
            )
            gid = gd["id"]
            if gid and gid[0] in "gp" and gid[1:].isdigit():
                max_gid = max(max_gid, int(gid[1:]))
        self._next_group_index = max(self._next_group_index, max_gid + 1)

        # Migration: the final group's source_rounds/source_groups used to
        # accumulate across every cycle (covering already-deleted rounds). Those
        # indices point at nothing now — drop them so old cache files shrink.
        final = self.find_group(FINAL_GROUP_ID)
        if final is not None:
            final.source_rounds = []
            final.source_groups = []

    def clear(self) -> None:
        """Wipe all state (used on severe context mismatch with inject_on_mismatch=false)."""
        self._rounds.clear()
        self._groups.clear()
        self._fingerprint_index.clear()
        self._next_round_index = 0
        self._next_group_index = 0

    # ---- sync with req.messages ---------------------------------------------

    def sync(self, messages: List[Any]) -> Tuple[List[CachedRound], List[CachedRound]]:
        """Sync the cache with the current req.messages.

        Returns (newly_added_rounds, present_rounds_in_req_order).
        """
        # Opportunistic hygiene: sealed empty groups (no content, no summary)
        # are dead records that would otherwise accumulate forever.
        self.gc_orphan_groups()
        candidates = self._parse_into_rounds(messages)
        added: List[CachedRound] = []
        present: List[CachedRound] = []
        seen_indices: set = set()

        for candidate in candidates:
            candidate.make_fingerprint()
            existing: Optional[CachedRound] = None
            if candidate.fingerprint:
                idx = self._fingerprint_index.get(candidate.fingerprint)
                if idx is not None:
                    existing = self._rounds[idx]

            if existing is not None:
                if existing.round_index in seen_indices:
                    # Same round appearing TWICE in one context (e.g. duplicated
                    # chunk in framework memory): count it once so anchor/growth
                    # math and message rebuilds never emit it twice.
                    continue
                seen_indices.add(existing.round_index)
                # Always refresh the ORIGINAL content from the framework (it may
                # provide updated/full messages). Preprocessing output is stored
                # separately in condensed_messages and is unaffected.
                existing.messages = candidate.messages
                if not existing.is_preprocessed:
                    existing.total_chars = candidate.total_chars
                present.append(existing)
            else:
                candidate.round_index = self._next_round_index
                self._next_round_index += 1
                candidate.calc_chars()
                self._rounds.append(candidate)
                if candidate.fingerprint:
                    self._fingerprint_index[candidate.fingerprint] = len(self._rounds) - 1
                seen_indices.add(candidate.round_index)
                added.append(candidate)
                present.append(candidate)

        return added, present

    @staticmethod
    def _parse_into_rounds(messages: List[Any]) -> List[CachedRound]:
        """Split a flat message list into rounds.

        A round starts at a user message and includes everything up to the next
        user message, so tool_call + tool_result chains stay atomic (§9.6).
        Messages before the first user message are ignored.
        """
        rounds: List[CachedRound] = []
        current: Optional[CachedRound] = None
        for msg in messages:
            # Ephemeral Prompt scaffolding (e.g. persist=False dynamic blocks)
            # is not conversation history: never cache, fingerprint, or rebuild
            # from it. Real messages always carry a role.
            if is_prompt(msg):
                continue
            if msg_get(msg, "role") == "user":
                if current is not None:
                    current.calc_chars()
                    rounds.append(current)
                current = CachedRound(round_index=-1)
            if current is not None:
                current.messages.append(msg)
        if current is not None:
            current.calc_chars()
            rounds.append(current)
        return rounds

    # ---- mismatch detection (§9.1) -------------------------------------------

    def detect_mismatch(self, present: List[CachedRound]) -> bool:
        """Severe mismatch: history was cleared/reset while the cache is far ahead.

        Two conditions must BOTH hold:
          1. Most tracked rounds vanished from req.messages;
          2. req.messages itself is nearly empty (<= 2 rounds).
        Condition 2 distinguishes a user clearing history from normal framework
        truncation (after truncation req.messages is still near max_memory_length,
        so no false positive).
        """
        tracked = [r for r in self._rounds if r.status in (STATUS_ACTIVE, STATUS_COMPRESSING)]
        if len(tracked) < 4 or len(present) > 2:
            return False
        present_ids = {r.round_index for r in present}
        present_tracked = sum(1 for r in tracked if r.round_index in present_ids)
        return present_tracked * 2 < len(tracked)

    def archive_absent(self, present: List[CachedRound]) -> List[CachedRound]:
        """Handle rounds that no longer appear in req.messages.

        Uncompressed rounds that still have their message content are KEPT
        tracked and returned, so the compression pipeline can still absorb
        them into the summary instead of losing the information. Content-less
        rounds are DELETED outright — once covered by the summary, raw content
        is never kept around.

        Returns the list of vanished-but-still-compressible rounds.
        """
        present_ids = {r.round_index for r in present}
        vanished_compressible: List[CachedRound] = []
        deleted = False
        for r in list(self._rounds):
            if r.round_index in present_ids:
                continue
            if r.status in (STATUS_COMPRESSED, STATUS_ARCHIVED):
                # Legacy states (pre-deletion-era caches): drop entirely
                self._rounds.remove(r)
                deleted = True
            elif r.status in (STATUS_ACTIVE, STATUS_COMPRESSING):
                if r.messages:
                    vanished_compressible.append(r)
                else:
                    logger.warning(
                        f"[context_condensation] Round {r.round_index} vanished from "
                        f"context with no cached content; dropping"
                    )
                    self._rounds.remove(r)
                    deleted = True
        if deleted:
            self._fingerprint_index = {
                r.fingerprint: i for i, r in enumerate(self._rounds) if r.fingerprint
            }
        return vanished_compressible

    def delete_rounds(self, round_indices: List[int]) -> None:
        """Fully remove rounds from the cache (called once the final summary
        covers them — pre-compression content is discarded, not archived)."""
        doomed = set(round_indices)
        if not doomed:
            return
        self._rounds = [r for r in self._rounds if r.round_index not in doomed]
        self._fingerprint_index = {
            r.fingerprint: i for i, r in enumerate(self._rounds) if r.fingerprint
        }

    def delete_group_tree(self, group_id: str) -> None:
        """Delete a group and its whole source_group subtree (called once the
        final summary has absorbed them; the final summary keeps a flat list
        of covered round indices for traceability)."""
        doomed: set = set()
        stack = [group_id]
        while stack:
            gid = stack.pop()
            if gid in doomed or gid == FINAL_GROUP_ID:
                continue
            doomed.add(gid)
            g = self.find_group(gid)
            if g is not None:
                stack.extend(g.source_groups)
        self._groups = [g for g in self._groups if g.group_id not in doomed]

    # ---- round queries ---------------------------------------------------------

    def tracked_rounds(self) -> List[CachedRound]:
        """All uncompressed rounds (active/compressing), chronological."""
        return [r for r in self._rounds if r.status in (STATUS_ACTIVE, STATUS_COMPRESSING)]

    @staticmethod
    def uncompressed_of(present: List[CachedRound]) -> List[CachedRound]:
        return [r for r in present if r.status in (STATUS_ACTIVE, STATUS_COMPRESSING)]

    def growth_of(self, uncompressed: List[CachedRound]) -> List[CachedRound]:
        """Compression candidates: uncompressed rounds outside the anchor tail."""
        if len(uncompressed) <= self._anchor_size:
            return []
        return uncompressed[:-self._anchor_size]

    def anchor_of(self, uncompressed: List[CachedRound]) -> List[CachedRound]:
        """Anchor zone: the most recent N uncompressed rounds, kept verbatim."""
        return uncompressed[-self._anchor_size:] if uncompressed else []

    def get_round_by_index(self, round_index: int) -> Optional[CachedRound]:
        for r in self._rounds:
            if r.round_index == round_index:
                return r
        return None

    def mark_rounds_status(self, round_indices: List[int], status: str) -> None:
        for ri in round_indices:
            r = self.get_round_by_index(ri)
            if r is not None:
                r.status = status

    @property
    def anchor_size(self) -> int:
        return self._anchor_size

    @property
    def rounds(self) -> List[CachedRound]:
        return list(self._rounds)

    @property
    def total_rounds(self) -> int:
        return len(self._rounds)

    # ---- group management ------------------------------------------------------

    @property
    def groups(self) -> List[CompressionGroup]:
        return list(self._groups)

    def find_group(self, group_id: str) -> Optional[CompressionGroup]:
        for g in self._groups:
            if g.group_id == group_id:
                return g
        return None

    def new_group_id(self, layer: int) -> str:
        prefix = "g" if layer <= 1 else "p"
        gid = f"{prefix}{self._next_group_index}"
        self._next_group_index += 1
        return gid

    def add_group(self, group: CompressionGroup) -> None:
        self._groups.append(group)

    def drop_non_final_groups(self) -> None:
        """Drop every group record except the final summary.

        Used by the resurrect path: covered rounds are being restored to the
        live context, so their old group summaries must not be merged into a
        future final summary again (that would duplicate information).
        """
        self._groups = [g for g in self._groups if g.group_id == FINAL_GROUP_ID]

    def final_summary_text(self) -> str:
        final = self.find_group(FINAL_GROUP_ID)
        if final is not None and final.compressed:
            return final.summary_text
        return ""

    def adopt_final_summary(self, text: str) -> None:
        """Adopt a summary chunk found in framework memory (e.g. written by a
        previous run whose cache file was lost) as the final summary."""
        final = self.find_group(FINAL_GROUP_ID)
        if final is None:
            final = CompressionGroup(group_id=FINAL_GROUP_ID, layer=99)
            self.add_group(final)
        final.summary_text = text
        final.compressed = True

    def unmerged_tops(self) -> List[CompressionGroup]:
        """Compressed groups not yet merged into a parent (excluding the final summary)."""
        children = {sg for g in self._groups for sg in g.source_groups}
        return [
            g for g in self._groups
            if g.compressed and g.group_id != FINAL_GROUP_ID and g.group_id not in children
        ]

    def trace_summary_to_rounds(self, group_id: str) -> List[int]:
        """Recursively trace any summary group back to its original round indices.

        Traverses BOTH source_rounds and source_groups (a group may have both,
        e.g. the final summary after an emergency merge). A visited set guards
        against accidental reference cycles.
        """
        result: List[int] = []
        visited: set = set()

        def walk(gid: str) -> None:
            if gid in visited:
                return
            visited.add(gid)
            group = self.find_group(gid)
            if group is None:
                return
            result.extend(group.source_rounds)
            for sg_id in group.source_groups:
                walk(sg_id)

        walk(group_id)
        return result

    def gc_orphan_groups(self) -> int:
        """Drop dead group records that would otherwise accumulate forever.

        The real growth vector is groups that were sealed with NO content:
        ``compress_group`` marks a content-less round group ``compressed`` with
        an empty ``summary_text``, and such a group is never merged into a
        parent (pairwise merges skip empty summaries) nor deleted — it lingers
        as a permanent empty top. A sealed empty group is dropped when no other
        group references it and it is not the final summary. Returns the number
        of groups dropped.
        """
        children = {sg for g in self._groups for sg in g.source_groups}
        keep = [
            g for g in self._groups
            if g.group_id == FINAL_GROUP_ID
            or g.group_id in children
            or g.summary_text  # live top (pending merge) or compressed with content
            or not g.compressed  # still-pending layer-1 group: keep for retry
        ]
        dropped = len(self._groups) - len(keep)
        if dropped:
            self._groups = keep
        return dropped
