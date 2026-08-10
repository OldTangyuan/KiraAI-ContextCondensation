"""
Preprocessing pipeline (DESIGN §6).

Shrinks oversized tool results and image descriptions inside the CACHE ONLY;
the original req.messages are never touched.

  - Tool results: JSON content is parsed; when the total text size exceeds
    the threshold, all text is summarized by the LLM and the longest text
    field is replaced with the summary. A `"_condensed": true` marker field
    prevents re-processing and tells the pipeline the content is pre-shrunk.
  - Image descriptions: `[Image: ...]` / `[图片描述: ...]` blocks longer than
    the threshold are individually summarized and suffixed with （已压缩）.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, List, Optional

from core.plugin import get_logger
from core.provider import LLMRequest
from core.agent.message import OpenAIMessage

from .context_cache import msg_get, msg_text

logger = get_logger('context_condensation.preprocessor', 'blue')

SUMMARIZE_PROMPT = """请简洁地总结以下内容。
保留所有关键事实、数字、名称和结论。
去除冗余格式、套话和无关内容。

只输出总结后的文本，不要加任何解释。

待总结的内容：
{content}
"""

# Matches [Image: ...], [Image ...] (space, no colon — seen in production
# data) and [图片描述: ...] blocks (DESIGN §6.3)
_IMAGE_DESC_PATTERN = re.compile(r"\[(?:Image|图片描述)[:：]?\s*(.+?)\]", re.DOTALL)

_CONDENSED_SUFFIX = "（已压缩）"
# Cap LLM input size for a single preprocessing call
_PREPROCESS_INPUT_MAX = 8000
# Per-call timeout (same rationale as the compressor)
_LLM_CALL_TIMEOUT = 120.0


async def _summarize(content: str, llm) -> Optional[str]:
    prompt = SUMMARIZE_PROMPT.format(content=content[:_PREPROCESS_INPUT_MAX])
    try:
        request = LLMRequest(messages=[OpenAIMessage(role="user", content=prompt)])
        response = await asyncio.wait_for(llm.chat(request), timeout=_LLM_CALL_TIMEOUT)
        summary = (response.text_response or "").strip()
        return summary or None
    except Exception as e:
        # Preprocessing is best-effort; a failing LLM must not spam the log.
        logger.debug(f"[context_condensation] Preprocess summarization failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tool result preprocessing (§6.2)
# ---------------------------------------------------------------------------

def _parse_json_stream(content: str) -> Optional[List[dict]]:
    """Tolerantly parse tool-result JSON.

    Real-world tool results are often MULTIPLE JSON objects concatenated with
    no separator (e.g. search hits: `{"url":...}{"url":...}`). Try a strict
    parse first, then a raw_decode stream. Returns a list of parsed objects,
    or None if the content is not JSON at all.
    """
    try:
        data = json.loads(content)
        return [data]
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    objects: List[dict] = []
    idx = 0
    length = len(content)
    while idx < length:
        while idx < length and content[idx] not in "{[":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            return None
        objects.append(obj)
        idx = end
    return objects or None


async def preprocess_tool_result(content: str, max_chars: int, llm) -> str:
    """Summarize an over-long tool result. Returns the (possibly new) content."""
    if len(content) <= max_chars:
        return content

    objects = _parse_json_stream(content)

    # Single JSON object: field-level replacement (§6.2)
    if objects is not None and len(objects) == 1 and isinstance(objects[0], dict):
        data = objects[0]
        if data.get("_condensed"):
            return content
        str_fields = {k: v for k, v in data.items() if isinstance(v, str)}
        total = sum(len(v) for v in str_fields.values())
        if total <= max_chars:
            return content
        joined = "\n".join(str_fields.values())
        summary = await _summarize(joined, llm)
        if summary and len(summary) < total:
            # Replace the longest text field with the summary (§6.2)
            longest_key = max(str_fields, key=lambda k: len(str_fields[k]))
            data[longest_key] = summary
            data["_condensed"] = True
            logger.debug(
                f"[context_condensation] Tool result condensed: {total}->{len(summary)} chars"
            )
            return json.dumps(data, ensure_ascii=False)
        return content

    # Concatenated multi-object result (e.g. search hits): summarize all text
    # fields into one summary while preserving source URLs as compact metadata.
    if objects is not None and len(objects) > 1:
        if all(isinstance(o, dict) and o.get("_condensed") for o in objects):
            return content
        texts: List[str] = []
        urls: List[str] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if isinstance(v, str):
                    if k == "url":
                        urls.append(v)
                    else:
                        texts.append(v)
        total = sum(len(t) for t in texts)
        if total <= max_chars:
            return content
        summary = await _summarize("\n".join(texts), llm)
        if summary and len(summary) < total:
            compact = {"summary": summary, "_condensed": True}
            if urls:
                compact["sources"] = urls[:10]
            logger.debug(
                f"[context_condensation] Tool result stream condensed: "
                f"{total}->{len(summary)} chars ({len(objects)} objects)"
            )
            return json.dumps(compact, ensure_ascii=False)
        return content

    # Non-JSON long content: plain-text summarization
    summary = await _summarize(content, llm)
    if summary and len(summary) < len(content):
        logger.debug(
            f"[context_condensation] Tool result condensed: {len(content)}->{len(summary)} chars"
        )
        return summary + _CONDENSED_SUFFIX
    return content


# ---------------------------------------------------------------------------
# Image description preprocessing (§6.3)
# ---------------------------------------------------------------------------

async def _preprocess_image_descriptions(content: str, max_chars: int, llm) -> str:
    matches = list(_IMAGE_DESC_PATTERN.finditer(content))
    if not matches:
        return content
    result = content
    for match in matches:
        desc = match.group(1)
        if len(desc) <= max_chars or desc.endswith(_CONDENSED_SUFFIX):
            continue
        summary = await _summarize(desc, llm)
        if summary and len(summary) < len(desc):
            new_block = match.group(0).replace(desc, summary + _CONDENSED_SUFFIX)
            result = result.replace(match.group(0), new_block, 1)
            logger.debug(
                f"[context_condensation] Image description condensed: "
                f"{len(desc)}->{len(summary)} chars"
            )
    return result


# ---------------------------------------------------------------------------
# Round-level preprocessing
# ---------------------------------------------------------------------------

async def preprocess_round(
    messages: List[Any],
    tool_max_chars: int,
    llm,
) -> List[Any]:
    """Return a preprocessed COPY of a round's messages (originals untouched).

    - role=tool messages: JSON-aware summarization with `_condensed` marker.
    - role=user messages: over-long image descriptions are summarized in place.
    """
    result: List[Any] = []
    for msg in messages:
        role = msg_get(msg, "role", "")
        content = msg_get(msg, "content", "")
        new_content = content

        if role == "tool" and isinstance(content, str) and len(content) > tool_max_chars:
            new_content = await preprocess_tool_result(content, tool_max_chars, llm)
        elif role == "user":
            text = msg_text(content)
            if text and ("[Image" in text or "[图片描述" in text):
                new_text = await _preprocess_image_descriptions(text, tool_max_chars, llm)
                if new_text != text and isinstance(content, str):
                    new_content = new_text

        if new_content is content:
            result.append(msg)
        elif isinstance(msg, dict):
            new_msg = dict(msg)
            new_msg["content"] = new_content
            result.append(new_msg)
        else:
            # OpenAIMessage-like object: build a plain dict copy
            result.append({
                "role": role,
                "content": new_content,
                **({"tool_calls": msg_get(msg, "tool_calls")} if msg_get(msg, "tool_calls") else {}),
                **({"tool_call_id": msg_get(msg, "tool_call_id")} if msg_get(msg, "tool_call_id") else {}),
                **({"name": msg_get(msg, "name")} if msg_get(msg, "name") else {}),
            })
    return result
