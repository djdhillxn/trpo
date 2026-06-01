from __future__ import annotations

from typing import Any


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Some chat datasets store multimodal-ish content lists. Keep text parts.
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(value)


def normalize_messages(context: Any) -> list[dict[str, str]]:
    """Normalize HelpSteer-style context objects into OpenAI/Qwen chat messages.

    Handles the common shapes:
    - list[{"role": ..., "content": ...}]
    - list[{"from": ..., "value": ...}]
    - string prompt
    - dict with context/messages/conversation key
    """
    if isinstance(context, dict):
        for key in ("messages", "context", "conversation", "conversations", "turns"):
            if key in context:
                return normalize_messages(context[key])
        # Last-resort single user prompt from dict content-ish fields.
        text = context.get("content") or context.get("prompt") or context.get("value") or str(context)
        return [{"role": "user", "content": _stringify_content(text)}]

    if isinstance(context, str):
        return [{"role": "user", "content": context.strip()}]

    if not isinstance(context, list):
        return [{"role": "user", "content": _stringify_content(context).strip()}]

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(context):
        if isinstance(item, str):
            role = "user" if idx % 2 == 0 else "assistant"
            content = item
        elif isinstance(item, dict):
            role_raw = item.get("role") or item.get("from") or item.get("speaker") or item.get("author")
            role = ROLE_ALIASES.get(str(role_raw).lower(), "user" if idx % 2 == 0 else "assistant")
            content = item.get("content")
            if content is None:
                content = item.get("value") or item.get("text") or item.get("message")
        else:
            role = "user" if idx % 2 == 0 else "assistant"
            content = item
        content_str = _stringify_content(content).strip()
        if content_str:
            messages.append({"role": role, "content": content_str})

    # Qwen chat templates are happiest when the last message before generation is
    # not an empty assistant response. If the dataset context already includes the
    # assistant answer being compared, data.py removes it before calling us.
    return messages or [{"role": "user", "content": ""}]


def strip_trailing_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove a final assistant turn if present.

    Preference datasets sometimes put the candidate answer in the context. For
    reward modeling and PPO prompting, we want context ending before the compared
    assistant response.
    """
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def render_prompt(tokenizer: Any, context: Any, *, add_generation_prompt: bool = True) -> str:
    messages = normalize_messages(context)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback format for tokenizers without a chat template.
    chunks: list[str] = []
    for msg in messages:
        role = msg["role"].capitalize()
        chunks.append(f"{role}: {msg['content']}")
    if add_generation_prompt:
        chunks.append("Assistant:")
    return "\n".join(chunks).strip() + "\n"


def render_prompt_with_response(tokenizer: Any, context: Any, response: str) -> str:
    prompt = render_prompt(tokenizer, context, add_generation_prompt=True)
    eos = tokenizer.eos_token or ""
    return prompt + response.strip() + eos
