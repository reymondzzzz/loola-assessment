import json

import tiktoken

ENCODING = tiktoken.get_encoding("o200k_base")
MESSAGE_OVERHEAD = 4


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text, disallowed_special=()))


def count_messages(messages: list[dict], tools: list[dict]) -> int:
    total = count_tokens(json.dumps(tools)) if tools else 0
    for message in messages:
        total += MESSAGE_OVERHEAD + count_tokens(message.get("content") or "")
        if message.get("tool_calls"):
            total += count_tokens(json.dumps(message["tool_calls"]))
    return total
