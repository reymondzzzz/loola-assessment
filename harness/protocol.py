def protocol_errors(messages: list[dict]) -> list[str]:
    errors, open_calls = [], set()
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            if message.get("tool_call_id") not in open_calls:
                errors.append(f"orphan tool result at {index}")
            open_calls.discard(message.get("tool_call_id"))
            continue
        if open_calls:
            errors.append(f"unresolved tool calls before {index}: {sorted(open_calls)}")
            open_calls = set()
        if message["role"] == "assistant":
            open_calls = {call["id"] for call in message.get("tool_calls") or []}
    if open_calls:
        errors.append(f"unresolved tool calls at end: {sorted(open_calls)}")
    return errors
