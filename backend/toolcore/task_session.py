from __future__ import annotations

import json
from typing import Any

from backend.adapter.standard_request import StandardRequest


def _client_tool_name(name: str, request: StandardRequest) -> str:
    if request.tool_catalog is None:
        return name
    canonical = request.tool_catalog.get_canonical_name(name)
    if canonical is None:
        return name
    return request.tool_catalog.get_client_name(canonical)


def build_openai_assistant_history_message(*, execution, request: StandardRequest, directive) -> dict[str, Any]:
    if directive.stop_reason == 'tool_use':
        tool_calls = [
            {
                'id': block['id'],
                'type': 'function',
                'function': {
                    'name': _client_tool_name(str(block['name']), request),
                    'arguments': json.dumps(block.get('input', {}), ensure_ascii=False),
                },
            }
            for block in directive.tool_blocks
            if block.get('type') == 'tool_use'
        ]
        return {'role': 'assistant', 'content': "", 'tool_calls': tool_calls}
    return {'role': 'assistant', 'content': execution.state.answer_text}
