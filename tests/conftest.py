"""pytest 全局夹具。

进程级状态（如文档 parse 熔断）必须逐测试清零，否则「先撞 WAF 的用例」会把
熔断打开，再让「断言上传链路」的用例凭空降级 —— 这类顺序相关的假失败排查成本
极高，统一在这里复位。
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_document_parse_breaker():
    """每个测试前后都把文档 parse 熔断清零。"""
    from backend.services.document_parse_breaker import reset_document_parse_breaker

    reset_document_parse_breaker()
    try:
        yield
    finally:
        reset_document_parse_breaker()
