import unittest

from backend.toolcore.request_normalizer import normalize_chat_request
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcore.types import ToolDefinition


class ToolCatalogTests(unittest.TestCase):
    def test_request_declared_tools_are_the_only_allowed_tools(self) -> None:
        catalog = ToolCatalog(
            [
                ToolDefinition(name="Read", client_name="Read", model_name="Read"),
                ToolDefinition(name="Write", client_name="Write", model_name="Write"),
            ]
        )

        self.assertTrue(catalog.is_declared("Read"))
        self.assertFalse(catalog.is_declared("Bash"))

    def test_aliases_only_resolve_within_current_request(self) -> None:
        catalog = ToolCatalog(
            [ToolDefinition(name="Bash", client_name="exec", model_name="Bash", aliases=("process",))]
        )

        self.assertEqual(catalog.get_canonical_name("exec"), "Bash")
        self.assertEqual(catalog.get_canonical_name("process"), "Bash")
        self.assertIsNone(catalog.get_canonical_name("Read"))

    def test_model_name_can_differ_from_client_name(self) -> None:
        catalog = ToolCatalog(
            [ToolDefinition(name="Bash", client_name="exec", model_name="Bash")]
        )

        self.assertEqual(catalog.get_client_name("Bash"), "exec")
        self.assertEqual(catalog.get_model_name("exec"), "Bash")

    def test_undeclared_tool_choice_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            normalize_chat_request(
                {
                    "messages": [{"role": "user", "content": "do something"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "Read", "parameters": {}},
                        }
                    ],
                    "tool_choice": {"type": "function", "function": {"name": "Write"}},
                }
            )

    def test_normalizer_attaches_catalog_to_request(self) -> None:
        request = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "read a file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "Read", "parameters": {}},
                    }
                ],
            }
        )

        self.assertTrue(hasattr(request, "tool_catalog"))
        self.assertIsNotNone(request.tool_catalog)
        catalog = request.tool_catalog
        assert catalog is not None
        self.assertEqual(catalog.get_canonical_name("Read"), "Read")

    def test_request_normalizer_assigns_dynamic_model_names(self) -> None:
        request = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "use tools"}],
                "tools": [
                    {"type": "function", "function": {"name": "exec", "parameters": {}}},
                    {"type": "function", "function": {"name": "image_generate", "parameters": {}}},
                ],
            }
        )

        self.assertEqual([tool.client_name for tool in request.tools], ["exec", "image_generate"])
        self.assertEqual([tool.model_name for tool in request.tools], ["bridge-0", "bridge-1"])
        catalog = request.tool_catalog
        assert catalog is not None
        self.assertEqual(catalog.get_canonical_name("bridge-0"), "exec")
        self.assertEqual(catalog.get_client_name("exec"), "exec")
        self.assertEqual(catalog.get_model_name("image_generate"), "bridge-1")


if __name__ == "__main__":
    unittest.main()
