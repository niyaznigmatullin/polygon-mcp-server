import base64
import unittest
from unittest.mock import Mock, patch

from polygon_api import File

from polygon_mcp import server


class StatementResourcesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _invoke(self, tool, *args, **kwargs):
        with patch.object(server, "_get_client", return_value=self.client):
            return tool.fn(*args, **kwargs)

    def test_lists_resources_as_file_objects(self) -> None:
        self.client.problem_statement_resources.return_value = [
            File(
                name="olymp.sty",
                modification_time_seconds=1700000000,
                length=4096,
                source_type=None,
                resource_advanced_properties=None,
            )
        ]

        result = self._invoke(server.problem_statement_resources, 42, pin="1234")

        self.client.problem_statement_resources.assert_called_once_with(42, pin="1234")
        self.assertEqual(
            result,
            [
                {
                    "name": "olymp.sty",
                    "modification_time_seconds": 1700000000,
                    "length": 4096,
                    "source_type": None,
                    "resource_advanced_properties": None,
                }
            ],
        )

    def test_view_requests_bytes_and_returns_text(self) -> None:
        self.client.problem_view_statement_resource.return_value = b"line 1\nline 2\n"

        result = self._invoke(
            server.problem_view_statement_resource, 42, "olymp.sty", pin="1234"
        )

        self.client.problem_view_statement_resource.assert_called_once_with(
            42, "olymp.sty", binary=True, pin="1234"
        )
        self.assertEqual(result, {"data": "line 1\nline 2\n", "encoding": "utf-8"})

    def test_view_paginates_by_line(self) -> None:
        self.client.problem_view_statement_resource.return_value = b"a\nb\nc\n"

        result = self._invoke(
            server.problem_view_statement_resource,
            42,
            "example.01",
            start_line=2,
            line_count=1,
        )

        self.assertEqual(result["data"], "b\n")
        self.assertEqual(result["next_start_line"], 3)

    def test_view_returns_binary_resource_as_base64(self) -> None:
        image = b"\x89PNG\r\n\x1a\n\x00\x01"
        self.client.problem_view_statement_resource.return_value = image

        result = self._invoke(
            server.problem_view_statement_resource, 42, "pic.png", binary=True
        )

        self.assertEqual(
            result,
            {"data": base64.b64encode(image).decode("ascii"), "encoding": "base64"},
        )

    def test_view_surfaces_polygon_failure_payload(self) -> None:
        self.client.problem_view_statement_resource.return_value = (
            b'{"status":"FAILED","comment":"name: Resource not found"}'
        )

        with self.assertRaises(RuntimeError) as ctx:
            self._invoke(server.problem_view_statement_resource, 42, "missing.png")

        self.assertEqual(
            str(ctx.exception), "Polygon API error: name: Resource not found"
        )


if __name__ == "__main__":
    unittest.main()
