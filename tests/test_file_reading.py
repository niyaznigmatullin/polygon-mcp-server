import base64
import unittest
from unittest.mock import Mock, patch

from polygon_mcp import server
from polygon_mcp.server import (
    MAX_FILE_CHARS,
    MAX_FILE_LINES,
    MAX_FILE_SEARCH_MATCHES,
    _file_content_response,
    _search_file_content,
)


class BinaryApiCallsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _assert_binary_call(self, invoke) -> None:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=b"hello\n") as call_polygon,
        ):
            invoke()

        self.assertIs(call_polygon.call_args.kwargs["binary"], True)

    def test_problem_view_file_requests_binary_response(self) -> None:
        self._assert_binary_call(
            lambda: server.problem_view_file.fn(1, "source", "main.cpp")
        )

    def test_problem_view_solution_requests_binary_response(self) -> None:
        self._assert_binary_call(
            lambda: server.problem_view_solution.fn(1, "main.cpp")
        )

    def test_problem_test_answer_requests_binary_response(self) -> None:
        self._assert_binary_call(
            lambda: server.problem_test_answer.fn(1, "tests", 1)
        )

    def test_problem_test_input_requests_binary_response(self) -> None:
        self._assert_binary_call(
            lambda: server.problem_test_input.fn(1, "tests", 1)
        )


class FileContentResponseTests(unittest.TestCase):
    def test_decodes_utf8_bytes(self) -> None:
        result = _file_content_response("hello\nworld\n".encode())

        self.assertEqual(result, {"data": "hello\nworld\n", "encoding": "utf-8"})

    def test_requires_explicit_binary_mode_for_non_utf8_content(self) -> None:
        data = b"\xff\x00"

        with self.assertRaisesRegex(ValueError, "binary=true"):
            _file_content_response(data)

        result = _file_content_response(data, binary=True)
        self.assertEqual(result["encoding"], "base64")
        self.assertEqual(result["data"], base64.b64encode(data).decode("ascii"))

    def test_binary_mode_rejects_line_pagination(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            _file_content_response(b"hello", start_line=1, binary=True)

    def test_rejects_more_than_maximum_lines(self) -> None:
        with self.assertRaisesRegex(ValueError, f"<= {MAX_FILE_LINES}"):
            _file_content_response("hello", line_count=MAX_FILE_LINES + 1)

    def test_defaults_to_first_maximum_lines_and_returns_next_page(self) -> None:
        data = "".join(f"line {index}\n" for index in range(MAX_FILE_LINES + 1))

        result = _file_content_response(data)

        self.assertTrue(result["truncated"])
        self.assertEqual(result["next_start_line"], MAX_FILE_LINES + 1)
        self.assertNotIn(f"line {MAX_FILE_LINES}\n", result["data"])

    def test_character_limit_counts_unicode_characters_not_bytes(self) -> None:
        data = "é" * (MAX_FILE_CHARS + 1)

        result = _file_content_response(data)

        self.assertEqual(len(result["data"]), MAX_FILE_CHARS)
        self.assertEqual(result["data"], "é" * MAX_FILE_CHARS)
        self.assertTrue(result["truncated"])
        self.assertIn("smaller line_count", result["message"])


class SearchFileContentTests(unittest.TestCase):
    def test_returns_matching_lines_with_context(self) -> None:
        data = "zero\none\nneedle here\nthree\nfour\n"

        result = _search_file_content(data, "needle", before=1, after=2)

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["returned_matches"], 1)
        match = result["matches"][0]
        self.assertEqual(match["line_number"], 3)
        self.assertEqual(match["line"], "needle here")
        self.assertEqual(match["before"], [{"line_number": 2, "text": "one"}])
        self.assertEqual(
            match["after"],
            [
                {"line_number": 4, "text": "three"},
                {"line_number": 5, "text": "four"},
            ],
        )

    def test_search_requires_utf8(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be searched"):
            _search_file_content(b"\xff", "needle")

    def test_limits_number_of_matches(self) -> None:
        data = "\n".join("needle" for _ in range(MAX_FILE_SEARCH_MATCHES + 1))

        result = _search_file_content(data, "needle", before=0, after=0)

        self.assertEqual(result["total_matches"], MAX_FILE_SEARCH_MATCHES + 1)
        self.assertEqual(result["returned_matches"], MAX_FILE_SEARCH_MATCHES)
        self.assertTrue(result["truncated"])

    def test_rejects_context_that_can_exceed_line_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, f"{MAX_FILE_LINES} lines"):
            _search_file_content("needle", "needle", before=25, after=0)

    def test_character_limit_counts_unicode_characters(self) -> None:
        data = "needle " + "é" * MAX_FILE_CHARS

        result = _search_file_content(data, "needle", before=0, after=0)

        self.assertEqual(len(result["matches"][0]["line"]), MAX_FILE_CHARS)
        self.assertTrue(result["truncated"])
        self.assertIn(str(MAX_FILE_CHARS), result["message"])


if __name__ == "__main__":
    unittest.main()
