import base64
import unittest

from polygon_mcp.server import MAX_FILE_CHARS, MAX_FILE_LINES, _file_content_response


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


if __name__ == "__main__":
    unittest.main()
