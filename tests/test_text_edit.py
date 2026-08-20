import unittest

from polygon_mcp.text_edit import apply_string_edit, count_lines


class CountLinesTests(unittest.TestCase):
    def test_counts_lines_by_newline(self) -> None:
        self.assertEqual(count_lines(""), 0)
        self.assertEqual(count_lines("a\nb\n"), 2)
        self.assertEqual(count_lines("a\nb"), 2)
        self.assertEqual(count_lines("a\r\nb\r\n"), 2)


class ApplyStringEditTests(unittest.TestCase):
    def test_replaces_unique_match(self) -> None:
        updated, spans = apply_string_edit("a\nb\nc\n", "b", "B", False, "source/val.cpp")
        self.assertEqual(updated, "a\nB\nc\n")
        self.assertEqual(spans, [(2, 2, 2)])

    def test_reports_end_line_of_inserted_text(self) -> None:
        updated, spans = apply_string_edit("a\nb\nc\n", "b", "b1\nb2", False, "f")
        self.assertEqual(updated, "a\nb1\nb2\nc\n")
        self.assertEqual(spans, [(2, 2, 3)])

    def test_rejects_missing_match(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            apply_string_edit("a\nb\n", "zzz", "B", False, "source/val.cpp")
        self.assertIn("old_string not found in source/val.cpp (2 lines)", str(ctx.exception))

    def test_rejects_ambiguous_match(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            apply_string_edit("x\nx\nx\n", "x", "y", False, "source/val.cpp")
        self.assertIn(
            "Found 3 matches of old_string in source/val.cpp at lines 1, 2, 3",
            str(ctx.exception),
        )
        self.assertIn("replace_all=true", str(ctx.exception))

    def test_replace_all_replaces_every_match(self) -> None:
        updated, spans = apply_string_edit("x\nx\nx\n", "x", "y", True, "f")
        self.assertEqual(updated, "y\ny\ny\n")
        self.assertEqual([span[1] for span in spans], [1, 2, 3])

    def test_replace_all_shifts_later_spans(self) -> None:
        updated, spans = apply_string_edit("x\nx\n", "x", "y1\ny2", True, "f")
        self.assertEqual(updated, "y1\ny2\ny1\ny2\n")
        self.assertEqual(spans, [(1, 1, 2), (2, 3, 4)])

    def test_crlf_content_matches_lf_old_string(self) -> None:
        updated, spans = apply_string_edit("a\r\nb\r\nc\r\n", "a\nb", "a\nB\nb", False, "f")
        self.assertEqual(updated, "a\r\nB\r\nb\r\nc\r\n")
        self.assertEqual(spans, [(1, 1, 3)])

    def test_mixed_line_endings_outside_edit_are_preserved(self) -> None:
        updated, _ = apply_string_edit("a\r\nb\nc\r\nd\n", "c", "C", False, "f")
        self.assertEqual(updated, "a\r\nb\nC\r\nd\n")

    def test_rejects_identical_strings(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            apply_string_edit("a\n", "a", "a", False, "f")
        self.assertIn("identical", str(ctx.exception))

    def test_rejects_empty_old_string(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            apply_string_edit("a\n", "", "b", False, "f")
        self.assertIn("old_string must not be empty", str(ctx.exception))

    def test_allows_deletion(self) -> None:
        updated, spans = apply_string_edit("a\nb\nc\n", "b\n", "", False, "f")
        self.assertEqual(updated, "a\nc\n")
        self.assertEqual(spans, [(2, 2, 2)])


if __name__ == "__main__":
    unittest.main()
