import unittest
from unittest.mock import Mock, patch

from polygon_api import FileType, Statement

from polygon_mcp import server


class ProblemEditToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _edit(self, **kwargs):
        with patch.object(server, "_get_client", return_value=self.client):
            return server.problem_edit.fn(**kwargs)

    def test_file_edit_reads_saves_and_reports_lines(self) -> None:
        self.client.problem_view_file.return_value = b"int a = 1;\nint b = 2;\n"
        self.client.problem_save_file.return_value = None

        response = self._edit(
            problem_id=7, target="file", type="source", name="val.cpp",
            old_string="int b = 2;", new_string="int b = 3;",
        )

        self.assertEqual(
            self.client.problem_view_file.call_args.args, (7, FileType.SOURCE, "val.cpp")
        )
        self.assertIs(self.client.problem_view_file.call_args.kwargs["binary"], True)
        self.assertEqual(
            self.client.problem_save_file.call_args.args,
            (7, FileType.SOURCE, "val.cpp", "int a = 1;\nint b = 3;\n"),
        )
        self.assertIsNone(self.client.problem_save_file.call_args.kwargs["source_type"])
        self.assertEqual(
            response,
            {
                "target": "source/val.cpp",
                "replacements": 1,
                "lines_before": 2,
                "lines_after": 2,
                "start_line": 2,
                "end_line": 2,
            },
        )

    def test_replace_all_response_lists_replacement_lines(self) -> None:
        self.client.problem_view_file.return_value = b"x\nx\nx\n"
        self.client.problem_save_file.return_value = None

        response = self._edit(
            problem_id=7, target="file", type="source", name="val.cpp",
            old_string="x", new_string="y", replace_all=True,
        )

        self.assertEqual(response["replacements"], 3)
        self.assertEqual(response["replacement_lines"], [1, 2, 3])
        self.assertNotIn("start_line", response)

    def test_solution_edit_sends_no_tag(self) -> None:
        self.client.problem_view_solution.return_value = b"int main() {}\n"
        self.client.problem_save_solution.return_value = None

        self._edit(
            problem_id=7, target="solution", name="main.cpp",
            old_string="int main() {}", new_string="int main() { return 0; }",
        )

        self.assertEqual(
            self.client.problem_save_solution.call_args.args,
            (7, "main.cpp", "int main() { return 0; }\n", None),
        )
        self.assertIsNone(self.client.problem_save_solution.call_args.kwargs["source_type"])

    def test_statement_edit_writes_only_selected_section(self) -> None:
        self.client.problem_statements.return_value = {
            "russian": Statement(legend="Дано число n.\n", input="Одно число.\n")
        }
        self.client.problem_save_statement.return_value = None

        response = self._edit(
            problem_id=7, target="statement", lang="russian", section="legend",
            old_string="число n", new_string="целое число n",
        )

        saved = self.client.problem_save_statement.call_args.args[2]
        self.assertEqual(saved.legend, "Дано целое число n.\n")
        self.assertIsNone(saved.input)
        self.assertEqual(response["target"], "statement russian/legend")

    def test_statement_edit_rejects_unknown_language(self) -> None:
        self.client.problem_statements.return_value = {"english": Statement(legend="a\n")}

        with self.assertRaises(ValueError) as ctx:
            self._edit(
                problem_id=7, target="statement", lang="russian", section="legend",
                old_string="a", new_string="b",
            )

        self.assertIn("Statement not found for lang: russian", str(ctx.exception))

    def test_absent_statement_section_reports_zero_lines(self) -> None:
        self.client.problem_statements.return_value = {"russian": Statement(legend="a\n")}

        with self.assertRaises(ValueError) as ctx:
            self._edit(
                problem_id=7, target="statement", lang="russian", section="notes",
                old_string="anything", new_string="b",
            )

        self.assertIn(
            "old_string not found in statement russian/notes (0 lines)", str(ctx.exception)
        )
        self.client.problem_save_statement.assert_not_called()

    def test_script_edit_reads_and_saves_source(self) -> None:
        self.client.problem_script.return_value = "gen 1 > 1\ngen 2 > 2\n"
        self.client.problem_save_script.return_value = None

        response = self._edit(
            problem_id=7, target="script", testset="tests",
            old_string="gen 2 > 2", new_string="gen 5 > 2",
        )

        self.assertEqual(
            self.client.problem_save_script.call_args.args,
            (7, "tests", "gen 1 > 1\ngen 5 > 2\n"),
        )
        self.assertEqual(response["target"], "script tests")

    def test_rejects_binary_content(self) -> None:
        self.client.problem_view_file.return_value = b"\x00\x01binary"

        with self.assertRaises(ValueError) as ctx:
            self._edit(
                problem_id=7, target="file", type="resource", name="a.bin",
                old_string="binary", new_string="text",
            )

        self.assertIn("binary", str(ctx.exception))
        self.client.problem_save_file.assert_not_called()

    def test_does_not_save_when_match_is_ambiguous(self) -> None:
        self.client.problem_view_file.return_value = b"x\nx\n"

        with self.assertRaises(ValueError):
            self._edit(
                problem_id=7, target="file", type="source", name="val.cpp",
                old_string="x", new_string="y",
            )

        self.client.problem_save_file.assert_not_called()

    def test_includes_api_result_from_save_call(self) -> None:
        self.client.problem_view_file.return_value = b"a\n"
        self.client.problem_save_file.return_value = {"revision": 18}

        response = self._edit(
            problem_id=7, target="file", type="source", name="val.cpp",
            old_string="a", new_string="b",
        )

        self.assertEqual(response["api_result"], {"revision": 18})

    def test_validation_runs_before_any_polygon_call(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._edit(
                problem_id=7, target="script", testset="tests", name="val.cpp",
                old_string="a", new_string="b",
            )

        self.assertIn('target="script" requires testset', str(ctx.exception))
        self.assertEqual(self.client.method_calls, [])

    def test_unknown_target_does_not_touch_the_client(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._edit(problem_id=7, target="checker", old_string="a", new_string="b")

        self.assertIn("Unknown target 'checker'", str(ctx.exception))
        self.assertEqual(self.client.method_calls, [])


if __name__ == "__main__":
    unittest.main()
