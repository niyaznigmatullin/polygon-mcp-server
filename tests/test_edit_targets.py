import unittest

from polygon_mcp.edit_targets import build_edit_response, validate_edit_target

NO_ADDRESS = {
    "type": None,
    "name": None,
    "lang": None,
    "section": None,
    "testset": None,
    "source_type": None,
}


class ValidateEditTargetTests(unittest.TestCase):
    def test_normalizes_target_and_returns_address(self) -> None:
        target, address = validate_edit_target(
            " File ", {**NO_ADDRESS, "type": "source", "name": "val.cpp"}
        )
        self.assertEqual(target, "file")
        self.assertEqual(address, {"type": "source", "name": "val.cpp"})

    def test_keeps_optional_source_type(self) -> None:
        _, address = validate_edit_target(
            "solution", {**NO_ADDRESS, "name": "main.cpp", "source_type": "cpp.g++17"}
        )
        self.assertEqual(address, {"name": "main.cpp", "source_type": "cpp.g++17"})

    def test_rejects_unknown_target(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_edit_target("checker", dict(NO_ADDRESS))
        self.assertIn("Unknown target 'checker'", str(ctx.exception))
        self.assertIn("file, script, solution, statement", str(ctx.exception))

    def test_rejects_missing_required_parameter(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_edit_target("statement", {**NO_ADDRESS, "lang": "russian"})
        self.assertEqual(
            str(ctx.exception),
            'target="statement" requires lang, section; '
            "type, name, testset, source_type are not allowed.",
        )

    def test_rejects_forbidden_parameter(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_edit_target(
                "script", {**NO_ADDRESS, "testset": "tests", "name": "val.cpp"}
            )
        self.assertEqual(
            str(ctx.exception),
            'target="script" requires testset; '
            "type, name, lang, section, source_type are not allowed.",
        )


class BuildEditResponseTests(unittest.TestCase):
    def test_single_span_reports_start_and_end_line(self) -> None:
        response = build_edit_response(
            "source/val.cpp", "a\nb\n", "a\nB1\nB2\n", [(2, 2, 3)], None
        )
        self.assertEqual(
            response,
            {
                "target": "source/val.cpp",
                "replacements": 1,
                "lines_before": 2,
                "lines_after": 3,
                "start_line": 2,
                "end_line": 3,
            },
        )

    def test_multiple_spans_report_replacement_lines_and_api_result(self) -> None:
        response = build_edit_response(
            "script tests", "x\nx\n", "y\ny\n", [(1, 1, 1), (2, 2, 2)], {"revision": 18}
        )
        self.assertEqual(response["replacements"], 2)
        self.assertEqual(response["replacement_lines"], [1, 2])
        self.assertNotIn("start_line", response)
        self.assertEqual(response["api_result"], {"revision": 18})


if __name__ == "__main__":
    unittest.main()
