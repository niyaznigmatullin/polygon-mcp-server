import inspect
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from polygon_api import Polygon

from polygon_mcp import server


class ProblemSaveTestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _bound_api_arguments(self, **kwargs) -> dict:
        """Invoke the tool and resolve the forwarded args to API parameter names."""
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            server.problem_save_test.fn(**kwargs)

        args, call_kwargs = call_polygon.call_args
        self.assertIs(args[0], self.client.problem_save_test)
        bound = inspect.signature(Polygon.problem_save_test).bind(
            None, *args[1:], **call_kwargs
        )
        bound.apply_defaults()
        return bound.arguments

    def _write_temp_file(self, data: bytes) -> str:
        handle = tempfile.NamedTemporaryFile(delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            handle.write(data)
        return handle.name

    def test_forwards_inline_test_input(self) -> None:
        arguments = self._bound_api_arguments(
            problem_id=42, testset="tests", test_index=3, test_input="1 2\n"
        )

        self.assertEqual(arguments["problem_id"], 42)
        self.assertEqual(arguments["testset"], "tests")
        self.assertEqual(arguments["test_index"], 3)
        self.assertEqual(arguments["test_input"], "1 2\n")

    def test_reads_test_input_from_local_path(self) -> None:
        path = self._write_temp_file(b"1 2\r\n\xff")

        arguments = self._bound_api_arguments(
            problem_id=42, testset="tests", test_index=3, local_path=path
        )

        self.assertEqual(arguments["test_input"], b"1 2\r\n\xff")

    def test_rejects_both_test_input_and_local_path(self) -> None:
        path = self._write_temp_file(b"1 2\n")

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self._bound_api_arguments(
                problem_id=42,
                testset="tests",
                test_index=3,
                test_input="1 2\n",
                local_path=path,
            )

    def test_omitting_input_keeps_existing_test_input(self) -> None:
        arguments = self._bound_api_arguments(
            problem_id=42, testset="tests", test_index=3, test_points=5
        )

        self.assertIsNone(arguments["test_input"])
        self.assertEqual(arguments["test_points"], 5)


if __name__ == "__main__":
    unittest.main()
