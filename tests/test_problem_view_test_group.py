import inspect
import unittest
from unittest.mock import Mock, patch

from polygon_api import Polygon

from polygon_mcp import server


class ProblemViewTestGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _bound_api_arguments(self, *args, **kwargs) -> dict:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=[]) as call_polygon,
        ):
            server.problem_view_test_group.fn(*args, **kwargs)

        call_args, call_kwargs = call_polygon.call_args
        self.assertIs(call_args[0], self.client.problem_view_test_group)
        bound = inspect.signature(Polygon.problem_view_test_group).bind(
            None, *call_args[1:], **call_kwargs
        )
        bound.apply_defaults()
        return bound.arguments

    def test_forwards_problem_id_testset_and_group(self) -> None:
        arguments = self._bound_api_arguments(42, "tests", "group1")

        self.assertEqual(arguments["problem_id"], 42)
        self.assertEqual(arguments["testset"], "tests")
        self.assertEqual(arguments["group"], "group1")

    def test_group_is_optional(self) -> None:
        arguments = self._bound_api_arguments(42, "tests")

        self.assertEqual(arguments["problem_id"], 42)
        self.assertEqual(arguments["testset"], "tests")
        self.assertIsNone(arguments["group"])


if __name__ == "__main__":
    unittest.main()
