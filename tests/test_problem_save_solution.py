import inspect
import unittest
from unittest.mock import Mock, patch

from polygon_api import Polygon

from polygon_mcp import server


class ProblemSaveSolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _bound_api_arguments(self, **kwargs) -> dict:
        """Invoke the tool and resolve the forwarded args to API parameter names."""
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            server.problem_save_solution.fn(**kwargs)

        args, call_kwargs = call_polygon.call_args
        self.assertIs(args[0], self.client.problem_save_solution)
        bound = inspect.signature(Polygon.problem_save_solution).bind(
            None, *args[1:], **call_kwargs
        )
        bound.apply_defaults()
        return bound.arguments

    def test_tag_and_source_type_reach_their_own_api_parameters(self) -> None:
        arguments = self._bound_api_arguments(
            problem_id=42,
            name="main.cpp",
            source_type="cpp.g++17",
            tag="MA",
            content="int main(){}",
        )

        self.assertEqual(arguments["tag"], "MA")
        self.assertEqual(arguments["source_type"], "cpp.g++17")

    def test_forwards_problem_id_name_and_file(self) -> None:
        arguments = self._bound_api_arguments(
            problem_id=42,
            name="main.cpp",
            content="int main(){}",
        )

        self.assertEqual(arguments["problem_id"], 42)
        self.assertEqual(arguments["name"], "main.cpp")
        self.assertEqual(arguments["file"], "int main(){}")


if __name__ == "__main__":
    unittest.main()
