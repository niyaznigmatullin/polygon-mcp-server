import inspect
import unittest
from unittest.mock import Mock, patch

from polygon_api import Polygon

from polygon_mcp import server


class ProblemBuildPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def test_passes_verify_and_full_as_keywords(self) -> None:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            result = server.problem_build_package.fn(42, False, True)

        call_polygon.assert_called_once_with(
            self.client.problem_build_package, 42, verify=True, full=False
        )
        self.assertTrue(result)

    def test_rejects_full_packages(self) -> None:
        with patch.object(server, "_get_client", return_value=self.client):
            with self.assertRaisesRegex(ValueError, "full packages are disabled"):
                server.problem_build_package.fn(42, True, True)

    def test_client_accepts_verify_and_full_keywords(self) -> None:
        parameters = inspect.signature(Polygon.problem_build_package).parameters

        self.assertIn("verify", parameters)
        self.assertIn("full", parameters)


if __name__ == "__main__":
    unittest.main()
