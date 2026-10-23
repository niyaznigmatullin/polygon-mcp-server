"""The pin argument must survive the trip from tool parameter to API call."""

import unittest
from unittest.mock import Mock, patch

from polygon_mcp import server


class PinForwardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _forwarded_kwargs(self, tool, *args, **kwargs) -> dict:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=None) as call_polygon,
        ):
            tool.fn(*args, **kwargs)
        return call_polygon.call_args.kwargs

    def test_read_tool_forwards_pin(self) -> None:
        kwargs = self._forwarded_kwargs(server.problem_info, 42, pin="1234")

        self.assertEqual(kwargs["pin"], "1234")

    def test_write_tool_forwards_pin(self) -> None:
        kwargs = self._forwarded_kwargs(
            server.problem_save_tags, 42, ["dp"], pin="1234"
        )

        self.assertEqual(kwargs["pin"], "1234")

    def test_pin_defaults_to_none(self) -> None:
        kwargs = self._forwarded_kwargs(server.problem_info, 42)

        self.assertIsNone(kwargs["pin"])

    def test_problem_edit_forwards_pin_to_both_read_and_save(self) -> None:
        self.client.problem_script.return_value = "gen 1 > 1"

        with patch.object(server, "_get_client", return_value=self.client):
            server.problem_edit.fn(
                problem_id=7,
                target="script",
                testset="tests",
                old_string="1 > 1",
                new_string="2 > 2",
                pin="1234",
            )

        self.assertEqual(self.client.problem_script.call_args.kwargs["pin"], "1234")
        self.assertEqual(
            self.client.problem_save_script.call_args.kwargs["pin"], "1234"
        )


if __name__ == "__main__":
    unittest.main()
