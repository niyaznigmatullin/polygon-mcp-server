"""The fields polygon_api 1.2.0a1 added to ProblemInfo and Statement."""

import unittest
from unittest.mock import Mock, patch

from polygon_mcp import server


class ProblemUpdateInfoFieldsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _sent_info(self, **kwargs):
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            server.problem_update_info.fn(42, **kwargs)
        return call_polygon.call_args.args[2]

    def test_sends_well_formed(self) -> None:
        info = self._sent_info(well_formed=False)

        self.assertIs(info.well_formed, False)

    def test_sends_skip_duplicated_tests_validation(self) -> None:
        info = self._sent_info(skip_duplicated_tests_validation=True)

        self.assertIs(info.skip_duplicated_tests_validation, True)

    def test_unset_new_fields_stay_none(self) -> None:
        info = self._sent_info(time_limit=2000)

        self.assertEqual(info.time_limit, 2000)
        self.assertIsNone(info.well_formed)
        self.assertIsNone(info.skip_duplicated_tests_validation)


class ProblemSaveStatementFieldsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _sent_statement(self, **kwargs):
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            server.problem_save_statement.fn(42, "english", **kwargs)
        return call_polygon.call_args.args[3]

    def test_sends_show_in_review(self) -> None:
        statement = self._sent_statement(show_in_review=True)

        self.assertIs(statement.show_in_review, True)

    def test_sends_show_cautions_and_grammatical_fixes(self) -> None:
        statement = self._sent_statement(show_cautions_and_grammatical_fixes=False)

        self.assertIs(statement.show_cautions_and_grammatical_fixes, False)

    def test_unset_new_fields_stay_none(self) -> None:
        statement = self._sent_statement(legend="Once upon a time")

        self.assertEqual(statement.legend, "Once upon a time")
        self.assertIsNone(statement.show_in_review)
        self.assertIsNone(statement.show_cautions_and_grammatical_fixes)


if __name__ == "__main__":
    unittest.main()
