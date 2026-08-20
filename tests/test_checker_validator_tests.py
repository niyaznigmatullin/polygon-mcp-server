import unittest
from unittest.mock import Mock, patch

from polygon_api import CheckerTestVerdict, ValidatorTestVerdict

from polygon_mcp import server


class CheckerValidatorTestToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def test_problem_checker_tests(self) -> None:
        checker_test = Mock(
            index=1,
            input="input",
            output="output",
            answer="answer",
            expected_verdict=CheckerTestVerdict.OK,
        )
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(
                server, "_call_polygon", return_value=[checker_test]
            ) as call_polygon,
        ):
            result = server.problem_checker_tests.fn(42)

        call_polygon.assert_called_once_with(self.client.problem_checker_tests, 42)
        self.assertEqual(result[0]["expected_verdict"], "OK")

    def test_problem_save_checker_test_converts_verdict(self) -> None:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            result = server.problem_save_checker_test.fn(
                42,
                3,
                test_input="input",
                test_output="output",
                test_answer="answer",
                test_verdict="wrong_answer",
                check_existing=True,
            )

        call_polygon.assert_called_once_with(
            self.client.problem_save_checker_test,
            42,
            3,
            test_input="input",
            test_output="output",
            test_answer="answer",
            test_verdict=CheckerTestVerdict.WRONG_ANSWER,
            check_existing=True,
        )
        self.assertTrue(result)

    def test_problem_validator_tests(self) -> None:
        validator_test = Mock(
            index=1,
            input="input",
            expected_verdict=ValidatorTestVerdict.INVALID,
            testset="tests",
            group="group-1",
        )
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(
                server, "_call_polygon", return_value=[validator_test]
            ) as call_polygon,
        ):
            result = server.problem_validator_tests.fn(42)

        call_polygon.assert_called_once_with(self.client.problem_validator_tests, 42)
        self.assertEqual(result[0]["expected_verdict"], "INVALID")

    def test_problem_save_validator_test_converts_verdict(self) -> None:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            result = server.problem_save_validator_test.fn(
                42,
                4,
                test_input="input",
                test_verdict="valid",
                test_group="group-1",
                testset="tests",
                check_existing=False,
            )

        call_polygon.assert_called_once_with(
            self.client.problem_save_validator_test,
            42,
            4,
            test_input="input",
            test_verdict=ValidatorTestVerdict.VALID,
            test_group="group-1",
            testset="tests",
            check_existing=False,
        )
        self.assertTrue(result)

    def test_rejects_unknown_verdict(self) -> None:
        with patch.object(server, "_get_client", return_value=self.client):
            with self.assertRaisesRegex(ValueError, "Unknown CheckerTestVerdict"):
                server.problem_save_checker_test.fn(
                    42, 1, test_verdict="accepted"
                )


class CheckerTestLineEndingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _save(self, **kwargs) -> dict:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            server.problem_save_checker_test.fn(42, 3, **kwargs)
        return call_polygon.call_args.kwargs

    def test_converts_lf_to_crlf_in_every_field(self) -> None:
        kwargs = self._save(
            test_input="1 2\n",
            test_output="3\n4\n",
            test_answer="3\n",
        )

        self.assertEqual(kwargs["test_input"], "1 2\r\n")
        self.assertEqual(kwargs["test_output"], "3\r\n4\r\n")
        self.assertEqual(kwargs["test_answer"], "3\r\n")

    def test_keeps_content_that_already_has_crlf(self) -> None:
        kwargs = self._save(test_input="a\r\nb\nc")

        self.assertEqual(kwargs["test_input"], "a\r\nb\nc")

    def test_raw_sends_content_unchanged(self) -> None:
        kwargs = self._save(test_input="1 2\n", raw=True)

        self.assertEqual(kwargs["test_input"], "1 2\n")

    def test_raw_is_not_forwarded_to_polygon(self) -> None:
        kwargs = self._save(test_input="1 2\n", raw=True)

        self.assertNotIn("raw", kwargs)

    def test_omitted_fields_stay_none(self) -> None:
        kwargs = self._save(test_input="1 2\n")

        self.assertIsNone(kwargs["test_output"])
        self.assertIsNone(kwargs["test_answer"])


class ValidatorTestLineEndingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _save(self, **kwargs) -> dict:
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=True) as call_polygon,
        ):
            server.problem_save_validator_test.fn(42, 4, **kwargs)
        return call_polygon.call_args.kwargs

    def test_converts_lf_to_crlf_in_input(self) -> None:
        kwargs = self._save(test_input="5\n1 2 3 4 5\n")

        self.assertEqual(kwargs["test_input"], "5\r\n1 2 3 4 5\r\n")

    def test_keeps_input_that_already_has_crlf(self) -> None:
        kwargs = self._save(test_input="5\r\n1 2 3 4 5\n")

        self.assertEqual(kwargs["test_input"], "5\r\n1 2 3 4 5\n")

    def test_raw_sends_input_unchanged(self) -> None:
        kwargs = self._save(test_input="5\n", raw=True)

        self.assertEqual(kwargs["test_input"], "5\n")

    def test_raw_is_not_forwarded_to_polygon(self) -> None:
        kwargs = self._save(test_input="5\n", raw=True)

        self.assertNotIn("raw", kwargs)


if __name__ == "__main__":
    unittest.main()
