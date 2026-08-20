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


if __name__ == "__main__":
    unittest.main()
