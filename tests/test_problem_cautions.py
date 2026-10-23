import unittest
from unittest.mock import Mock, patch

from polygon_api import (
    AiTips,
    Caution,
    CautionCategory,
    CautionSeverity,
    PackageReadinessIssue,
    ProblemCautions,
    SourceAiTip,
    StatementAiTip,
)

from polygon_mcp import server


def _sample_cautions() -> ProblemCautions:
    return ProblemCautions(
        common=[
            Caution(
                type="NO_TAGS",
                severity=CautionSeverity.SOFT,
                category=CautionCategory.COMMON,
                message="Problem has no tags",
                parameters={},
            )
        ],
        statement=[],
        structure=[
            Caution(
                type="NO_VALIDATOR",
                severity=CautionSeverity.HARD,
                category=CautionCategory.STRUCTURE,
                message="Problem has no validator",
                parameters={},
            )
        ],
        issues=[],
        package_readiness_issues=[
            PackageReadinessIssue(
                type="NoTestsException", message="No tests", reason="tests"
            )
        ],
        latest_package_warnings=["unused file: brute.cpp"],
        ai=AiTips(
            disabled=False,
            statements=[
                StatementAiTip(
                    language="english",
                    source="Original legend",
                    processing="DONE",
                    suggestion="Corrected legend",
                )
            ],
            validator=SourceAiTip(name="validator.cpp", comment="Looks fine"),
            checker=None,
        ),
    )


class ProblemCautionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()

    def _invoke(self, **kwargs):
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(
                server, "_call_polygon", return_value=_sample_cautions()
            ) as call_polygon,
        ):
            result = server.problem_cautions.fn(42, **kwargs)
        return result, call_polygon

    def test_returns_every_caution_bucket(self) -> None:
        result, call_polygon = self._invoke()

        call_polygon.assert_called_once_with(
            self.client.problem_cautions, 42, pin=None
        )
        self.assertEqual(result["common"][0]["type"], "NO_TAGS")
        self.assertEqual(result["common"][0]["severity"], "SOFT")
        self.assertEqual(result["common"][0]["category"], "COMMON")
        self.assertEqual(result["common"][0]["message"], "Problem has no tags")
        self.assertEqual(result["statement"], [])
        self.assertEqual(result["structure"][0]["severity"], "HARD")
        self.assertEqual(result["issues"], [])
        self.assertEqual(
            result["package_readiness_issues"][0],
            {"type": "NoTestsException", "message": "No tests", "reason": "tests"},
        )
        self.assertEqual(result["latest_package_warnings"], ["unused file: brute.cpp"])

    def test_ai_tips_are_omitted_by_default(self) -> None:
        result, _ = self._invoke()

        self.assertNotIn("ai", result)

    def test_include_ai_returns_the_ai_block(self) -> None:
        result, _ = self._invoke(include_ai=True)

        self.assertIs(result["ai"]["disabled"], False)
        self.assertEqual(result["ai"]["statements"][0]["language"], "english")
        self.assertEqual(
            result["ai"]["statements"][0]["suggestion"], "Corrected legend"
        )
        self.assertEqual(result["ai"]["validator"]["name"], "validator.cpp")
        self.assertIsNone(result["ai"]["checker"])

    def test_forwards_pin(self) -> None:
        _, call_polygon = self._invoke(pin="1234")

        self.assertEqual(call_polygon.call_args.kwargs["pin"], "1234")

    def test_tool_is_marked_read_only(self) -> None:
        self.assertTrue(server.problem_cautions.annotations.readOnlyHint)


if __name__ == "__main__":
    unittest.main()
