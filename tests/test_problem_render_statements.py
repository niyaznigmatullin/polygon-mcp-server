import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from polygon_api import RenderedStatement, RenderResult, RenderStatements, RenderStatus

from polygon_mcp import server

HTML = b"<html>legend</html>"
PDF = b"%PDF-1.5 statement"


def _sample_render(with_content: bool) -> RenderStatements:
    return RenderStatements(
        revision=12,
        rendering_time_seconds=3,
        statements=[
            RenderedStatement(
                language="english",
                html=RenderResult(
                    status=RenderStatus.OK,
                    sha256="aa11",
                    size_bytes=len(HTML),
                    content_bytes=HTML if with_content else None,
                ),
                pdf=RenderResult(
                    status=RenderStatus.OK,
                    sha256="bb22",
                    size_bytes=len(PDF),
                    content_bytes=PDF if with_content else None,
                ),
            )
        ],
        tutorials=[
            RenderedStatement(
                language="english",
                html=RenderResult(
                    status=RenderStatus.FAILED, message="Undefined control sequence"
                ),
                pdf=RenderResult(
                    status=RenderStatus.FAILED, message="Undefined control sequence"
                ),
            )
        ],
    )


class ProblemRenderStatementsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patcher = patch.dict(
            os.environ, {"POLYGON_MCP_OUTPUT_ROOTS": self.tmpdir.name}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _invoke(self, *args, **kwargs):
        with patch.object(server, "_get_client", return_value=self.client):
            return server.problem_render_statements.fn(*args, **kwargs)

    def test_without_output_dir_reports_statuses_only(self) -> None:
        self.client.problem_render_statements.return_value = _sample_render(False)

        result = self._invoke(42, pin="1234")

        self.client.problem_render_statements.assert_called_once_with(
            42, include_content=False, pin="1234"
        )
        self.assertEqual(
            result,
            {
                "revision": 12,
                "rendering_time_seconds": 3,
                "statements": [
                    {
                        "language": "english",
                        "html": {
                            "status": "OK",
                            "sha256": "aa11",
                            "size_bytes": len(HTML),
                            "message": None,
                        },
                        "pdf": {
                            "status": "OK",
                            "sha256": "bb22",
                            "size_bytes": len(PDF),
                            "message": None,
                        },
                    }
                ],
                "tutorials": [
                    {
                        "language": "english",
                        "html": {
                            "status": "FAILED",
                            "sha256": None,
                            "size_bytes": None,
                            "message": "Undefined control sequence",
                        },
                        "pdf": {
                            "status": "FAILED",
                            "sha256": None,
                            "size_bytes": None,
                            "message": "Undefined control sequence",
                        },
                    }
                ],
            },
        )

    def test_output_dir_writes_rendered_files(self) -> None:
        self.client.problem_render_statements.return_value = _sample_render(True)
        output_dir = os.path.join(self.tmpdir.name, "render")

        result = self._invoke(42, output_dir=output_dir)

        self.assertIs(
            self.client.problem_render_statements.call_args.kwargs["include_content"],
            True,
        )
        statement = result["statements"][0]
        html_path = os.path.join(output_dir, "statement-english.html")
        pdf_path = os.path.join(output_dir, "statement-english.pdf")
        self.assertEqual(statement["html"]["saved_to"], html_path)
        self.assertEqual(statement["pdf"]["saved_to"], pdf_path)
        with open(html_path, "rb") as handle:
            self.assertEqual(handle.read(), HTML)
        with open(pdf_path, "rb") as handle:
            self.assertEqual(handle.read(), PDF)

    def test_failed_render_writes_no_file(self) -> None:
        self.client.problem_render_statements.return_value = _sample_render(True)
        output_dir = os.path.join(self.tmpdir.name, "render")

        result = self._invoke(42, output_dir=output_dir)

        tutorial = result["tutorials"][0]
        self.assertNotIn("saved_to", tutorial["html"])
        self.assertNotIn("saved_to", tutorial["pdf"])
        self.assertEqual(
            sorted(os.listdir(output_dir)),
            ["statement-english.html", "statement-english.pdf"],
        )

    def test_rejects_output_dir_outside_allowed_roots_before_api_call(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._invoke(42, output_dir="/definitely/not/allowed")

        self.assertIn("output_path must be within", str(ctx.exception))
        self.client.problem_render_statements.assert_not_called()


if __name__ == "__main__":
    unittest.main()
