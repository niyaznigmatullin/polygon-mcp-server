import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from polygon_api import PackageType

from polygon_mcp import server


class ProblemPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.output_path = os.path.join(self.tmpdir.name, "package.zip")
        patcher = patch.dict(
            os.environ, {"POLYGON_MCP_OUTPUT_ROOTS": self.tmpdir.name}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _invoke(self, data, **kwargs):
        with (
            patch.object(server, "_get_client", return_value=self.client),
            patch.object(server, "_call_polygon", return_value=data) as call_polygon,
        ):
            result = server.problem_package.fn(
                42, 7, self.output_path, **kwargs
            )
        return result, call_polygon

    def test_writes_archive_to_output_path(self) -> None:
        archive = b"PK\x03\x04payload"

        result, call_polygon = self._invoke(archive)

        call_polygon.assert_called_once_with(
            self.client.problem_package, 42, 7, type=None
        )
        self.assertEqual(
            result, {"saved_to": self.output_path, "size_bytes": len(archive)}
        )
        with open(self.output_path, "rb") as handle:
            self.assertEqual(handle.read(), archive)

    def test_passes_package_type_as_lowercase_name(self) -> None:
        _, call_polygon = self._invoke(b"PK\x03\x04", type="Linux")

        self.assertEqual(call_polygon.call_args.kwargs["type"], "linux")
        self.assertEqual(str(PackageType.LINUX), "linux")

    def test_rejects_unknown_package_type(self) -> None:
        with patch.object(server, "_get_client", return_value=self.client):
            with self.assertRaisesRegex(ValueError, "Unknown PackageType"):
                server.problem_package.fn(42, 7, self.output_path, type="full")

    def test_rejects_output_path_outside_allowed_roots(self) -> None:
        with patch.object(server, "_get_client", return_value=self.client):
            with self.assertRaisesRegex(ValueError, "output_path must be within"):
                server.problem_package.fn(42, 7, "/etc/polygon-package.zip")

    def test_surfaces_failed_json_body_as_error(self) -> None:
        body = json.dumps(
            {"status": "FAILED", "comment": "packageId: Package not found"}
        ).encode()

        with self.assertRaisesRegex(RuntimeError, "Package not found"):
            self._invoke(body)

        self.assertFalse(os.path.exists(self.output_path))


class ErrorPayloadDetectionTests(unittest.TestCase):
    def test_ignores_binary_archive(self) -> None:
        server._raise_if_polygon_error_payload(b"PK\x03\x04\xff\x00")

    def test_ignores_large_payload_that_starts_like_json(self) -> None:
        data = b"{" + b"a" * server.MAX_ERROR_PAYLOAD_BYTES

        server._raise_if_polygon_error_payload(data)

    def test_ignores_ok_status(self) -> None:
        server._raise_if_polygon_error_payload(b'{"status": "OK", "result": 1}')

    def test_defaults_comment_when_missing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unknown error"):
            server._raise_if_polygon_error_payload(b'{"status": "FAILED"}')


if __name__ == "__main__":
    unittest.main()
