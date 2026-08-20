import base64
import json
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from enum import Enum
from typing import Any, Optional

from fastmcp import FastMCP
from polygon_api import (
    CheckerTestVerdict,
    FeedbackPolicy,
    FileType,
    PackageType,
    PointsPolicy,
    Polygon,
    PolygonRequestFailedException,
    ProblemInfo,
    ResourceAdvancedProperties,
    Statement,
    ValidatorTestVerdict,
)

from .edit_targets import build_edit_response, validate_edit_target
from .text_edit import apply_string_edit

try:
    from polygon_api import HTTPRequestFailedException
except ImportError:  # pragma: no cover - older exports
    from polygon_api.api import HTTPRequestFailedException

DEFAULT_API_URL = "https://polygon.codeforces.com/api/"
DEFAULT_CONFIG_PATH = os.path.join(
    os.getenv("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")),
    "polygon-mcp",
    "config.json",
)

mcp = FastMCP("polygon")

READ_ONLY_TOOL_ANNOTATIONS = {"readOnlyHint": True}
MAX_FILE_LINES = 500
MAX_FILE_CHARS = 12_000
MAX_FILE_SEARCH_MATCHES = 20
MAX_ERROR_PAYLOAD_BYTES = 4_096


def _warn_stderr(message: str) -> None:
    try:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("polygon_mcp")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log_path = os.getenv("POLYGON_MCP_LOG_FILE")
    if not log_path:
        state_home = os.getenv("XDG_STATE_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "state"
        )
        log_path = os.path.join(state_home, "polygon-mcp", "polygon-mcp.log")

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8"):
            pass
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
    except OSError as exc:
        _warn_stderr(
            f"polygon-mcp warning: failed to open log file '{log_path}': {exc}"
        )
        handler = logging.NullHandler()

    logger.addHandler(handler)
    return logger


_LOGGER = _configure_logger()


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        data = {}
        for key, item in vars(value).items():
            if key.startswith("_"):
                continue
            data[key] = _to_jsonable(item)
        return data
    return str(value)


def _load_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    parsed = data
    return parsed if isinstance(parsed, dict) else {}


def _resolve_config_path() -> str:
    return os.getenv("POLYGON_MCP_CONFIG") or DEFAULT_CONFIG_PATH


def _write_config(path: str, payload: dict) -> None:
    if not path:
        raise ValueError("config_path is empty")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _save_config(path: str, updates: dict) -> dict:
    existing = _load_config(path)
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    _write_config(path, merged)
    return merged


def _resolve_config() -> tuple[str, str, str]:
    config_path = _resolve_config_path()
    stored = _load_config(config_path)

    api_url = os.getenv("POLYGON_API_URL") or stored.get("api_url") or DEFAULT_API_URL
    api_key = os.getenv("POLYGON_API_KEY") or stored.get("api_key")
    api_secret = os.getenv("POLYGON_API_SECRET") or stored.get("api_secret")

    if not api_key or not api_secret:
        raise ValueError("Missing credentials: set POLYGON_API_KEY and POLYGON_API_SECRET")

    return api_url, api_key, api_secret


_polygon_client: Optional[Polygon] = None


def _get_client() -> Polygon:
    global _polygon_client
    if _polygon_client is None:
        api_url, api_key, api_secret = _resolve_config()
        _polygon_client = Polygon(api_url, api_key, api_secret)
    return _polygon_client


@mcp.tool()
def configure_polygon_credentials(
    api_key: str,
    api_secret: str,
    api_url: Optional[str] = None,
) -> Any:
    """Store Polygon API credentials in the MCP config file."""
    key = api_key.strip()
    secret = api_secret.strip()
    if not key:
        raise ValueError("api_key is empty")
    if not secret:
        raise ValueError("api_secret is empty")
    url = api_url.strip() if api_url is not None else None
    config_path = _resolve_config_path()
    stored = _save_config(
        config_path,
        {"api_key": key, "api_secret": secret, "api_url": url},
    )
    global _polygon_client
    _polygon_client = None
    return {
        "config_path": config_path,
        "api_url": stored.get("api_url") or DEFAULT_API_URL,
        "stored": {"api_key": True, "api_secret": True, "api_url": url is not None},
    }


def _call_polygon(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (PolygonRequestFailedException, HTTPRequestFailedException) as exc:
        message = getattr(exc, "comment", None) or str(exc)
        raise RuntimeError(f"Polygon API error: {message}") from exc


def _parse_file_type(value: Optional[str]):
    if value is None:
        return None
    if isinstance(value, FileType):
        return value
    normalized = str(value).strip().lower()
    for file_type in FileType:
        if normalized in (str(file_type).lower(), file_type.name.lower()):
            return file_type
    raise ValueError(f"Unknown file type: {value}")


def _parse_enum(enum_cls, value, *, allow_none: bool = True):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Missing value for {enum_cls.__name__}")
    if isinstance(value, enum_cls):
        return value
    normalized = str(value).strip().upper()
    for item in enum_cls:
        if normalized == item.name.upper():
            return item
    raise ValueError(f"Unknown {enum_cls.__name__}: {value}")


def _decode_content(content: str, content_base64: bool) -> Any:
    if content_base64:
        return base64.b64decode(content)
    return content


def _to_crlf(value: Optional[str]) -> Optional[str]:
    """Convert line endings to CRLF, the form Polygon stores test data in.

    Content that already contains a CRLF is passed through untouched: the
    caller picked its line endings deliberately.
    """
    if value is None or "\r\n" in value:
        return value
    return value.replace("\n", "\r\n")


def _read_local_file(path: str) -> bytes:
    if not path:
        raise ValueError("local_path is empty")
    with open(path, "rb") as handle:
        return handle.read()


def _raise_if_polygon_error_payload(data: bytes) -> None:
    """Raw endpoints skip the status check, so surface FAILED JSON bodies as errors."""
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_ERROR_PAYLOAD_BYTES:
        return
    stripped = bytes(data).lstrip()
    if not stripped.startswith(b"{"):
        return
    try:
        payload = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or payload.get("status") != "FAILED":
        return
    raise RuntimeError(f"Polygon API error: {payload.get('comment') or 'unknown error'}")


def _resolve_output_path(path: str) -> str:
    if not path:
        raise ValueError("output_path is empty")
    abs_path = os.path.abspath(path)
    allowed_roots = [os.getcwd(), "/tmp"]
    extra_roots = os.getenv("POLYGON_MCP_OUTPUT_ROOTS")
    if extra_roots:
        allowed_roots.extend([os.path.abspath(p) for p in extra_roots.split(os.pathsep) if p])
    if not any(
        abs_path == root or abs_path.startswith(root.rstrip(os.sep) + os.sep)
        for root in allowed_roots
    ):
        raise ValueError(
            "output_path must be within the project directory, /tmp, or POLYGON_MCP_OUTPUT_ROOTS"
        )
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return abs_path


def _decode_file_content(data: str | bytes) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        try:
            return bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "File is not valid UTF-8; rerun with binary=true to receive base64-encoded content"
            ) from exc
    raise TypeError(f"Unsupported file content type: {type(data).__name__}")


def _file_content_response(
    data: str | bytes,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    binary: bool = False,
) -> dict[str, Any]:
    if binary:
        if start_line is not None or line_count is not None:
            raise ValueError("start_line and line_count cannot be used with binary=true")
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        return {
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }

    text = _decode_file_content(data)
    if start_line is None:
        start_line = 1
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if line_count is not None and line_count < 0:
        raise ValueError("line_count must be >= 0")
    if line_count is not None and line_count > MAX_FILE_LINES:
        raise ValueError(f"line_count must be <= {MAX_FILE_LINES}")

    effective_line_count = MAX_FILE_LINES if line_count is None else line_count
    lines = text.splitlines(keepends=True)
    start_index = start_line - 1
    end_index = start_index + effective_line_count
    selected_lines = lines[start_index:end_index]
    selected_text = "".join(selected_lines)
    has_more_lines = end_index < len(lines)
    exceeds_char_limit = len(selected_text) > MAX_FILE_CHARS

    response: dict[str, Any] = {
        "data": selected_text[:MAX_FILE_CHARS],
        "encoding": "utf-8",
    }
    if has_more_lines or exceeds_char_limit:
        response["truncated"] = True
        if exceeds_char_limit:
            response["message"] = (
                f"Content exceeds {MAX_FILE_CHARS} characters; returned the first "
                f"{MAX_FILE_CHARS} characters. Retry with a smaller line_count and paginate "
                "with start_line."
            )
        else:
            response["next_start_line"] = start_line + len(selected_lines)
            response["message"] = (
                f"Returned at most {MAX_FILE_LINES} lines. Continue with "
                f"start_line={response['next_start_line']}."
            )
    return response


def _search_file_content(
    data: str | bytes,
    query: str,
    before: int = 5,
    after: int = 15,
    max_matches: int = MAX_FILE_SEARCH_MATCHES,
) -> dict[str, Any]:
    if not query:
        raise ValueError("query is empty")
    if "\n" in query or "\r" in query:
        raise ValueError("query must be a single line")
    if before < 0:
        raise ValueError("before must be >= 0")
    if after < 0:
        raise ValueError("after must be >= 0")
    if max_matches < 1 or max_matches > MAX_FILE_SEARCH_MATCHES:
        raise ValueError(f"max_matches must be between 1 and {MAX_FILE_SEARCH_MATCHES}")
    if (before + after + 1) * max_matches > MAX_FILE_LINES:
        raise ValueError(
            f"before/after context across max_matches must not exceed {MAX_FILE_LINES} lines"
        )

    try:
        text = _decode_file_content(data)
    except ValueError as exc:
        raise ValueError(
            "File is not valid UTF-8 and cannot be searched; use problem_view_file "
            "with binary=true to receive base64-encoded content"
        ) from exc
    lines = text.splitlines()
    matching_indices = [index for index, line in enumerate(lines) if query in line]
    selected_indices = matching_indices[:max_matches]
    remaining_chars = MAX_FILE_CHARS
    matches: list[dict[str, Any]] = []
    content_truncated = False

    def take_line(index: int) -> dict[str, Any]:
        nonlocal remaining_chars, content_truncated
        line = lines[index]
        visible = line[:remaining_chars]
        if len(visible) < len(line):
            content_truncated = True
        remaining_chars -= len(visible)
        return {"line_number": index + 1, "text": visible}

    for index in selected_indices:
        if remaining_chars <= 0:
            content_truncated = True
            break

        match_line = take_line(index)
        start_index = max(0, index - before)
        end_index = min(len(lines), index + after + 1)
        before_lines = []
        after_lines = []

        for context_index in range(start_index, index):
            if remaining_chars <= 0:
                content_truncated = True
                break
            before_lines.append(take_line(context_index))
        for context_index in range(index + 1, end_index):
            if remaining_chars <= 0:
                content_truncated = True
                break
            after_lines.append(take_line(context_index))

        matches.append(
            {
                "line_number": index + 1,
                "line": match_line["text"],
                "before": before_lines,
                "after": after_lines,
            }
        )
        if content_truncated:
            break

    response: dict[str, Any] = {
        "query": query,
        "total_matches": len(matching_indices),
        "returned_matches": len(matches),
        "matches": matches,
        "encoding": "utf-8",
    }
    limited_by_matches = len(matching_indices) > len(selected_indices)
    if limited_by_matches or content_truncated:
        response["truncated"] = True
        reasons = []
        if limited_by_matches:
            reasons.append(
                f"found {len(matching_indices)} matching lines and considered the first "
                f"{len(selected_indices)}"
            )
        if content_truncated:
            reasons.append(f"context exceeded {MAX_FILE_CHARS} characters")
        response["message"] = (
            "; ".join(reasons)
            + ". Refine query or reduce before, after, or max_matches."
        )
    return response


_STATEMENT_SECTIONS = {
    "legend",
    "input",
    "output",
    "notes",
    "tutorial",
    "scoring",
    "interaction",
}


def _normalize_statement_section(section: str) -> str:
    normalized = section.strip().lower()
    if normalized not in _STATEMENT_SECTIONS:
        raise ValueError(
            "Unknown statement section. Use one of: legend, input, output, notes, tutorial, scoring, interaction."
        )
    return normalized


def _resource_adv_from_dict(data: Optional[dict]) -> Optional[ResourceAdvancedProperties]:
    if data is None:
        return None
    if data.get("delete") is True:
        return ResourceAdvancedProperties.DELETE
    return ResourceAdvancedProperties(
        for_types=data.get("for_types"),
        main=data.get("main"),
        stages=data.get("stages"),
        assets=data.get("assets"),
    )


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problems_list(
    show_deleted: Optional[bool] = None,
    id: Optional[int] = None,
    name: Optional[str] = None,
    owner: Optional[str] = None,
) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problems_list, show_deleted=show_deleted, id=id, name=name, owner=owner)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_info(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_info, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_create(name: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_create, name)
    return _to_jsonable(result)


@mcp.tool()
def problem_update_info(
    problem_id: int,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    interactive: Optional[bool] = None,
    time_limit: Optional[int] = None,
    memory_limit: Optional[int] = None,
) -> Any:
    polygon = _get_client()
    info = ProblemInfo(
        input_file=input_file,
        output_file=output_file,
        interactive=interactive,
        time_limit=time_limit,
        memory_limit=memory_limit,
    )
    result = _call_polygon(polygon.problem_update_info, problem_id, info)
    return _to_jsonable(result)


@mcp.tool()
def problem_update_working_copy(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_update_working_copy, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_discard_working_copy(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_discard_working_copy, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_commit_changes(
    problem_id: int,
    minor_changes: Optional[bool] = None,
    message: Optional[str] = None,
) -> Any:
    polygon = _get_client()
    result = _call_polygon(
        polygon.problem_commit_changes,
        problem_id,
        minor_changes=minor_changes,
        message=message,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_statements(
    problem_id: int,
    lang: Optional[str] = None,
    fields: Optional[list[str]] = None,
) -> Any:
    """Get problem statements, optionally selecting a language and fields.

    fields can include: encoding, name, legend, input, output, scoring,
    interaction, notes, tutorial.
    """
    polygon = _get_client()
    result = _call_polygon(polygon.problem_statements, problem_id)
    data = _to_jsonable(result)
    if not isinstance(data, dict):
        return data
    if lang is not None:
        statement = data.get(lang)
        if statement is None:
            raise ValueError(f"Statement not found for lang: {lang}")
        if fields is None:
            return {lang: statement}
        field_set = set(fields)
        return {lang: {k: v for k, v in statement.items() if k in field_set}}
    if fields is None:
        return data
    field_set = set(fields)
    return {k: {sk: sv for sk, sv in v.items() if sk in field_set} for k, v in data.items()}


@mcp.tool()
def problem_save_statement(
    problem_id: int,
    lang: str,
    encoding: Optional[str] = None,
    name: Optional[str] = None,
    legend: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    scoring: Optional[str] = None,
    interaction: Optional[str] = None,
    notes: Optional[str] = None,
    tutorial: Optional[str] = None,
) -> Any:
    """Save or partially update a statement. Use None to leave fields unchanged."""
    polygon = _get_client()
    statement = Statement(
        encoding=encoding,
        name=name,
        legend=legend,
        input=input,
        output=output,
        scoring=scoring,
        interaction=interaction,
        notes=notes,
        tutorial=tutorial,
    )
    result = _call_polygon(polygon.problem_save_statement, problem_id, lang, statement)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_statement_resources(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_statement_resources, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_statement_resource(
    problem_id: int,
    name: str,
    content: Optional[str] = None,
    content_base64: bool = False,
    local_path: Optional[str] = None,
    check_existing: Optional[bool] = None,
) -> Any:
    polygon = _get_client()
    if local_path:
        file_value = _read_local_file(local_path)
    else:
        if content is None:
            raise ValueError("content or local_path is required")
        file_value = _decode_content(content, content_base64)
    result = _call_polygon(
        polygon.problem_save_statement_resource,
        problem_id,
        name,
        file_value,
        check_existing=check_existing,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_checker(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_checker, problem_id)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_checker_tests(problem_id: int) -> Any:
    """List checker tests and their expected verdicts."""
    polygon = _get_client()
    result = _call_polygon(polygon.problem_checker_tests, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_checker_test(
    problem_id: int,
    test_index: int,
    test_input: Optional[str] = None,
    test_output: Optional[str] = None,
    test_answer: Optional[str] = None,
    test_verdict: Optional[str] = None,
    check_existing: Optional[bool] = None,
    raw: Optional[bool] = None,
) -> Any:
    """Add or update a checker test.

    test_verdict can be OK, WRONG_ANSWER, CRASHED, or PRESENTATION_ERROR.
    Line endings in test_input/test_output/test_answer are converted to CRLF,
    the form Polygon stores test data in; content that already contains a CRLF
    is left as is. Pass raw=true to send the content unchanged.
    """
    polygon = _get_client()
    verdict = _parse_enum(CheckerTestVerdict, test_verdict)
    if not raw:
        test_input = _to_crlf(test_input)
        test_output = _to_crlf(test_output)
        test_answer = _to_crlf(test_answer)
    result = _call_polygon(
        polygon.problem_save_checker_test,
        problem_id,
        test_index,
        test_input=test_input,
        test_output=test_output,
        test_answer=test_answer,
        test_verdict=verdict,
        check_existing=check_existing,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_validator(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_validator, problem_id)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_validator_tests(problem_id: int) -> Any:
    """List validator tests and their expected verdicts."""
    polygon = _get_client()
    result = _call_polygon(polygon.problem_validator_tests, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_validator_test(
    problem_id: int,
    test_index: int,
    test_input: Optional[str] = None,
    test_verdict: Optional[str] = None,
    test_group: Optional[str] = None,
    testset: Optional[str] = None,
    check_existing: Optional[bool] = None,
    raw: Optional[bool] = None,
) -> Any:
    """Add or update a validator test.

    test_verdict can be VALID or INVALID.
    Line endings in test_input are converted to CRLF, the form Polygon stores
    test data in; content that already contains a CRLF is left as is. Pass
    raw=true to send the content unchanged.
    """
    polygon = _get_client()
    verdict = _parse_enum(ValidatorTestVerdict, test_verdict)
    if not raw:
        test_input = _to_crlf(test_input)
    result = _call_polygon(
        polygon.problem_save_validator_test,
        problem_id,
        test_index,
        test_input=test_input,
        test_verdict=verdict,
        test_group=test_group,
        testset=testset,
        check_existing=check_existing,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_interactor(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_interactor, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_set_validator(problem_id: int, validator: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_set_validator, problem_id, validator)
    return _to_jsonable(result)


@mcp.tool()
def problem_set_checker(problem_id: int, checker: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_set_checker, problem_id, checker)
    return _to_jsonable(result)


@mcp.tool()
def problem_set_interactor(problem_id: int, interactor: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_set_interactor, problem_id, interactor)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_files(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_files, problem_id)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_view_file(
    problem_id: int,
    type: str,
    name: str,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    binary: bool = False,
) -> Any:
    """Read a problem file with bounded line pagination.

    Text must be valid UTF-8. At most 500 lines and 12,000 characters are
    returned per call. Set binary=true to receive the entire file as base64;
    line pagination is unavailable in binary mode.
    """
    polygon = _get_client()
    file_type = _parse_file_type(type)
    data = _call_polygon(
        polygon.problem_view_file, problem_id, file_type, name, binary=True
    )
    return _file_content_response(data, start_line, line_count, binary)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_search_file(
    problem_id: int,
    type: str,
    name: str,
    query: str,
    before: int = 5,
    after: int = 15,
    max_matches: int = MAX_FILE_SEARCH_MATCHES,
) -> Any:
    """Search a UTF-8 problem file without returning the whole file.

    Search is literal and case-sensitive. Results include matching line numbers
    and bounded context. At most 20 matches, 500 context lines, and 12,000
    characters are returned per call.
    """
    polygon = _get_client()
    file_type = _parse_file_type(type)
    data = _call_polygon(
        polygon.problem_view_file, problem_id, file_type, name, binary=True
    )
    return _search_file_content(data, query, before, after, max_matches)


@mcp.tool()
def problem_save_file(
    problem_id: int,
    type: str,
    name: str,
    content: Optional[str] = None,
    content_base64: bool = False,
    local_path: Optional[str] = None,
    source_type: Optional[str] = None,
    resource_advanced_properties: Optional[dict] = None,
) -> Any:
    """Add or edit a file (source, resource, attachment, etc.).

    For edits, all parameters except problem_id, type and name are optional.

    Parameters:
        problem_id: problem ID
        type: file type (source, resource, attachment, etc.)
        name: file name
        content/local_path: file content, mutually exclusive; required when adding a new file
        content_base64: if true, content is base64-encoded (for binary files)
        source_type: source type (e.g. cpp.g++17, python.3, java.8); only for source files
        resource_advanced_properties: advanced properties for resource files
    """
    polygon = _get_client()
    file_type = _parse_file_type(type)
    if local_path:
        file_value = _read_local_file(local_path)
    else:
        if content is None:
            raise ValueError("content or local_path is required")
        file_value = _decode_content(content, content_base64)
    adv = _resource_adv_from_dict(resource_advanced_properties)
    result = _call_polygon(
        polygon.problem_save_file,
        problem_id,
        file_type,
        name,
        file_value,
        source_type=source_type,
        resource_advanced_properties=adv,
    )
    return _to_jsonable(result)


def _read_edit_target(
    polygon, problem_id: int, target: str, address: dict[str, str]
) -> tuple[str, str]:
    if target == "file":
        file_type = _parse_file_type(address["type"])
        data = _call_polygon(
            polygon.problem_view_file, problem_id, file_type, address["name"], binary=True
        )
        return _decode_file_content(data), f'{file_type}/{address["name"]}'
    if target == "solution":
        data = _call_polygon(
            polygon.problem_view_solution, problem_id, address["name"], binary=True
        )
        return _decode_file_content(data), f'solution {address["name"]}'
    if target == "script":
        source = _call_polygon(polygon.problem_script, problem_id, address["testset"])
        return source or "", f'script {address["testset"]}'

    section_key = _normalize_statement_section(address["section"])
    statements = _call_polygon(polygon.problem_statements, problem_id)
    statement = statements.get(address["lang"]) if isinstance(statements, dict) else None
    if statement is None:
        raise ValueError(f'Statement not found for lang: {address["lang"]}')
    current = getattr(statement, section_key, None)
    return current or "", f'statement {address["lang"]}/{section_key}'


def _save_edit_target(
    polygon, problem_id: int, target: str, address: dict[str, str], updated: str
) -> Any:
    if target == "file":
        file_type = _parse_file_type(address["type"])
        return _call_polygon(
            polygon.problem_save_file,
            problem_id,
            file_type,
            address["name"],
            updated,
            source_type=address.get("source_type"),
        )
    if target == "solution":
        return _call_polygon(
            polygon.problem_save_solution,
            problem_id,
            address["name"],
            updated,
            None,
            source_type=address.get("source_type"),
        )
    if target == "script":
        return _call_polygon(
            polygon.problem_save_script, problem_id, address["testset"], updated
        )

    section_key = _normalize_statement_section(address["section"])
    return _call_polygon(
        polygon.problem_save_statement,
        problem_id,
        address["lang"],
        Statement(**{section_key: updated}),
    )


@mcp.tool()
def problem_edit(
    problem_id: int,
    target: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    type: Optional[str] = None,
    name: Optional[str] = None,
    lang: Optional[str] = None,
    section: Optional[str] = None,
    testset: Optional[str] = None,
    source_type: Optional[str] = None,
) -> Any:
    """Replace an exact string in a problem's text content and save it back.

    old_string must occur exactly once unless replace_all is true; \\r\\n and \\n
    match interchangeably. target selects what to edit and which addressing
    parameters are required: file (type, name), solution (name), statement
    (lang, section), script (testset).
    """
    normalized_target, address = validate_edit_target(
        target,
        {
            "type": type,
            "name": name,
            "lang": lang,
            "section": section,
            "testset": testset,
            "source_type": source_type,
        },
    )
    polygon = _get_client()
    current, label = _read_edit_target(polygon, problem_id, normalized_target, address)
    if "\x00" in current:
        raise ValueError("Content appears to be binary; edits are not supported")
    updated, spans = apply_string_edit(
        current, old_string, new_string, replace_all, label
    )
    api_result = _save_edit_target(
        polygon, problem_id, normalized_target, address, updated
    )
    return build_edit_response(label, current, updated, spans, _to_jsonable(api_result))


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_solutions(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_solutions, problem_id)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_view_solution(
    problem_id: int,
    name: str,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    binary: bool = False,
) -> Any:
    """Read a solution with bounded line pagination.

    Text must be valid UTF-8. At most 500 lines and 12,000 characters are
    returned per call. Set binary=true to receive the entire file as base64;
    line pagination is unavailable in binary mode.
    """
    polygon = _get_client()
    data = _call_polygon(
        polygon.problem_view_solution, problem_id, name, binary=True
    )
    return _file_content_response(data, start_line, line_count, binary)


@mcp.tool()
def problem_save_solution(
    problem_id: int,
    name: str,
    source_type: Optional[str] = None,
    tag: Optional[str] = None,
    content: Optional[str] = None,
    content_base64: bool = False,
    local_path: Optional[str] = None,
    check_existing: Optional[bool] = None,
) -> Any:
    """Add or edit a solution.

    For edits, all parameters except problem_id and name are optional.

    Parameters:
        check_existing: if true, only adding solutions is allowed
        name: solution name
        content/local_path: solution content (file), mutually exclusive
        source_type: source type
        tag: solution tag (MA - Main, OK - Accepted, RJ - Rejected, TL - Time Limit, TO - Time Limit Exceeded or Accepted, WA - Wrong Answer, PE - Presentation Error, ML - Memory Limit, RE - Runtime Error)
    """
    polygon = _get_client()
    if local_path and content is not None:
        raise ValueError("content and local_path are mutually exclusive")
    if local_path:
        file_value = _read_local_file(local_path)
    elif content is not None:
        file_value = _decode_content(content, content_base64)
    else:
        file_value = None
    result = _call_polygon(
        polygon.problem_save_solution,
        problem_id,
        name,
        file_value,
        source_type,
        tag,
        check_existing=check_existing,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_tests(
    problem_id: int,
    testset: str,
    no_inputs: Optional[bool] = None,
    fields: Optional[list[str]] = None,
    input_line_limit: Optional[int] = None,
    examples_only: bool = False,
) -> Any:
    """List tests for a testset, optionally selecting fields.

    fields can include: testset, index, group, points, description,
    use_in_statements, input_for_statements, output_for_statements,
    verify_input_output_for_statements, input (manual tests only), script_line (generated tests only).
    For each test, only one of input or script_line is present (manual vs generated).
    If input_line_limit is set, returned test inputs are truncated to the first N lines.
    If examples_only is true, only tests with use_in_statements=true are returned.
    """
    if no_inputs is not True:
        raise ValueError("problem_tests requires no_inputs=true; use problem_test_input to fetch test input")
    polygon = _get_client()
    result = _call_polygon(polygon.problem_tests, problem_id, testset, no_inputs=no_inputs)
    data = _to_jsonable(result)
    if examples_only and isinstance(data, list):
        data = [item for item in data if item.get("use_in_statements") is True]
    if fields is None:
        if input_line_limit is None or not isinstance(data, list):
            return data
        for item in data:
            value = item.get("input")
            if isinstance(value, str):
                lines = value.splitlines(keepends=True)
                item["input"] = "".join(lines[: max(0, input_line_limit)])
        return data
    field_set = set(fields)
    if not isinstance(data, list):
        return data
    result = [{k: v for k, v in item.items() if k in field_set} for item in data]
    if input_line_limit is not None and "input" in field_set:
        for item in result:
            value = item.get("input")
            if isinstance(value, str):
                lines = value.splitlines(keepends=True)
                item["input"] = "".join(lines[: max(0, input_line_limit)])
    return result


@mcp.tool()
def problem_test_answer(
    problem_id: int,
    testset: str,
    test_index: int,
    output_path: Optional[str] = None,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    binary: bool = False,
) -> Any:
    """Get generated test answer for a test.

    If output_path is provided, the result is written to a local file.
    Otherwise text must be valid UTF-8 and is limited to 500 lines and 12,000
    characters. Set binary=true to receive the entire answer as base64.
    """
    polygon = _get_client()
    data = _call_polygon(
        polygon.problem_test_answer,
        problem_id,
        testset,
        test_index,
        binary=True,
    )
    if output_path:
        if start_line is not None or line_count is not None or binary:
            raise ValueError(
                "start_line, line_count, and binary cannot be used with output_path"
            )
        path = _resolve_output_path(output_path)
        with open(path, "wb") as handle:
            handle.write(data.encode("utf-8") if isinstance(data, str) else data)
        return {"saved_to": path}
    return _file_content_response(data, start_line, line_count, binary)


@mcp.tool()
def problem_test_input(
    problem_id: int,
    testset: str,
    test_index: int,
    output_path: Optional[str] = None,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
    binary: bool = False,
) -> Any:
    """Get generated test input for a test.

    If output_path is provided, the result is written to a local file.
    Otherwise text must be valid UTF-8 and is limited to 500 lines and 12,000
    characters. Set binary=true to receive the entire input as base64.
    """
    polygon = _get_client()
    data = _call_polygon(
        polygon.problem_test_input,
        problem_id,
        testset,
        test_index,
        binary=True,
    )
    if output_path:
        if start_line is not None or line_count is not None or binary:
            raise ValueError(
                "start_line, line_count, and binary cannot be used with output_path"
            )
        path = _resolve_output_path(output_path)
        with open(path, "wb") as handle:
            handle.write(data.encode("utf-8") if isinstance(data, str) else data)
        return {"saved_to": path}
    return _file_content_response(data, start_line, line_count, binary)


@mcp.tool()
def problem_save_test(
    problem_id: int,
    testset: str,
    test_index: int,
    test_input: Optional[str] = None,
    test_group: Optional[str] = None,
    test_points: Optional[int] = None,
    test_description: Optional[str] = None,
    test_use_in_statements: Optional[bool] = None,
    test_input_for_statements: Optional[str] = None,
    test_output_for_statements: Optional[str] = None,
    verify_input_output_for_statements: Optional[bool] = None,
    check_existing: Optional[bool] = None,
    test_input_base64: bool = False,
) -> Any:
    """Save or update a test.

    test_input is optional when editing; omit it to keep the existing test input
    and update only metadata (group/points/description/statement fields).
    """
    polygon = _get_client()
    input_value = None
    if test_input is not None:
        input_value = _decode_content(test_input, test_input_base64)
    result = _call_polygon(
        polygon.problem_save_test,
        problem_id,
        testset,
        test_index,
        input_value,
        test_group=test_group,
        test_points=test_points,
        test_description=test_description,
        test_use_in_statements=test_use_in_statements,
        test_input_for_statements=test_input_for_statements,
        test_output_for_statements=test_output_for_statements,
        verify_input_output_for_statements=verify_input_output_for_statements,
        check_existing=check_existing,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_script(problem_id: int, testset: str) -> Any:
    """Get the test generation script for a testset.

    The script uses Freemarker-like syntax and references generators (source files)
    to produce tests. Each non-empty line in the script corresponds to a generated test.
    """
    polygon = _get_client()
    data = _call_polygon(polygon.problem_script, problem_id, testset)
    return {"data": data}


@mcp.tool()
def problem_save_script(problem_id: int, testset: str, source: str) -> Any:
    """Save or update the test generation script for a testset.

    The source is the full script content. Each non-empty line typically calls
    a generator, e.g. 'gen 10 20 > $' or uses Freemarker syntax.
    """
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_script, problem_id, testset, source)
    return _to_jsonable(result)


@mcp.tool()
def problem_set_test_group(
    problem_id: int,
    testset: str,
    test_group: str,
    test_index: Optional[int] = None,
    test_indices: Optional[list[int]] = None,
) -> Any:
    """Assign tests to a test group.

    Use test_index for a single test or test_indices for multiple tests.
    At least one of test_index or test_indices must be provided.
    """
    if test_index is None and test_indices is None:
        raise ValueError("At least one of test_index or test_indices must be provided")
    polygon = _get_client()
    result = _call_polygon(
        polygon.problem_set_test_group,
        problem_id,
        testset,
        test_group,
        test_index=test_index,
        test_indices=test_indices,
    )
    return _to_jsonable(result)


@mcp.tool()
def problem_enable_groups(problem_id: int, testset: str, enable: bool) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_enable_groups, problem_id, testset, enable)
    return _to_jsonable(result)


@mcp.tool()
def problem_enable_points(problem_id: int, enable: bool) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_enable_points, problem_id, enable)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_view_test_group(testset: str, group: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_test_group, testset, group)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_test_group(
    problem_id: int,
    testset: str,
    group: str,
    points_policy: Optional[str] = None,
    feedback_policy: Optional[str] = None,
    dependencies: Optional[list] = None,
) -> Any:
    """Save or update a test group.

    points_policy: COMPLETE_GROUP or EACH_TEST
    feedback_policy: NONE, POINTS, ICPC, COMPLETE
    """
    polygon = _get_client()
    points_policy_enum = _parse_enum(PointsPolicy, points_policy) if points_policy is not None else None
    feedback_policy_enum = _parse_enum(FeedbackPolicy, feedback_policy) if feedback_policy is not None else None
    result = _call_polygon(
        polygon.problem_save_test_group,
        problem_id,
        testset,
        group,
        points_policy=points_policy_enum,
        feedback_policy=feedback_policy_enum,
        dependencies=dependencies,
    )
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_view_tags(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_tags, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_tags(problem_id: int, tags: list[str]) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_tags, problem_id, tags)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_view_general_description(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_general_description, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_general_description(problem_id: int, description: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_general_description, problem_id, description)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_view_general_tutorial(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_general_tutorial, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_general_tutorial(problem_id: int, tutorial: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_general_tutorial, problem_id, tutorial)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def contest_problems(contest_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.contest_problems, contest_id)
    return _to_jsonable(result)


@mcp.tool(annotations=READ_ONLY_TOOL_ANNOTATIONS)
def problem_packages(problem_id: int) -> Any:
    """List packages available for the problem."""
    polygon = _get_client()
    result = _call_polygon(polygon.problem_packages, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_package(
    problem_id: int,
    package_id: int,
    output_path: str,
    type: Optional[str] = None,
) -> Any:
    """Download a built package archive to a local file.

    package_id comes from problem_packages and must refer to a READY package.
    type can be standard, linux, or windows; when omitted Polygon picks the
    default for the package. Packages are binary archives, so they are written
    to output_path instead of being returned inline.
    """
    package_type = _parse_enum(PackageType, type)
    requested_type = str(package_type) if package_type is not None else None
    path = _resolve_output_path(output_path)
    polygon = _get_client()
    data = _call_polygon(
        polygon.problem_package,
        problem_id,
        package_id,
        type=requested_type,
    )
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            f"Polygon returned {data.__class__.__name__} instead of package bytes"
        )
    _raise_if_polygon_error_payload(data)
    raw = bytes(data)
    with open(path, "wb") as handle:
        handle.write(raw)
    return {"saved_to": path, "size_bytes": len(raw), "type": requested_type}


@mcp.tool()
def problem_build_package(problem_id: int, full: bool, verify: bool) -> Any:
    """Start building a new package."""
    if full:
        raise ValueError("full packages are disabled; set full=false")
    polygon = _get_client()
    result = _call_polygon(
        polygon.problem_build_package, problem_id, verify=verify, full=full
    )
    return _to_jsonable(result)


if __name__ == "__main__":
    mcp.run()
