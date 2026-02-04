import base64
import io
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from enum import Enum
from typing import Any, Optional

import patch_ng
from fastmcp import FastMCP
from polygon_api import (
    FeedbackPolicy,
    FileType,
    PointsPolicy,
    Polygon,
    PolygonRequestFailedException,
    ProblemInfo,
    ResourceAdvancedProperties,
    Statement,
)

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

_LOGGER = logging.getLogger("polygon_mcp")
if not _LOGGER.handlers:
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log_path = os.getenv("POLYGON_MCP_LOG_FILE")
    if not log_path:
        state_home = os.getenv("XDG_STATE_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "state"
        )
        log_path = os.path.join(state_home, "polygon-mcp", "polygon-mcp.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8"):
            pass
        os.chmod(log_path, 0o600)
    except OSError:
        pass
    _handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    _handler.setFormatter(_formatter)
    _LOGGER.addHandler(_handler)
_LOGGER.setLevel(logging.INFO)


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


def _read_local_file(path: str) -> bytes:
    if not path:
        raise ValueError("local_path is empty")
    with open(path, "rb") as handle:
        return handle.read()


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


def _slice_lines(text: str | bytes, start_line: Optional[int], line_count: Optional[int]) -> str | bytes:
    if start_line is None and line_count is None:
        return text
    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Line slicing is supported only for UTF-8 files") from exc
    if start_line is None:
        start_line = 1
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if line_count is not None and line_count < 0:
        raise ValueError("line_count must be >= 0")
    lines = text.splitlines(keepends=True)
    start_index = start_line - 1
    end_index = None if line_count is None else start_index + line_count
    return "".join(lines[start_index:end_index])


def _apply_line_edit(text: str, start_line: int, line_count: int, replacement: str) -> str:
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if line_count < 0:
        raise ValueError("line_count must be >= 0")
    lines = text.splitlines(keepends=True)
    start_index = start_line - 1
    if start_index > len(lines):
        raise ValueError("start_line is beyond end of file")
    end_index = start_index + line_count
    if end_index > len(lines):
        raise ValueError("line_count goes beyond end of file")
    new_lines = lines[:start_index] + [replacement] + lines[end_index:]
    return "".join(new_lines)


def _parse_unified_diff(patch_text: str) -> list[dict]:
    lines = patch_text.splitlines(keepends=True)
    hunks: list[dict] = []
    i = 0
    file_headers_seen = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            file_headers_seen += 1
            if file_headers_seen > 1:
                raise ValueError("Only single-file patches are supported")
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise ValueError("Invalid patch: missing '+++' header")
            i += 1
            continue
        if line.startswith("@@ "):
            header = line
            i += 1
            # @@ -l,s +l,s @@
            try:
                meta = header.strip().split()
                old_part = meta[1]  # -l,s
                new_part = meta[2]  # +l,s
                old_nums = old_part[1:].split(",")
                new_nums = new_part[1:].split(",")
                start_old = int(old_nums[0])
                len_old = int(old_nums[1]) if len(old_nums) > 1 else 1
                start_new = int(new_nums[0])
                len_new = int(new_nums[1]) if len(new_nums) > 1 else 1
            except Exception as exc:
                raise ValueError(f"Invalid hunk header: {header.strip()}") from exc

            hunk_lines: list[str] = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith("@@ "):
                    break
                if next_line.startswith("--- ") or next_line.startswith("+++ "):
                    break
                if not next_line.startswith((" ", "+", "-", "\\")):
                    raise ValueError(f"Invalid patch line: {next_line!r}")
                hunk_lines.append(next_line)
                i += 1
            hunks.append(
                {
                    "start_old": start_old,
                    "len_old": len_old,
                    "start_new": start_new,
                    "len_new": len_new,
                    "lines": hunk_lines,
                }
            )
            continue
        i += 1

    if not hunks:
        raise ValueError("Patch contains no hunks")
    return hunks


def _apply_unified_diff(text: str, patch_text: str, expected_name: Optional[str] = None) -> str:
    patch_bytes = patch_text.encode("utf-8")
    if not patch_bytes.endswith(b"\n"):
        patch_bytes += b"\n"
    patch_set = patch_ng.fromstring(patch_bytes)
    if not patch_set or patch_set.errors:
        details = ""
        if patch_set and getattr(patch_set, "errors", None):
            details = f" (errors={patch_set.errors})"
        raise ValueError(f"Invalid patch format{details}")
    if len(patch_set.items) != 1:
        raise ValueError("Only single-file patches are supported")

    item = patch_set.items[0]
    source = item.source.decode("utf-8", errors="ignore") if isinstance(item.source, (bytes, bytearray)) else str(item.source)
    target = item.target.decode("utf-8", errors="ignore") if isinstance(item.target, (bytes, bytearray)) else str(item.target)
    patch_name = os.path.basename(target or source)
    if expected_name and patch_name and patch_name != expected_name:
        raise ValueError(f"Patch file name '{patch_name}' does not match expected '{expected_name}'")

    instream = io.BytesIO(text.encode("utf-8"))
    patched_bytes = b"".join(patch_set.patch_stream(instream, item.hunks))
    return patched_bytes.decode("utf-8")


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


@mcp.tool()
def problems_list(
    show_deleted: Optional[bool] = None,
    id: Optional[int] = None,
    name: Optional[str] = None,
    owner: Optional[str] = None,
) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problems_list, show_deleted=show_deleted, id=id, name=name, owner=owner)
    return _to_jsonable(result)


@mcp.tool()
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
    time_limit: Optional[float] = None,
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def problem_checker(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_checker, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_validator(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_validator, problem_id)
    return _to_jsonable(result)


@mcp.tool()
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


@mcp.tool()
def problem_files(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_files, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_view_file(
    problem_id: int,
    type: str,
    name: str,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
) -> Any:
    polygon = _get_client()
    file_type = _parse_file_type(type)
    data = _call_polygon(polygon.problem_view_file, problem_id, file_type, name)
    data = _slice_lines(data, start_line, line_count)
    return {"data": data, "encoding": "utf-8"}


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


@mcp.tool()
def problem_patch_file(
    problem_id: int,
    type: str,
    name: str,
    patch: str,
    source_type: Optional[str] = None,
    resource_advanced_properties: Optional[dict] = None,
) -> Any:
    """Apply a unified diff (single-file) patch to a text file and save it back.

    The server reads the current file, applies the patch, and saves it back.
    If the patch doesn't apply, it returns an error.
    """
    polygon = _get_client()
    file_type = _parse_file_type(type)
    current = _call_polygon(polygon.problem_view_file, problem_id, file_type, name)
    if not isinstance(current, str):
        raise ValueError("File content is not text; patch edits are not supported")
    if "\x00" in current:
        raise ValueError("File appears to be binary; patch edits are not supported")
    updated = _apply_unified_diff(current, patch, expected_name=name)
    if updated == current:
        raise ValueError("Patch did not change file content")
    adv = _resource_adv_from_dict(resource_advanced_properties)
    result = _call_polygon(
        polygon.problem_save_file,
        problem_id,
        file_type,
        name,
        updated,
        source_type=source_type,
        resource_advanced_properties=adv,
    )
    return _to_jsonable(result)


@mcp.tool()
def problem_patch_statement(
    problem_id: int,
    lang: str,
    section: str,
    patch: str,
) -> Any:
    """Apply a unified diff patch to a statement section and save it back.

    The server reads the current statement, applies the patch to the selected
    section, and saves it back. If the patch doesn't apply, it returns an error.
    Sections: legend, input, output, notes, tutorial, scoring, interaction.
    """
    polygon = _get_client()
    section_key = _normalize_statement_section(section)
    try:
        statements = _call_polygon(polygon.problem_statements, problem_id)
        statement = statements.get(lang) if isinstance(statements, dict) else None
        if statement is None:
            raise ValueError(f"Statement not found for lang: {lang}")
        current = getattr(statement, section_key, None)
        current_text = current or ""
        updated = _apply_unified_diff(current_text, patch, expected_name=section_key)
        if updated == current_text:
            raise ValueError("Patch did not change statement section")
        statement_patch = Statement(**{section_key: updated})
        result = _call_polygon(polygon.problem_save_statement, problem_id, lang, statement_patch)
    except Exception as exc:
        raise
    return _to_jsonable(result)


@mcp.tool()
def problem_solutions(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_solutions, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_view_solution(
    problem_id: int,
    name: str,
    start_line: Optional[int] = None,
    line_count: Optional[int] = None,
) -> Any:
    polygon = _get_client()
    data = _call_polygon(polygon.problem_view_solution, problem_id, name)
    data = _slice_lines(data, start_line, line_count)
    return {"data": data, "encoding": "utf-8"}


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


@mcp.tool()
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
) -> Any:
    """Get generated test answer for a test.

    If output_path is provided, the result is written to a local file.
    """
    polygon = _get_client()
    data = _call_polygon(polygon.problem_test_answer, problem_id, testset, test_index)
    if output_path:
        path = _resolve_output_path(output_path)
        with open(path, "wb") as handle:
            handle.write(data.encode("utf-8") if isinstance(data, str) else data)
        return {"saved_to": path}
    return {"data": data, "encoding": "utf-8"}


@mcp.tool()
def problem_test_input(
    problem_id: int,
    testset: str,
    test_index: int,
    output_path: Optional[str] = None,
) -> Any:
    """Get generated test input for a test.

    If output_path is provided, the result is written to a local file.
    """
    polygon = _get_client()
    data = _call_polygon(polygon.problem_test_input, problem_id, testset, test_index)
    if output_path:
        path = _resolve_output_path(output_path)
        with open(path, "wb") as handle:
            handle.write(data.encode("utf-8") if isinstance(data, str) else data)
        return {"saved_to": path}
    return {"data": data, "encoding": "utf-8"}


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


@mcp.tool()
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


@mcp.tool()
def problem_view_tags(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_tags, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_tags(problem_id: int, tags: list[str]) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_tags, problem_id, tags)
    return _to_jsonable(result)


@mcp.tool()
def problem_view_general_description(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_general_description, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_general_description(problem_id: int, description: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_general_description, problem_id, description)
    return _to_jsonable(result)


@mcp.tool()
def problem_view_general_tutorial(problem_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_view_general_tutorial, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_save_general_tutorial(problem_id: int, tutorial: str) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.problem_save_general_tutorial, problem_id, tutorial)
    return _to_jsonable(result)


@mcp.tool()
def contest_problems(contest_id: int) -> Any:
    polygon = _get_client()
    result = _call_polygon(polygon.contest_problems, contest_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_packages(problem_id: int) -> Any:
    """List packages available for the problem."""
    polygon = _get_client()
    result = _call_polygon(polygon.problem_packages, problem_id)
    return _to_jsonable(result)


@mcp.tool()
def problem_build_package(problem_id: int, full: bool, verify: bool) -> Any:
    """Start building a new package."""
    if full:
        raise ValueError("full packages are disabled; set full=false")
    polygon = _get_client()
    result = _call_polygon(polygon.problem_build_package, problem_id, full, verify)
    return _to_jsonable(result)


if __name__ == "__main__":
    mcp.run()
