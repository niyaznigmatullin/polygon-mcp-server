"""Addressing rules and response shape for problem_edit."""

from typing import Any, Optional

from .text_edit import count_lines

ADDRESS_PARAMS = ("type", "name", "lang", "section", "testset", "source_type")

EDIT_TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "file": {"required": ("type", "name"), "optional": ("source_type",)},
    "solution": {"required": ("name",), "optional": ("source_type",)},
    "statement": {"required": ("lang", "section"), "optional": ()},
    "script": {"required": ("testset",), "optional": ()},
}


def validate_edit_target(
    target: str, params: dict[str, Optional[str]]
) -> tuple[str, dict[str, str]]:
    """Normalize the target name and keep only the parameters it addresses with.

    A parameter belonging to another target is an error rather than a silently
    ignored field, so a misaddressed call fails instead of editing the wrong
    object.
    """
    normalized = str(target or "").strip().lower()
    if normalized not in EDIT_TARGETS:
        valid = ", ".join(sorted(EDIT_TARGETS))
        raise ValueError(f"Unknown target '{target}'. Use one of: {valid}.")

    spec = EDIT_TARGETS[normalized]
    allowed = set(spec["required"]) | set(spec["optional"])
    missing = [key for key in spec["required"] if params.get(key) is None]
    forbidden = [
        key for key in ADDRESS_PARAMS if key not in allowed and params.get(key) is not None
    ]
    if missing or forbidden:
        not_allowed = [key for key in ADDRESS_PARAMS if key not in allowed]
        raise ValueError(
            f'target="{normalized}" requires {", ".join(spec["required"])}; '
            f'{", ".join(not_allowed)} are not allowed.'
        )

    return normalized, {
        key: params[key]
        for key in ADDRESS_PARAMS
        if key in allowed and params.get(key) is not None
    }


def build_edit_response(
    label: str,
    current: str,
    updated: str,
    spans: list[tuple[int, int, int]],
    api_result: Any,
) -> dict[str, Any]:
    """Describe where the replacements landed in the saved content."""
    response: dict[str, Any] = {
        "target": label,
        "replacements": len(spans),
        "lines_before": count_lines(current),
        "lines_after": count_lines(updated),
    }
    if len(spans) == 1:
        _, post_start_line, post_end_line = spans[0]
        response["start_line"] = post_start_line
        response["end_line"] = post_end_line
    else:
        response["replacement_lines"] = [span[1] for span in spans]
    if api_result:
        response["api_result"] = api_result
    return response
