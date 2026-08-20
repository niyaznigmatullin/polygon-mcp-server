"""Exact-string editing of text content, tolerant of CRLF line endings."""


def count_lines(text: str) -> int:
    """Count lines by newline occurrences.

    str.splitlines is deliberately avoided: it also breaks on a lone CR, \\x0b
    and \\x0c, which would disagree with the newline-based line numbers reported
    alongside this count.
    """
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def normalize_newlines_with_map(text: str) -> tuple[str, list[int]]:
    """Drop each CR that precedes an LF, keeping a map to original offsets."""
    chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(text):
        if char == "\r" and text[index + 1 : index + 2] == "\n":
            continue
        chars.append(char)
        index_map.append(index)
    return "".join(chars), index_map


def find_string_matches(norm: str, old_norm: str) -> list[int]:
    """Offsets of non-overlapping occurrences, left to right."""
    matches: list[int] = []
    start = 0
    while True:
        found = norm.find(old_norm, start)
        if found < 0:
            return matches
        matches.append(found)
        start = found + len(old_norm)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def apply_string_edit(
    text: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    label: str,
) -> tuple[str, list[tuple[int, int, int]]]:
    """Replace old_string with new_string, matching exactly modulo CRLF.

    Returns the updated text and a (pre_start_line, post_start_line,
    post_end_line) span per replacement. Line endings outside the replaced
    regions are preserved byte for byte; inside them they follow the text being
    replaced. label names the edited object in error messages.
    """
    if old_string == "":
        raise ValueError(
            "old_string must not be empty; use the matching save tool to create content"
        )
    if old_string == new_string:
        raise ValueError("old_string and new_string are identical; nothing to change")

    norm, index_map = normalize_newlines_with_map(text)
    old_norm, _ = normalize_newlines_with_map(old_string)
    new_norm, _ = normalize_newlines_with_map(new_string)

    matches = find_string_matches(norm, old_norm)
    if not matches:
        raise ValueError(
            f"old_string not found in {label} ({count_lines(norm)} lines). "
            "Check whitespace, indentation and line endings, or re-read the current content."
        )
    if len(matches) > 1 and not replace_all:
        lines = ", ".join(str(line_of(norm, offset)) for offset in matches)
        raise ValueError(
            f"Found {len(matches)} matches of old_string in {label} at lines {lines}. "
            "Add surrounding context to make it unique, or pass replace_all=true."
        )

    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    newlines_emitted = 0
    for offset in matches:
        orig_start = index_map[offset]
        orig_end = index_map[offset + len(old_norm) - 1] + 1
        gap = text[cursor:orig_start]
        pieces.append(gap)
        newlines_emitted += gap.count("\n")

        matched = text[orig_start:orig_end]
        replacement = new_norm.replace("\n", "\r\n") if "\r\n" in matched else new_norm
        replacement_newlines = replacement.count("\n")

        post_start_line = newlines_emitted + 1
        post_end_line = post_start_line + replacement_newlines
        if replacement.endswith("\n"):
            post_end_line -= 1
        spans.append((line_of(norm, offset), post_start_line, post_end_line))

        pieces.append(replacement)
        newlines_emitted += replacement_newlines
        cursor = orig_end

    pieces.append(text[cursor:])
    return "".join(pieces), spans
