import json
from typing import Any

import cv2
import numpy as np


DEBUG_VIDEO_TEXT_HEIGHT = 180
DEBUG_VIDEO_HEAD_SIZE = 360
DEBUG_VIDEO_WRIST_SIZE = 180
DEBUG_VIDEO_WIDTH = DEBUG_VIDEO_WRIST_SIZE + DEBUG_VIDEO_HEAD_SIZE
DEBUG_VIDEO_HEIGHT = DEBUG_VIDEO_TEXT_HEIGHT + DEBUG_VIDEO_HEAD_SIZE
DEBUG_VIDEO_RESOLUTION = (DEBUG_VIDEO_HEIGHT, DEBUG_VIDEO_WIDTH)
DEBUG_VIDEO_TEXT_MAX_CHARS = max(20, int((DEBUG_VIDEO_WIDTH - 20) / 8.0))
DEBUG_VIDEO_CONTEXT_BUDGET_LINES = 3


def json_compact(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(value)


def wrap_text(text: str, *, max_chars: int) -> list[str]:
    lines = []
    for raw_line in str(text).replace("\r", "").split("\n"):
        words = raw_line.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                lines.append(current)
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            current = word
        lines.append(current)
    return lines


def ellipsize_text(text: str, *, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3].rstrip()}..."


def fixed_budget_text_lines(
    lines: list[str],
    *,
    budget: int,
    max_chars: int = DEBUG_VIDEO_TEXT_MAX_CHARS,
    wrap: bool = True,
) -> list[str]:
    output = []
    truncated = False
    for line in lines:
        pieces = wrap_text(line, max_chars=max_chars) if wrap else [str(line)]
        for piece in pieces:
            if len(output) >= budget:
                truncated = True
                break
            output.append(ellipsize_text(piece, max_chars=max_chars))
        if truncated:
            break

    if truncated and output:
        output[-1] = ellipsize_text(f"{output[-1]}...", max_chars=max_chars)
    while len(output) < budget:
        output.append("")
    return output[:budget]


def draw_wrapped_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    max_width: int,
    max_y: int,
    font_scale: float = 0.45,
    line_height: int = 18,
    color: tuple[int, int, int] = (235, 235, 235),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    max_chars = max(20, int(max_width / 8.0))
    x, y = origin
    for text in lines:
        for wrapped in wrap_text(text, max_chars=max_chars):
            if y > max_y:
                cv2.putText(frame, "...", (x, max_y), font, font_scale, color, thickness, cv2.LINE_AA)
                return
            cv2.putText(frame, wrapped, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height


def resize_debug_video_views(
    *,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    head_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        cv2.resize(left_wrist_rgb[..., :3], (DEBUG_VIDEO_WRIST_SIZE, DEBUG_VIDEO_WRIST_SIZE)),
        cv2.resize(right_wrist_rgb[..., :3], (DEBUG_VIDEO_WRIST_SIZE, DEBUG_VIDEO_WRIST_SIZE)),
        cv2.resize(head_rgb[..., :3], (DEBUG_VIDEO_HEAD_SIZE, DEBUG_VIDEO_HEAD_SIZE)),
    )


def build_debug_video_frame(
    *,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    head_rgb: np.ndarray,
    header_lines: list[str],
    context_label: str,
    context_value: Any,
    mapped_subtask: str | None,
    prompt: str | None,
    context_lines: list[str] | None = None,
    context_budget_lines: int = DEBUG_VIDEO_CONTEXT_BUDGET_LINES,
) -> np.ndarray:
    camera_panel = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])
    text_panel = np.full((DEBUG_VIDEO_TEXT_HEIGHT, DEBUG_VIDEO_WIDTH, 3), 18, dtype=np.uint8)
    lines = build_debug_video_text_lines(
        header_lines=header_lines,
        context_label=context_label,
        context_value=context_value,
        mapped_subtask=mapped_subtask,
        prompt=prompt,
        context_lines=context_lines,
        context_budget_lines=context_budget_lines,
    )
    draw_wrapped_text(
        text_panel,
        lines,
        origin=(10, 22),
        max_width=DEBUG_VIDEO_WIDTH - 20,
        max_y=DEBUG_VIDEO_TEXT_HEIGHT - 12,
    )
    return np.vstack([text_panel, camera_panel])


def build_debug_video_text_lines(
    *,
    header_lines: list[str],
    context_label: str,
    context_value: Any,
    mapped_subtask: str | None,
    prompt: str | None,
    context_lines: list[str] | None = None,
    context_budget_lines: int = DEBUG_VIDEO_CONTEXT_BUDGET_LINES,
) -> list[str]:
    if context_lines is None:
        raw_context_lines = [f"{context_label}: {json_compact(context_value)}"]
        wrap_context = True
    else:
        raw_context_lines = context_lines
        wrap_context = False

    return [
        *(ellipsize_text(line, max_chars=DEBUG_VIDEO_TEXT_MAX_CHARS) for line in header_lines),
        *fixed_budget_text_lines(
            raw_context_lines,
            budget=context_budget_lines,
            wrap=wrap_context,
        ),
        ellipsize_text(
            f"subtask (mapped): {mapped_subtask if mapped_subtask else 'n/a'}",
            max_chars=DEBUG_VIDEO_TEXT_MAX_CHARS,
        ),
        "===prompt===",
        prompt if prompt else "n/a",
    ]
