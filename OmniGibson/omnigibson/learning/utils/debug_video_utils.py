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
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_chars = max(20, int(max_width / 8.0))
    x, y = origin
    for text in lines:
        for wrapped in wrap_text(text, max_chars=max_chars):
            if y > max_y:
                cv2.putText(frame, "...", (x, max_y), font, font_scale, color, thickness, cv2.LINE_AA)
                return
            cv2.putText(frame, wrapped, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height


def draw_compact_subtask_text(
    frame: np.ndarray,
    *,
    header: str,
    subtask: str,
    origin: tuple[int, int] = (16, 34),
    max_width: int | None = None,
    max_y: int | None = None,
) -> None:
    max_width = max_width if max_width is not None else frame.shape[1] - 32
    max_y = max_y if max_y is not None else frame.shape[0] - 16
    draw_wrapped_text(
        frame,
        [header, subtask],
        origin=origin,
        max_width=max_width,
        max_y=max_y,
        font_scale=0.78,
        line_height=34,
        color=(245, 245, 245),
        thickness=2,
    )


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
    text_height: int = DEBUG_VIDEO_TEXT_HEIGHT,
    include_prompt: bool = True,
    compact_header: str | None = None,
    compact_subtask: str | None = None,
) -> np.ndarray:
    camera_panel = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])
    width = int(camera_panel.shape[1])
    text_panel = np.full((text_height, width, 3), 18, dtype=np.uint8)
    if compact_header is not None:
        draw_compact_subtask_text(
            text_panel,
            header=compact_header,
            subtask=compact_subtask or "subtask: n/a",
            max_width=width - 32,
            max_y=text_height - 12,
        )
    else:
        lines = build_debug_video_text_lines(
            header_lines=header_lines,
            context_label=context_label,
            context_value=context_value,
            mapped_subtask=mapped_subtask,
            prompt=prompt,
            context_lines=context_lines,
            context_budget_lines=context_budget_lines,
            max_chars=max(20, int((width - 20) / 8.0)),
            include_prompt=include_prompt,
        )
        draw_wrapped_text(
            text_panel,
            lines,
            origin=(10, 22),
            max_width=width - 20,
            max_y=text_height - 12,
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
    max_chars: int = DEBUG_VIDEO_TEXT_MAX_CHARS,
    include_prompt: bool = True,
) -> list[str]:
    if context_lines is None:
        raw_context_lines = [f"{context_label}: {json_compact(context_value)}"]
        wrap_context = True
    else:
        raw_context_lines = context_lines
        wrap_context = False

    lines = [
        *(ellipsize_text(line, max_chars=max_chars) for line in header_lines),
        *fixed_budget_text_lines(
            raw_context_lines,
            budget=context_budget_lines,
            max_chars=max_chars,
            wrap=wrap_context,
        ),
        ellipsize_text(
            f"subtask (mapped): {mapped_subtask if mapped_subtask else 'n/a'}",
            max_chars=max_chars,
        ),
    ]
    if include_prompt:
        lines.extend(["===prompt===", prompt if prompt else "n/a"])
    return lines
