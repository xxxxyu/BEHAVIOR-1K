from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
from typing import Any

import av
import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OPENPI_BEHAVIOR_ROOT = pathlib.Path(os.environ.get("OPENPI_BEHAVIOR_ROOT", "/home/lixiangyu/repos/openpi-behavior"))
sys.path.insert(0, str(OPENPI_BEHAVIOR_ROOT / "src"))

from openpi.shared import behavior_subtask_mapping  # noqa: E402
from openpi.shared import task_progress as task_progress_lib  # noqa: E402

DEBUG_VIDEO_UTILS_PATH = REPO_ROOT / "OmniGibson" / "omnigibson" / "learning" / "utils" / "debug_video_utils.py"
_debug_video_spec = importlib.util.spec_from_file_location("behavior_debug_video_utils", DEBUG_VIDEO_UTILS_PATH)
if _debug_video_spec is None or _debug_video_spec.loader is None:
    raise ImportError(f"Failed to load debug video utilities from {DEBUG_VIDEO_UTILS_PATH}")
_debug_video_utils = importlib.util.module_from_spec(_debug_video_spec)
_debug_video_spec.loader.exec_module(_debug_video_utils)

DEBUG_VIDEO_RESOLUTION = _debug_video_utils.DEBUG_VIDEO_RESOLUTION
DEBUG_VIDEO_TEXT_HEIGHT = _debug_video_utils.DEBUG_VIDEO_TEXT_HEIGHT
build_debug_video_frame = _debug_video_utils.build_debug_video_frame
resize_debug_video_views = _debug_video_utils.resize_debug_video_views


CAMERA_DIRS = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    return [str(value)]


def _clean_object_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return ""
    parts = text.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return " ".join(parts)


def _format_demo_segment_lines(segment: dict, source: str) -> list[str]:
    action_key = "primitive_description" if source == "primitive_annotation" else "skill_description"
    action = " ".join(_flatten_strings(segment.get(action_key))).strip()
    objects = [_clean_object_name(name) for name in _flatten_strings(segment.get("object_id"))]
    objects = [name for name in objects if name]
    manipulating = [_clean_object_name(name) for name in _flatten_strings(segment.get("manipulating_object_id"))]
    manipulating = [name for name in manipulating if name]
    lines = [f"primitive/skill (from demo): action={action or 'n/a'}"]
    if manipulating:
        lines.append(f"manipulating={', '.join(manipulating)}")
    if objects:
        lines.append(f"objects={', '.join(objects)}")
    return lines


def _normalize_frame_range(frame_duration: Any, frame_count: int) -> list[int] | None:
    if not isinstance(frame_duration, list | tuple):
        return None

    values: list[int] = []
    stack = [frame_duration]
    while stack:
        item = stack.pop()
        if isinstance(item, list | tuple):
            stack.extend(item)
        else:
            values.append(int(item))
    if len(values) < 2:
        return None

    start_frame, end_frame_exclusive = min(values), max(values)
    if start_frame < 0 or start_frame >= frame_count:
        return None
    end_frame_exclusive = min(end_frame_exclusive, frame_count)
    if end_frame_exclusive <= start_frame:
        return None
    return [start_frame, end_frame_exclusive]


def _valid_annotation_segments(annotation: dict, source: str, frame_count: int) -> list[dict] | None:
    raw_segments = annotation.get(source)
    if not isinstance(raw_segments, list) or not raw_segments:
        return None
    segments = []
    for segment in raw_segments:
        frame_duration = _normalize_frame_range(segment.get("frame_duration"), frame_count)
        if frame_duration is None:
            continue
        segments.append({**segment, "frame_duration": frame_duration})
    return segments or None


def _select_annotation_segments(annotation: dict, requested_source: str, frame_count: int) -> tuple[str, list[dict]]:
    sources = ("skill_annotation", "primitive_annotation") if requested_source == "auto" else (requested_source,)
    for source in sources:
        segments = _valid_annotation_segments(annotation, source, frame_count)
        if segments:
            return source, segments
    raise ValueError(f"No valid annotation segments found for source={requested_source!r}, frame_count={frame_count}.")


def _segment_for_frame(segments: list[dict], frame_idx: int) -> dict | None:
    for segment in segments:
        start, end = segment["frame_duration"]
        if start <= frame_idx < end:
            return segment
    return None


def _segment_index_for_frame(segments: list[dict], frame_idx: int) -> int | None:
    for idx, segment in enumerate(segments):
        start, end = segment["frame_duration"]
        if start <= frame_idx < end:
            return idx
    return None


def _subtask_content(subtask: str | None) -> str | None:
    if subtask is None:
        return None
    text = " ".join(str(subtask).strip().split())
    if text.startswith("Subtask:"):
        text = text.removeprefix("Subtask:").strip()
    return text.rstrip(".!?").strip() or None


def _finish_sentence(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(text.strip().split())
    if not text:
        return None
    return text if text[-1] in ".!?" else f"{text}."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _is_navigation_subtask(subtask: str | None) -> bool:
    content = _subtask_content(subtask)
    return bool(content and content.startswith("Move to "))


def _macro_nav_manipulation_subtasks(mapped_subtasks: list[str | None]) -> list[str | None]:
    macro_subtasks = list(mapped_subtasks)
    idx = 0
    while idx + 1 < len(mapped_subtasks):
        current = mapped_subtasks[idx]
        next_subtask = mapped_subtasks[idx + 1]
        current_content = _subtask_content(current)
        next_content = _subtask_content(next_subtask)
        if current_content and next_content and _is_navigation_subtask(current) and not _is_navigation_subtask(next_subtask):
            macro = _finish_sentence(f"{current_content} and {_lower_first(next_content)}")
            macro_subtasks[idx] = macro
            macro_subtasks[idx + 1] = macro
            idx += 2
            continue
        idx += 1
    return macro_subtasks


def _build_prompt(task_prompt: str, mapped_subtask: str | None, prompt_mode: str) -> str:
    if prompt_mode == "subtask_only":
        return _subtask_content(mapped_subtask) or ""
    if prompt_mode == "long_task_subtask":
        return task_progress_lib.inject_subtask_block(task_prompt, mapped_subtask)
    raise ValueError(f"Unsupported debug prompt_mode={prompt_mode!r}.")


def _read_rgb(cap: cv2.VideoCapture, *, camera_name: str, frame_idx: int) -> np.ndarray:
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read {camera_name} frame {frame_idx}.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _grab_rgb(cap: cv2.VideoCapture, *, camera_name: str, frame_idx: int) -> None:
    ok = cap.grab()
    if not ok:
        raise RuntimeError(f"Failed to grab {camera_name} frame {frame_idx}.")


def _write_frame(container: av.container.OutputContainer, stream: av.video.stream.VideoStream, frame: np.ndarray) -> None:
    video_frame = av.VideoFrame.from_ndarray(frame[..., :3], format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def _close_writer(container: av.container.OutputContainer, stream: av.video.stream.VideoStream) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _target_debug_sizes(caps: dict[str, cv2.VideoCapture], *, high_resolution: bool) -> tuple[int, int, int]:
    if not high_resolution:
        return DEBUG_VIDEO_RESOLUTION[0], DEBUG_VIDEO_RESOLUTION[1], DEBUG_VIDEO_TEXT_HEIGHT

    head_width = int(caps["head"].get(cv2.CAP_PROP_FRAME_WIDTH))
    head_height = int(caps["head"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    head_size = min(head_width, head_height)
    if head_size <= 0:
        raise RuntimeError("Failed to read head camera resolution.")
    wrist_size = max(1, head_size // 2)
    text_height = 104
    return text_height + head_size, wrist_size + head_size, text_height


def _resize_views_for_mode(
    *,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    head_rgb: np.ndarray,
    high_resolution: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not high_resolution:
        return resize_debug_video_views(
            left_wrist_rgb=left_wrist_rgb,
            right_wrist_rgb=right_wrist_rgb,
            head_rgb=head_rgb,
        )

    head_size = min(head_rgb.shape[0], head_rgb.shape[1])
    wrist_size = max(1, head_size // 2)
    return (
        cv2.resize(left_wrist_rgb[..., :3], (wrist_size, wrist_size)),
        cv2.resize(right_wrist_rgb[..., :3], (wrist_size, wrist_size)),
        cv2.resize(head_rgb[..., :3], (head_size, head_size)),
    )


def _display_subtask(mapped_subtask: str | None) -> str:
    content = _subtask_content(mapped_subtask)
    return content if content is not None else ""


def render_demo_debug_video(
    *,
    dataset_root: pathlib.Path,
    output_path: pathlib.Path,
    task_index: int,
    task_name: str,
    episode_id: int,
    annotation_source: str,
    stride: int,
    output_fps: int,
    max_frames: int | None,
    mapping_path: pathlib.Path | None,
    prompt_mode: str,
    macro_subtasks: bool,
    high_resolution: bool = False,
    lossless: bool = False,
    include_prompt: bool = True,
    overlay_mode: str = "debug",
    target_bitrate: int | None = None,
) -> None:
    task_dir = f"task-{task_index:04d}"
    episode_stem = f"episode_{episode_id:08d}"
    annotation_path = dataset_root / "annotations" / task_dir / f"{episode_stem}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    with annotation_path.open(encoding="utf-8") as f:
        annotation = json.load(f)

    mapping = behavior_subtask_mapping.load_behavior_subtask_mapping(mapping_path)
    task_mapping = mapping.require(task_name)
    task_prompt = task_mapping.task_prompt or annotation.get("task_name") or task_name

    video_paths = {
        camera: dataset_root / "videos" / task_dir / camera_dir / f"{episode_stem}.mp4"
        for camera, camera_dir in CAMERA_DIRS.items()
    }
    missing = [str(path) for path in video_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing camera videos: {missing}")

    caps = {camera: cv2.VideoCapture(str(path)) for camera, path in video_paths.items()}
    try:
        frame_counts = {camera: int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for camera, cap in caps.items()}
        frame_count = min(frame_counts.values())
        if max_frames is not None:
            frame_count = min(frame_count, max_frames * stride)
        selected_annotation_source, segments = _select_annotation_segments(annotation, annotation_source, frame_count)
        mapped_subtasks = [
            task_mapping.subtask_for_annotation(
                selected_annotation_source,
                segment,
                with_prefix=False,
            )
            for segment in segments
        ]
        if macro_subtasks:
            mapped_subtasks = _macro_nav_manipulation_subtasks(mapped_subtasks)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_height, output_width, text_height = _target_debug_sizes(caps, high_resolution=high_resolution)
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=output_fps)
        stream.height = output_height
        stream.width = output_width
        if lossless:
            stream.pix_fmt = "yuv444p"
            stream.options = {"crf": "0", "preset": "veryslow"}
        else:
            stream.pix_fmt = "yuv420p"
            if target_bitrate is not None:
                stream.bit_rate = target_bitrate
                stream.options = {"preset": "slow", "profile": "high"}

        written = 0
        last_segment_id: tuple[int, int] | None = None
        try:
            for frame_idx in range(frame_count):
                if frame_idx % stride:
                    _grab_rgb(caps["head"], camera_name="head", frame_idx=frame_idx)
                    _grab_rgb(caps["left_wrist"], camera_name="left_wrist", frame_idx=frame_idx)
                    _grab_rgb(caps["right_wrist"], camera_name="right_wrist", frame_idx=frame_idx)
                    continue

                head = _read_rgb(caps["head"], camera_name="head", frame_idx=frame_idx)
                left_wrist = _read_rgb(caps["left_wrist"], camera_name="left_wrist", frame_idx=frame_idx)
                right_wrist = _read_rgb(caps["right_wrist"], camera_name="right_wrist", frame_idx=frame_idx)

                segment_idx = _segment_index_for_frame(segments, frame_idx)
                segment = segments[segment_idx] if segment_idx is not None else None
                segment_id = tuple(segment["frame_duration"]) if segment is not None else None
                if segment_id != last_segment_id:
                    last_segment_id = segment_id

                mapped_subtask = mapped_subtasks[segment_idx] if segment_idx is not None else None
                prompt = _build_prompt(task_prompt, mapped_subtask, prompt_mode)
                left_wrist, right_wrist, head = _resize_views_for_mode(
                    left_wrist_rgb=left_wrist,
                    right_wrist_rgb=right_wrist,
                    head_rgb=head,
                    high_resolution=high_resolution,
                )
                frame = build_debug_video_frame(
                    left_wrist_rgb=left_wrist,
                    right_wrist_rgb=right_wrist,
                    head_rgb=head,
                    header_lines=[
                        f"task: {task_name} | episode: {episode_id} | frame: {frame_idx}",
                        f"prompt_mode: {prompt_mode} | macro_subtasks: {macro_subtasks}",
                    ],
                    context_label="primitive/skill (from demo)",
                    context_value=None,
                    context_lines=(
                        _format_demo_segment_lines(segment, selected_annotation_source)
                        if segment is not None
                        else []
                    ),
                    mapped_subtask=mapped_subtask,
                    prompt=prompt if include_prompt else None,
                    text_height=text_height,
                    include_prompt=include_prompt,
                    compact_header=(
                        f"task-{task_index:04d}: {task_name} | episode: {episode_id:08d} | timestep/frame: {frame_idx}"
                        if overlay_mode == "compact_subtask"
                        else None
                    ),
                    compact_subtask=(
                        f"subtask: {_display_subtask(mapped_subtask)}"
                        if overlay_mode == "compact_subtask"
                        else None
                    ),
                )
                _write_frame(container, stream, frame)
                written += 1
        finally:
            _close_writer(container, stream)

        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "resolution_hw": [output_height, output_width],
                    "source_frame_counts": frame_counts,
                    "written_frames": written,
                    "requested_annotation_source": annotation_source,
                    "selected_annotation_source": selected_annotation_source,
                    "stride": stride,
                    "output_fps": output_fps,
                    "prompt_mode": prompt_mode,
                    "macro_subtasks": macro_subtasks,
                    "high_resolution": high_resolution,
                    "lossless": lossless,
                    "include_prompt": include_prompt,
                    "overlay_mode": overlay_mode,
                    "target_bitrate": target_bitrate,
                },
                sort_keys=True,
            )
        )
    finally:
        for cap in caps.values():
            cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render challenge demo video with rollout-style debug text.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/lixiangyu/repos/behavior-datasets/2025-challenge-demos"),
    )
    parser.add_argument("--output-path", type=pathlib.Path, default=pathlib.Path("outputs/demo_debug_video.mp4"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-name", default="turning_on_radio")
    parser.add_argument("--episode-id", type=int, default=10)
    parser.add_argument(
        "--annotation-source",
        choices=("auto", "primitive_annotation", "skill_annotation"),
        default="auto",
    )
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum written frames after striding.")
    parser.add_argument("--mapping-path", type=pathlib.Path, default=None)
    parser.add_argument("--prompt-mode", choices=("long_task_subtask", "subtask_only"), default="long_task_subtask")
    parser.add_argument("--macro-subtasks", action="store_true")
    parser.add_argument("--high-resolution", action="store_true")
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--hide-prompt", action="store_true")
    parser.add_argument("--overlay-mode", choices=("debug", "compact_subtask"), default="debug")
    parser.add_argument("--target-bitrate", type=int, default=None)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.output_fps < 1:
        raise ValueError("--output-fps must be >= 1")
    render_demo_debug_video(
        dataset_root=args.dataset_root.expanduser(),
        output_path=args.output_path.expanduser(),
        task_index=args.task_index,
        task_name=args.task_name,
        episode_id=args.episode_id,
        annotation_source=args.annotation_source,
        stride=args.stride,
        output_fps=args.output_fps,
        max_frames=args.max_frames,
        mapping_path=args.mapping_path.expanduser() if args.mapping_path is not None else None,
        prompt_mode=args.prompt_mode,
        macro_subtasks=args.macro_subtasks,
        high_resolution=args.high_resolution,
        lossless=args.lossless,
        include_prompt=not args.hide_prompt,
        overlay_mode=args.overlay_mode,
        target_bitrate=args.target_bitrate,
    )


if __name__ == "__main__":
    main()
