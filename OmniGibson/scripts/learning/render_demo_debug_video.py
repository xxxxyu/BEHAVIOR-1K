from __future__ import annotations

import argparse
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
sys.path.insert(0, str(REPO_ROOT / "OmniGibson"))
sys.path.insert(0, str(OPENPI_BEHAVIOR_ROOT / "src"))

from omnigibson.learning.utils.debug_video_utils import DEBUG_VIDEO_RESOLUTION  # noqa: E402
from omnigibson.learning.utils.debug_video_utils import build_debug_video_frame  # noqa: E402
from omnigibson.learning.utils.debug_video_utils import resize_debug_video_views  # noqa: E402
from openpi.shared import behavior_subtask_mapping  # noqa: E402
from openpi.shared import task_progress as task_progress_lib  # noqa: E402


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


def _segment_for_frame(segments: list[dict], frame_idx: int) -> dict:
    for segment in segments:
        start, end = segment["frame_duration"]
        if start <= frame_idx < end:
            return segment
    return segments[-1]


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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=output_fps)
        stream.height = DEBUG_VIDEO_RESOLUTION[0]
        stream.width = DEBUG_VIDEO_RESOLUTION[1]
        stream.pix_fmt = "yuv420p"

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

                segment = _segment_for_frame(segments, frame_idx)
                segment_id = tuple(segment["frame_duration"])
                if segment_id != last_segment_id:
                    last_segment_id = segment_id

                mapped_subtask = task_mapping.subtask_for_annotation(
                    selected_annotation_source,
                    segment,
                    with_prefix=False,
                )
                prompt = task_progress_lib.inject_subtask_block(task_prompt, mapped_subtask)
                left_wrist, right_wrist, head = resize_debug_video_views(
                    left_wrist_rgb=left_wrist,
                    right_wrist_rgb=right_wrist,
                    head_rgb=head,
                )
                frame = build_debug_video_frame(
                    left_wrist_rgb=left_wrist,
                    right_wrist_rgb=right_wrist,
                    head_rgb=head,
                    header_lines=[
                        f"task: {task_name} | episode: {episode_id} | frame: {frame_idx}",
                    ],
                    context_label="primitive/skill (from demo)",
                    context_value=None,
                    context_lines=_format_demo_segment_lines(segment, selected_annotation_source),
                    mapped_subtask=mapped_subtask,
                    prompt=prompt,
                )
                _write_frame(container, stream, frame)
                written += 1
        finally:
            _close_writer(container, stream)

        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "resolution_hw": DEBUG_VIDEO_RESOLUTION,
                    "source_frame_counts": frame_counts,
                    "written_frames": written,
                    "requested_annotation_source": annotation_source,
                    "selected_annotation_source": selected_annotation_source,
                    "stride": stride,
                    "output_fps": output_fps,
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
    )


if __name__ == "__main__":
    main()
