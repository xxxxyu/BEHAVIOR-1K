from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys
import time

import cv2

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_demo_debug_video import render_demo_debug_video  # noqa: E402
from render_demo_debug_video import CAMERA_DIRS  # noqa: E402
from render_demo_debug_video import _finish_sentence  # noqa: E402
from render_demo_debug_video import _format_demo_segment_lines  # noqa: E402
from render_demo_debug_video import _macro_nav_manipulation_subtasks  # noqa: E402
from render_demo_debug_video import _select_annotation_segments  # noqa: E402
from render_demo_debug_video import _subtask_content  # noqa: E402
from render_demo_debug_video import behavior_subtask_mapping  # noqa: E402


def _episode_id_from_stem(stem: str) -> int:
    return int(stem.removeprefix("episode_"))


def _task_name_from_annotation(annotation_path: pathlib.Path) -> str:
    with annotation_path.open(encoding="utf-8") as f:
        annotation = json.load(f)
    task_name = annotation.get("task_name")
    if not task_name:
        raise ValueError(f"Missing task_name in {annotation_path}")
    return str(task_name)


def _source_frame_count(dataset_root: pathlib.Path, task_dir: str, episode_stem: str) -> int:
    frame_counts = []
    for camera_dir in CAMERA_DIRS.values():
        path = dataset_root / "videos" / task_dir / camera_dir / f"{episode_stem}.mp4"
        cap = cv2.VideoCapture(str(path))
        try:
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()
    if not frame_counts or min(frame_counts) <= 0:
        raise RuntimeError(f"Failed to read source frame count for {task_dir}/{episode_stem}.")
    return min(frame_counts)


def _display_subtask(subtask: str | None) -> str:
    normalized = _finish_sentence(_subtask_content(subtask))
    return normalized if normalized is not None else "[unmapped or excluded]"


def _episode_subtask_rows(
    *,
    dataset_root: pathlib.Path,
    record: dict,
    mapping: behavior_subtask_mapping.BehaviorSubtaskMapping,
    annotation_source: str,
) -> list[dict[str, str]]:
    task_name = record.get("task_name", "unknown")
    task_mapping = mapping.get(task_name)
    if task_mapping is None:
        return []

    task_dir = str(record["task_dir"])
    episode_stem = str(record["episode"])
    annotation_path = dataset_root / "annotations" / task_dir / f"{episode_stem}.json"
    with annotation_path.open(encoding="utf-8") as f:
        annotation = json.load(f)
    frame_count = _source_frame_count(dataset_root, task_dir, episode_stem)
    selected_source, segments = _select_annotation_segments(annotation, annotation_source, frame_count)
    mapped_subtasks = [
        task_mapping.subtask_for_annotation(selected_source, segment, with_prefix=False)
        for segment in segments
    ]
    macro_subtasks = _macro_nav_manipulation_subtasks(mapped_subtasks)

    rows: list[dict[str, str]] = []
    for idx, (segment, mapped, macro) in enumerate(zip(segments, mapped_subtasks, macro_subtasks, strict=True)):
        demo_lines = _format_demo_segment_lines(segment, selected_source)
        frame_start, frame_end = segment["frame_duration"]
        rows.append(
            {
                "idx": str(idx),
                "frames": f"{frame_start}-{frame_end}",
                "demo": " | ".join(demo_lines),
                "mapped": _display_subtask(mapped),
                "macro": _display_subtask(macro),
            }
        )
    return rows


def _write_index(
    output_root: pathlib.Path,
    records: list[dict],
    *,
    dataset_root: pathlib.Path,
    mapping: behavior_subtask_mapping.BehaviorSubtaskMapping,
    annotation_source: str,
    macro_subtasks: bool,
) -> None:
    videos_by_task: dict[str, list[dict]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        videos_by_task.setdefault(record["task_dir"], []).append(record)

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>BEHAVIOR Demo Subtask Debug Videos</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#202124}",
        "h1{font-size:24px} h2{font-size:18px;margin-top:28px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #ddd;border-radius:6px;padding:10px}",
        "video{width:100%;height:auto;background:#111}",
        ".meta{font-size:13px;color:#555;margin:6px 0 8px}",
        "table.subtasks{border-collapse:collapse;width:100%;margin:8px 0 10px;background:#fff}",
        "table.subtasks th,table.subtasks td{border:1px solid #ddd;padding:5px 6px;vertical-align:top;font-size:12px;line-height:1.3}",
        "table.subtasks th{background:#eee}",
        ".demo{color:#555}",
        ".missing{color:#9a3412;font-weight:600}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>BEHAVIOR Demo Subtask Debug Videos</h1>",
        f"<p>{len(records)} rendered records, grouped by task. Each video is preceded by the exact segment-order demo annotation and mapped subtask.</p>",
    ]
    for task_dir in sorted(videos_by_task):
        task_records = sorted(videos_by_task[task_dir], key=lambda item: item["episode"])
        task_name = task_records[0].get("task_name", "unknown")
        lines.append(f"<h2>{html.escape(task_dir)}: {html.escape(task_name)}</h2>")
        lines.append('<div class="grid">')
        for record in task_records:
            video_path = pathlib.Path(record["output_path"])
            rel_path = video_path.relative_to(output_root)
            rows = _episode_subtask_rows(
                dataset_root=dataset_root,
                record=record,
                mapping=mapping,
                annotation_source=annotation_source,
            )
            lines.extend(
                [
                    '<div class="card">',
                    f'<div class="meta">{html.escape(record["episode"])}</div>',
                    '<table class="subtasks">',
                    "<tr><th>#</th><th>frames</th><th>demo annotation</th><th>mapped subtask</th>"
                    + ("<th>macro subtask</th>" if macro_subtasks else "")
                    + "</tr>",
                ]
            )
            for row in rows:
                mapped_cls = "missing" if row["mapped"].startswith("[") else ""
                macro_cls = "missing" if row["macro"].startswith("[") else ""
                row_html = (
                    "<tr>"
                    f"<td>{html.escape(row['idx'])}</td>"
                    f"<td>{html.escape(row['frames'])}</td>"
                    f"<td class=\"demo\">{html.escape(row['demo'])}</td>"
                    f"<td class=\"{mapped_cls}\">{html.escape(row['mapped'])}</td>"
                )
                if macro_subtasks:
                    row_html += f"<td class=\"{macro_cls}\">{html.escape(row['macro'])}</td>"
                row_html += "</tr>"
                lines.append(row_html)
            lines.extend(
                [
                    "</table>",
                    f'<video controls preload="metadata" src="{html.escape(rel_path.as_posix())}"></video>',
                    "</div>",
                ]
            )
        lines.append("</div>")
    lines.extend(["</body>", "</html>"])
    (output_root / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render rollout-style debug videos for first N challenge demos/task.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/lixiangyu/repos/behavior-datasets/2025-challenge-demos"),
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/lixiangyu/repos/behavior-videos/2025-challenge-demos"),
    )
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--annotation-source", choices=("auto", "primitive_annotation", "skill_annotation"), default="auto")
    parser.add_argument("--mapping-path", type=pathlib.Path, default=None)
    parser.add_argument("--prompt-mode", choices=("long_task_subtask", "subtask_only"), default="long_task_subtask")
    parser.add_argument("--macro-subtasks", action="store_true")
    parser.add_argument("--task-indices", default=None, help="Comma-separated task indices to render, e.g. 0,1.")
    parser.add_argument("--high-resolution", action="store_true")
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--hide-prompt", action="store_true")
    parser.add_argument("--overlay-mode", choices=("debug", "compact_subtask"), default="debug")
    parser.add_argument("--target-mb", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser()
    output_root = args.output_root.expanduser()
    mapping = behavior_subtask_mapping.load_behavior_subtask_mapping(
        args.mapping_path.expanduser() if args.mapping_path is not None else None
    )
    video_root = output_root / "debug_videos"
    manifest_path = output_root / "manifest.jsonl"
    failures_path = output_root / "failures.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    rendered_ok = set()
    existing_records_by_output = {}
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") == "ok":
                    rendered_ok.add(record.get("output_path"))
                    existing_records_by_output[record.get("output_path")] = record

    selected_task_indices = None
    if args.task_indices:
        selected_task_indices = {
            int(item.strip().removeprefix("task-")) for item in args.task_indices.split(",") if item.strip()
        }

    annotation_tasks = sorted((dataset_root / "annotations").glob("task-*"))
    jobs = []
    for task_dir in annotation_tasks:
        task_index = int(task_dir.name.removeprefix("task-"))
        if selected_task_indices is not None and task_index not in selected_task_indices:
            continue
        for annotation_path in sorted(task_dir.glob("episode_*.json"))[: args.episodes_per_task]:
            jobs.append((task_index, task_dir.name, annotation_path))

    successes = 0
    failures = 0
    skipped = 0
    records: list[dict] = []
    started = time.time()
    with manifest_path.open("a", encoding="utf-8") as manifest_f, failures_path.open("a", encoding="utf-8") as failures_f:
        for job_index, (task_index, task_dir_name, annotation_path) in enumerate(jobs, start=1):
            episode_stem = annotation_path.stem
            output_path = video_root / task_dir_name / f"{episode_stem}.mp4"
            if output_path.exists() and not args.overwrite and str(output_path) in rendered_ok:
                skipped += 1
                existing_record = existing_records_by_output.get(str(output_path))
                if existing_record is not None:
                    records.append(existing_record)
                continue

            record = {
                "job_index": job_index,
                "num_jobs": len(jobs),
                "task_index": task_index,
                "task_dir": task_dir_name,
                "episode": episode_stem,
                "output_path": str(output_path),
            }
            try:
                task_name = _task_name_from_annotation(annotation_path)
                target_bitrate = None
                if args.target_mb is not None:
                    source_frames = _source_frame_count(dataset_root, task_dir_name, episode_stem)
                    source_duration = source_frames / args.output_fps
                    target_bitrate = int(args.target_mb * 1024 * 1024 * 8 / source_duration * 0.94)
                render_demo_debug_video(
                    dataset_root=dataset_root,
                    output_path=output_path,
                    task_index=task_index,
                    task_name=task_name,
                    episode_id=_episode_id_from_stem(episode_stem),
                    annotation_source=args.annotation_source,
                    stride=args.stride,
                    output_fps=args.output_fps,
                    max_frames=None,
                    mapping_path=args.mapping_path.expanduser() if args.mapping_path is not None else None,
                    prompt_mode=args.prompt_mode,
                    macro_subtasks=args.macro_subtasks,
                    high_resolution=args.high_resolution,
                    lossless=args.lossless,
                    include_prompt=not args.hide_prompt,
                    overlay_mode=args.overlay_mode,
                    target_bitrate=target_bitrate,
                )
                successes += 1
                ok_record = {**record, "task_name": task_name, "status": "ok"}
                records.append(ok_record)
                manifest_f.write(json.dumps(ok_record, sort_keys=True) + "\n")
                manifest_f.flush()
            except Exception as exc:
                failures += 1
                failed_record = {**record, "status": "failed", "error": repr(exc)}
                records.append(failed_record)
                failures_f.write(json.dumps(failed_record, sort_keys=True) + "\n")
                failures_f.flush()
                print(f"[failed] {task_dir_name}/{episode_stem}: {exc}", flush=True)

            if job_index % 10 == 0 or job_index == len(jobs):
                elapsed = time.time() - started
                print(
                    f"[progress] {job_index}/{len(jobs)} done, ok={successes}, failed={failures}, "
                    f"skipped={skipped}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

    _write_index(
        output_root,
        records,
        dataset_root=dataset_root,
        mapping=mapping,
        annotation_source=args.annotation_source,
        macro_subtasks=args.macro_subtasks,
    )
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "successes": successes,
                "failures": failures,
                "skipped": skipped,
                "output_root": str(output_root),
                "index_path": str(output_root / "index.html"),
                "manifest_path": str(manifest_path),
                "failures_path": str(failures_path),
                "elapsed_s": round(time.time() - started, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
