from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_demo_debug_video import render_demo_debug_video  # noqa: E402


def _episode_id_from_stem(stem: str) -> int:
    return int(stem.removeprefix("episode_"))


def _task_name_from_annotation(annotation_path: pathlib.Path) -> str:
    with annotation_path.open(encoding="utf-8") as f:
        annotation = json.load(f)
    task_name = annotation.get("task_name")
    if not task_name:
        raise ValueError(f"Missing task_name in {annotation_path}")
    return str(task_name)


def _write_index(output_root: pathlib.Path, records: list[dict]) -> None:
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
        "</style>",
        "</head>",
        "<body>",
        "<h1>BEHAVIOR Demo Subtask Debug Videos</h1>",
        f"<p>{len(records)} rendered records, grouped by task. Overlay shows demo annotation and mapped subtask.</p>",
    ]
    for task_dir in sorted(videos_by_task):
        task_records = sorted(videos_by_task[task_dir], key=lambda item: item["episode"])
        task_name = task_records[0].get("task_name", "unknown")
        lines.append(f"<h2>{task_dir}: {task_name}</h2>")
        lines.append('<div class="grid">')
        for record in task_records:
            video_path = pathlib.Path(record["output_path"])
            rel_path = video_path.relative_to(output_root)
            lines.extend(
                [
                    '<div class="card">',
                    f'<div class="meta">{record["episode"]}</div>',
                    f'<video controls preload="metadata" src="{rel_path.as_posix()}"></video>',
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser()
    output_root = args.output_root.expanduser()
    video_root = output_root / "debug_videos"
    manifest_path = output_root / "manifest.jsonl"
    failures_path = output_root / "failures.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    rendered_ok = set()
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") == "ok":
                    rendered_ok.add(record.get("output_path"))

    annotation_tasks = sorted((dataset_root / "annotations").glob("task-*"))
    jobs = []
    for task_dir in annotation_tasks:
        task_index = int(task_dir.name.removeprefix("task-"))
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

    _write_index(output_root, records)
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
