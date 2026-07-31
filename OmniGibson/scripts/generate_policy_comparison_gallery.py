#!/usr/bin/env python3
"""Build a symlink-only task-only/subtask-only video review gallery."""

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
from pathlib import Path


TASKS = [
    ("turning_on_radio", "Turning on the radio"),
    ("picking_up_trash", "Picking up trash"),
    ("moving_boxes_to_storage", "Moving boxes to storage"),
    ("putting_shoes_on_rack", "Putting shoes on rack"),
    ("hanging_pictures", "Hanging pictures"),
    ("make_microwave_popcorn", "Making microwave popcorn"),
    ("putting_away_Halloween_decorations", "Putting away Halloween decorations"),
    ("spraying_for_bugs", "Spraying for bugs"),
    ("bringing_water", "Bringing water"),
    ("cook_hot_dogs", "Cooking hot dogs"),
    ("bringing_in_wood", "Bringing in wood"),
    ("spraying_fruit_trees", "Spraying fruit trees"),
    ("tidying_bedroom", "Tidying bedroom"),
    ("carrying_in_groceries", "Carrying in groceries"),
]

# Only retained runs compatible with the selected final progress contract are
# eligible. Formal minimal-output runs intentionally have no videos.
SOURCES = {
    ("moving_boxes_to_storage", "task_only"): (
        "moving_boxes_taskonly399999_50rollout_baseline_20260705",
        "*task_moving_boxes_to_storage__repeat0/videos/*.mp4",
        "formal 50-rollout repeat0",
    ),
    ("hanging_pictures", "subtask_only"): (
        "pictures_popcorn_bugs_stage_threshold_sweep_subtask399999_20260711",
        "*task_hanging_pictures__default_repeat1/videos/*.mp4",
        "formal setting, representative repeat1",
    ),
    ("make_microwave_popcorn", "subtask_only"): (
        "popcorn_f085_demoaligned_factorial_subtask399999_20260714",
        "*task_make_microwave_popcorn__f085_O_repeat0/videos/*.mp4",
        "selected formal setting, repeat0",
    ),
    ("putting_away_Halloween_decorations", "subtask_only"): (
        "halloween_support_eef_sweep_subtask399999_20260709",
        "*task_putting_away_Halloween_decorations__pickup_eef050_support_eef050_repeat0/videos/*.mp4",
        "selected formal setting, repeat0",
    ),
    ("spraying_for_bugs", "subtask_only"): (
        "bugs_completion_latch_fixed2_probe_subtask399999_20260713",
        "*task_spraying_for_bugs__bugs_eef050_completion_latched_fixed2_repeat0/videos/*.mp4",
        "selected formal setting, repeat0",
    ),
    ("bringing_water", "subtask_only"): (
        "bringing_water_placementlivefix_probe399999_20260716",
        "*task_bringing_water__repeat0/videos/*.mp4",
        "selected generic-corrected contract, repeat0",
    ),
    ("cook_hot_dogs", "subtask_only"): (
        "hot_dogs_contractfix_probe399999_20260715",
        "*task_cook_hot_dogs__repeat0/videos/*.mp4",
        "selected generic-current contract, repeat0",
    ),
    ("bringing_in_wood", "task_only"): (
        "trumpet_wood_contractfix_probe399999_20260716",
        "*taskonly400k_step399999__task_bringing_in_wood__repeat0/videos/*.mp4",
        "formal setting, repeat0",
    ),
    ("bringing_in_wood", "subtask_only"): (
        "trumpet_wood_contractfix_probe399999_20260716",
        "*subtask400k_contractfix_step399999__task_bringing_in_wood__repeat0/videos/*.mp4",
        "formal setting, repeat0",
    ),
    ("tidying_bedroom", "task_only"): (
        "tidying_bedlocal_taskonly_confirmation399999_20260720",
        "*/videos/*.mp4",
        "representative final-checkpoint diagnostics",
    ),
    ("tidying_bedroom", "subtask_only"): (
        "tidying_bed_eefaabb030_fix_probe399999_20260720",
        "*/videos/*.mp4",
        "selected bed EEF-AABB 0.30 contract",
    ),
    ("carrying_in_groceries", "subtask_only"): (
        "groceries_subtask_50rollout_releaseguard2_399999_20260723",
        "*/videos/*.mp4",
        "selected releaseguard2 contract",
    ),
}

GAPFILL_ROOT = "video_comparison_gapfill399999_20260727"


def episode_id(path: Path) -> str:
    match = re.search(r"_(\d+)_\d+(?:_success)?\.mp4$", path.name)
    return match.group(1) if match else path.stem


def repeat_id(path: Path) -> str:
    match = re.search(r"repeat(\d+)", str(path.parent.parent))
    return match.group(1) if match else "-"


def collect(source_root: Path, output_root: Path):
    records = []
    for task_key, task_name in TASKS:
        for mode in ("task_only", "subtask_only"):
            mode_label = "taskonly400k_video" if mode == "task_only" else "subtask400k_video"
            sources = [
                (
                    GAPFILL_ROOT,
                    f"*__{mode_label}_step399999__task_{task_key}__repeat0/videos/*.mp4",
                    "dedicated matched-instance gap-fill repeat0",
                )
            ]
            retained_source = SOURCES.get((task_key, mode))
            if retained_source is not None:
                sources.append(retained_source)
            seen = set()
            for root_name, pattern, note in sources:
                for source_path in sorted((source_root / root_name).glob(pattern)):
                    instance = episode_id(source_path)
                    if instance in seen:
                        continue
                    seen.add(instance)
                    relative = Path("videos") / task_key / mode / f"instance_{instance}.mp4"
                    link = output_root / relative
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(source_path.resolve())
                    records.append(
                        {
                            "task": task_key,
                            "task_name": task_name,
                            "mode": mode,
                            "instance": instance,
                            "repeat": repeat_id(source_path),
                            "path": relative.as_posix(),
                            "version": str(source_path.stat().st_mtime_ns),
                            "source": str(source_path),
                            "note": note,
                        }
                    )
                    if len(seen) == 10:
                        break
                if len(seen) == 10:
                    break
    return records


def generate_posters(records, output_root: Path, *, mode: str, workers: int) -> None:
    if mode == "off":
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("warning: ffmpeg is unavailable; gallery will use lazy video placeholders without posters")
        return

    def generate(record: dict) -> tuple[dict, str | None]:
        relative = Path("posters") / record["task"] / record["mode"] / f"instance_{record['instance']}.jpg"
        destination = output_root / relative
        source = Path(record["source"])
        if mode != "refresh" and destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return record, relative.as_posix()

        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "1",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=360:-2",
            "-q:v",
            "5",
            "-y",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"warning: failed to generate poster for {source}")
            destination.unlink(missing_ok=True)
            return record, None
        return record, relative.as_posix()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for record, poster in executor.map(generate, records):
            if poster is not None:
                record["poster"] = poster


def write_browser_data(records, output_root: Path) -> list[dict]:
    browser_fields = ("task", "mode", "instance", "repeat", "path", "poster", "version", "note")
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    task_summaries = []
    for task_key, task_name in TASKS:
        task_records = [record for record in records if record["task"] == task_key]
        browser_records = [{key: record[key] for key in browser_fields if key in record} for record in task_records]
        data_path = data_root / f"{task_key}.json"
        data_path.write_text(json.dumps(browser_records, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
        task_ids = {record["instance"] for record in task_records if record["mode"] == "task_only"}
        subtask_ids = {record["instance"] for record in task_records if record["mode"] == "subtask_only"}
        version = max((int(record["version"]) for record in task_records), default=1)
        task_summaries.append(
            {
                "key": task_key,
                "name": task_name,
                "task_only": len(task_ids),
                "subtask_only": len(subtask_ids),
                "paired": len(task_ids & subtask_ids),
                "data": f"data/{task_key}.json?v={version}",
            }
        )
    return task_summaries


def render(task_summaries):
    tasks = json.dumps(task_summaries, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task-only vs subtask-only video review</title>
<style>
:root{{--ink:#17201c;--muted:#65706a;--line:#dfe4e1;--paper:#f7f8f6;--white:#fff;--task:#25745a;--sub:#b85234}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;background:rgba(247,248,246,.96);border-bottom:1px solid var(--line);padding:18px 24px}}
.head{{max-width:1500px;margin:auto;display:flex;gap:20px;align-items:end;justify-content:space-between}}h1{{font-size:22px;margin:0 0 3px}}p{{margin:0;color:var(--muted)}}.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end}}select{{min-width:250px;padding:9px 34px 9px 11px;border:1px solid #bfc8c3;border-radius:5px;background:#fff;color:var(--ink)}}.metrics-link{{color:var(--ink);text-decoration:none;border-bottom:1px solid #9ca7a1;padding:7px 1px 5px}}.metrics-link:hover{{color:var(--task);border-color:var(--task)}}
.segments{{display:flex;border:1px solid #bfc8c3;border-radius:5px;overflow:hidden;background:#fff}}.segments button{{border:0;border-right:1px solid #d5dcd8;background:#fff;color:var(--muted);padding:9px 11px;cursor:pointer}}.segments button:last-child{{border-right:0}}.segments button.active{{background:var(--ink);color:#fff}}
main{{max-width:1500px;margin:auto;padding:22px 24px 60px}}.task{{margin:0 0 34px}}h2{{font-size:18px;margin:0 0 10px}}.status{{font-size:12px;color:var(--muted);margin-left:8px;font-weight:400}}
	.task{{border-bottom:1px solid var(--line);margin:0 0 12px;padding:0 0 12px}}.task summary{{cursor:pointer;list-style:none;display:flex;align-items:center;gap:9px;padding:8px 0}}.task summary h2{{margin:0}}.task summary::-webkit-details-marker{{display:none}}.task summary::before{{content:'';width:8px;height:8px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(-45deg);transition:transform .16s ease;margin-left:2px}}.task[open] summary::before{{transform:rotate(45deg)}}.task-body{{padding-top:2px}}
	.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;content-visibility:auto;contain-intrinsic-size:760px 450px}}.panel{{background:var(--white);border:1px solid var(--line);border-radius:7px;overflow:hidden;min-width:0}}.panel.missing{{display:flex;min-height:260px;align-items:center;justify-content:center;color:var(--muted);background:#fbfcfb}}
		.label{{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px}}.mode{{font-weight:650}}.task-only .mode{{color:var(--task)}}.subtask-only .mode{{color:var(--sub)}}.media-shell{{position:relative;background:#101311;aspect-ratio:16/9;overflow:hidden}}.poster,video{{display:block;width:100%;height:100%;object-fit:contain;background:#101311}}.poster-missing{{width:100%;height:100%;background:#151a17}}.load-video{{position:absolute;inset:0;margin:auto;width:max-content;height:40px;padding:0 17px;border:1px solid rgba(255,255,255,.72);border-radius:6px;background:rgba(12,16,14,.82);color:#fff;font:600 13px/1 Inter,ui-sans-serif,system-ui,sans-serif;letter-spacing:0;cursor:pointer;box-shadow:0 3px 18px rgba(0,0,0,.28)}}.load-video:hover{{background:rgba(37,116,90,.94)}}.meta{{padding:9px 12px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
.empty{{padding:26px;border:1px dashed #bdc7c1;border-radius:7px;color:var(--muted);background:#fff}}code{{font-family:ui-monospace,SFMono-Regular,monospace;background:#edf0ee;border-radius:3px;padding:1px 4px}}
@media(max-width:800px){{.head{{display:block}}.controls{{margin-top:12px;justify-content:flex-start}}select{{width:100%}}.pair{{grid-template-columns:1fr}}header,main{{padding-left:14px;padding-right:14px}}}}
</style></head><body><header><div class="head"><div><h1>Task-only vs subtask-only</h1><p>Checkpoint 399999. Final-contract retained videos; missing formal videos are called out explicitly.</p></div><div class="controls"><a class="metrics-link" href="metrics.html">50-rollout metrics</a><div class="segments" aria-label="Comparison coverage"><button class="active" data-scope="all">All</button><button data-scope="paired_tasks">Paired tasks</button><button data-scope="paired_instances">Paired instances</button></div><select id="filter"></select></div></div></header><main id="app"></main>
	<script>
		const tasks={tasks};const filter=document.querySelector('#filter');
		let scope='all';const expanded=new Set();const taskCache=new Map();
		filter.innerHTML='<option value="all">All tasks</option>'+tasks.map(t=>`<option value="${{t.key}}">${{t.name}}</option>`).join('');
		const esc=s=>String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
		function versioned(path,row){{return `${{path}}?v=${{encodeURIComponent(row.version||'1')}}`}}
		function panel(row,mode){{if(!row)return `<div class="panel missing">No retained final-contract ${{mode.replace('_','-')}} video</div>`;const poster=row.poster?versioned(row.poster,row):'';const preview=poster?`<img class="poster" loading="lazy" decoding="async" src="${{esc(poster)}}" alt="">`:'<div class="poster-missing"></div>';return `<article class="panel ${{mode.replace('_','-')}}"><div class="label"><span class="mode">${{mode.replace('_','-')}}</span><span>instance ${{esc(row.instance)}} · repeat ${{esc(row.repeat)}}</span></div><div class="media-shell">${{preview}}<button class="load-video" type="button" data-src="${{esc(versioned(row.path,row))}}" data-poster="${{esc(poster)}}">Press to play</button></div><div class="meta">${{esc(row.note)}}</div></article>`}}
		function indexTask(t,rows){{const byMode={{task_only:new Map(),subtask_only:new Map()}};rows.forEach(row=>byMode[row.mode].set(row.instance,row));const allIds=[...new Set(rows.map(row=>row.instance))].sort((a,b)=>Number(a)-Number(b));const pairedIds=allIds.filter(id=>byMode.task_only.has(id)&&byMode.subtask_only.has(id));return {{t,byMode,allIds,pairedIds}}}}
		async function loadTask(t){{if(!taskCache.has(t.key))taskCache.set(t.key,fetch(t.data,{{cache:'no-cache'}}).then(response=>{{if(!response.ok)throw new Error(`HTTP ${{response.status}}`);return response.json()}}).then(rows=>indexTask(t,rows)));return taskCache.get(t.key)}}
		function idsFor(d){{return scope==='paired_instances'?d.pairedIds:d.allIds}}
		function taskBody(d){{const ids=idsFor(d);return ids.length?ids.map(id=>`<div class="pair">${{panel(d.byMode.task_only.get(id),'task_only')}}${{panel(d.byMode.subtask_only.get(id),'subtask_only')}}</div>`).join(''):'<div class="empty">No retained final-contract videos. The formal evaluation used <code>--minimal-output</code>.</div>'}}
		function activate(button){{const shell=button.parentElement;const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=button.dataset.src;if(button.dataset.poster)video.poster=button.dataset.poster;shell.replaceChildren(video);video.play().catch(()=>{{}})}}
		function prepareMedia(root){{root.querySelectorAll('.load-video').forEach(button=>button.addEventListener('click',()=>activate(button)))}}
		async function mountTask(details,t){{const body=details.querySelector('.task-body');if(body.dataset.mounted==='1'||body.dataset.loading==='1')return;body.dataset.loading='1';body.innerHTML='<div class="empty">Loading task videos...</div>';try{{const d=await loadTask(t);if(!details.isConnected||!details.open)return;body.innerHTML=taskBody(d);body.dataset.mounted='1';prepareMedia(body)}}catch(error){{if(details.isConnected&&details.open)body.innerHTML=`<div class="empty">Unable to load task videos: ${{esc(error.message)}}</div>`}}finally{{body.dataset.loading='0'}}}}
		function unmountTask(details){{const body=details.querySelector('.task-body');body.querySelectorAll('video').forEach(video=>{{video.pause();video.removeAttribute('src');video.load()}});body.replaceChildren();body.dataset.mounted='0';body.dataset.loading='0'}}
		function unloadVideos(){{document.querySelectorAll('video').forEach(video=>{{video.pause();video.removeAttribute('src');video.load()}})}}
		function render(){{unloadVideos();const chosen=filter.value;const visible=tasks.filter(t=>(chosen==='all'||t.key===chosen)&&(scope==='all'||(scope==='paired_tasks'&&t.task_only&&t.subtask_only)||(scope==='paired_instances'&&t.paired)));document.querySelector('#app').innerHTML=visible.map(t=>`<details class="task" data-task="${{t.key}}" ${{expanded.has(t.key)?'open':''}}><summary><h2>${{t.name}}<span class="status">${{t.task_only}} task-only · ${{t.subtask_only}} subtask-only · ${{t.paired}} paired</span></h2></summary><div class="task-body"></div></details>`).join('')||'<div class="empty">No comparisons match this filter yet.</div>';document.querySelectorAll('details.task').forEach(details=>{{const t=visible.find(item=>item.key===details.dataset.task);details.addEventListener('toggle',()=>{{if(details.open){{expanded.add(t.key);mountTask(details,t)}}else{{expanded.delete(t.key);unmountTask(details)}}}});if(details.open)mountTask(details,t)}})}}
	document.querySelectorAll('[data-scope]').forEach(button=>button.addEventListener('click',()=>{{scope=button.dataset.scope;document.querySelectorAll('[data-scope]').forEach(b=>b.classList.toggle('active',b===button));render()}}));
	filter.addEventListener('change',()=>{{expanded.clear();if(filter.value!=='all')expanded.add(filter.value);render()}});render();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poster-mode", choices=("auto", "off", "refresh"), default="auto")
    parser.add_argument("--poster-workers", type=int, default=4)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = collect(args.source_root, args.output_root)
    generate_posters(records, args.output_root, mode=args.poster_mode, workers=args.poster_workers)
    (args.output_root / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    task_summaries = write_browser_data(records, args.output_root)
    (args.output_root / "index.html").write_text(render(task_summaries), encoding="utf-8")
    metrics_source = args.source_root.parents[2] / "docs" / "taskonly_vs_subtask_50rollout_comparison_20260710.html"
    metrics_link = args.output_root / "metrics.html"
    if metrics_source.exists():
        if metrics_link.is_symlink() or metrics_link.exists():
            metrics_link.unlink()
        metrics_link.symlink_to(metrics_source.resolve())
    serve_script = args.output_root / "serve.sh"
    serve_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "port=${1:-18080}\n"
        'root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'server={str((args.source_root.parents[1] / "OmniGibson/scripts/serve_policy_comparison_gallery.py").resolve())!r}\n'
        'exec python3 "$server" --root "$root" --port "$port" --bind 127.0.0.1\n',
        encoding="utf-8",
    )
    serve_script.chmod(0o755)
    (args.output_root / "README.md").write_text(
        "# Policy comparison video review\n\n"
        "On GD, run:\n\n"
        "```bash\n./serve.sh 18080\n```\n\n"
        "On the MacBook, create an SSH tunnel:\n\n"
        "```bash\n"
        "ssh -N -L 18080:127.0.0.1:18080 -p 43274 "
        "root@guangdong-b-is.cloud.infini-ai.com\n"
        "```\n\n"
        "Then open <http://127.0.0.1:18080>. The gallery contains symlinks to "
        "retained GD outputs, so serve it from its generated directory.\n",
        encoding="utf-8",
    )
    print(f"generated {args.output_root} with {len(records)} linked videos")


if __name__ == "__main__":
    main()
