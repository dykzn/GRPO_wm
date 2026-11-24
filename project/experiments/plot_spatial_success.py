"""Plot audited real LIBERO-Spatial success rates and world-model update events.

CPU-only report generation. Training/evaluation/queue source files and live run
state are read but never changed. Each plotted point retains its source path
and source digest in the exported CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT = Path(__file__).resolve().parents[1]
EARLY = "spatial_wovr20_fast_pi05_12000_8gpu/20261020_131300_node146"
MIDDLE = "spatial_wovr20_shared_pool_resume10_8gpu/20261021_061423_node146"
UPDATED = "spatial_reconstructedwm_resume20_8gpu/20261023_113225_node147/grpo_primary"
ITERATION = "spatial_iterative_policy40_8gpu/20261024_091807_node147"
WM20 = "spatial_policy20_500_20261022_052054_node146_8gpu"
WM40 = "spatial_policy40_500_20261024_091807_node147"
COLORS = dict(original="#2463AD", updated="#D76C28", next="#8B5BB1", event="#159477", branch="#808B9A")


def read(path):
    return json.loads(Path(path).read_text())


def json_file(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_run(root, series):
    import yaml

    if not (root / "resolved_config.yaml").is_file():
        return []
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    evaluation = config["env"]["eval"]
    protocol = {key:evaluation.get(key) for key in ["task_suite_name", "episodes_per_task", "seed",
        "max_episode_steps", "use_fixed_reset_state_ids", "use_ordered_reset_state_ids"]}
    assert protocol == dict(task_suite_name="libero_spatial", episodes_per_task=50, seed=0,
        max_episode_steps=240, use_fixed_reset_state_ids=True, use_ordered_reset_state_ids=True), (root,protocol)
    points = []
    for path in sorted(root.glob("*/evaluations/global_step_*/metrics.json")):
        raw = path.read_bytes()
        value = json.loads(raw)
        rows = value["episode_results"]
        keys = {(row["task_id"],row["trial_id"]) for row in rows}
        assert len(rows) == len(keys) == 500
        assert keys == {(task,trial) for task in range(10) for trial in range(50)}
        success = sum(bool(row["success"]) for row in rows)
        step = int(path.parent.name.removeprefix("global_step_"))
        assert value["global_step"] == step and value["episodes"] == 500
        assert value["max_episode_steps"] == 240 and value["success"] == success
        assert abs(value["success_rate"] - success/500) < 1e-12
        per_task = {str(task):sum(bool(row["success"]) for row in rows if row["task_id"]==task)
                    for task in range(10)}
        for task, count in per_task.items():
            assert value["per_task"][task]["episodes"] == 50
            assert value["per_task"][task]["success"] == count
        points.append(dict(series=series, global_step=step, success=success, episodes=500,
            success_rate_pct=success/5, failure=500-success,
            phase="original_world_models_fallback" if series!="main" else
                  "original_world_models" if step <= 20 else "after_update_1" if step <= 40 else "after_update_2",
            checkpoint=value["checkpoint"], source_metrics=str(path),
            source_sha256=hashlib.sha256(raw).hexdigest(), protocol=protocol,
            resume_dir=config["runner"].get("resume_dir"), per_task_success=per_task))
    return points


def gather(project):
    runs = project.parent / "grpo_runs"
    points = []
    for name in (EARLY, MIDDLE, UPDATED, ITERATION + "/grpo_primary"):
        points.extend(load_run(runs / name, "main"))
    points.sort(key=lambda row:row["global_step"])
    assert len({row["global_step"] for row in points}) == len(points), "Conflicting mainline checkpoints"
    by_step = {row["global_step"]:row for row in points}
    for step, parent in [(15,10),(25,20),(45,40)]:
        if step in by_step:
            assert Path(by_step[step]["resume_dir"]).resolve() == Path(by_step[parent]["checkpoint"]).resolve()
    fallback = project / "runtime_artifacts/followup_chains/spatial_after20_20261021_node146/grpo_fallback"
    branch = load_run(fallback,"old_wm_fallback")
    if branch:
        assert all(Path(row["resume_dir"]).resolve()==Path(by_step[20]["checkpoint"]).resolve() for row in branch)
    events = []
    for number,step,folder in [(1,20,WM20),(2,40,WM40)]:
        wm = project / "wm_updates" / folder
        collection = read(wm / "collection/collection.json")
        manifest = read(wm / "manifest.json")
        assert collection["summary"]["episodes"] == len(collection["episodes"]) == 500
        policy = Path(collection["plan"]["policy_checkpoint"]).parents[2]
        assert policy.resolve() == Path(by_step[step]["checkpoint"]).resolve()
        events.append(dict(update=number, after_policy_step=step, new_episodes=500,
            replay_episodes=0 if number==1 else 500,
            collection_mode=collection["plan"]["mode"], collection_success=collection["summary"]["success"],
            training_episodes=manifest["summary"]["train"]["episodes"],
            validation_episodes=manifest["summary"]["val"]["episodes"],
            status="complete_and_used_for_policy_25_40" if number==1 else "training_in_progress_no_new_policy_evaluation",
            wan_steps_used=1400 if number==1 else None, cosmos_steps_used=2000 if number==1 else None,
            source_collection=str(wm / "collection/collection.json"), source_manifest=str(wm / "manifest.json")))
    return points,branch,events


def export_data(output, main, branch, events):
    columns = ["series", "global_step", "success_rate_pct", "success", "episodes", "failure", "phase",
               "source_metrics", "source_sha256", "checkpoint", "resume_dir"]
    with (output / "success_rates.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(main+branch)
    with (output / "per_task_success.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream);writer.writerow(["series","global_step","task_id","success","episodes","success_rate_pct"])
        for row in main+branch:
            for task,count in row["per_task_success"].items():
                writer.writerow([row["series"],row["global_step"],task,count,50,count*2])
    payload=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        metric="real LIBERO-Spatial simulator success rate", mainline=main, other_branches=branch,
        world_model_updates=events, protocol=main[0]["protocol"],
        audit=dict(complete_unique_500_episodes_per_point=True, ten_tasks_fifty_trials_each=True,
                   counts_recomputed_from_episode_results=True, resume_lineage_verified=True,
                   collection_success_is_not_used_as_policy_evaluation=True),
        notes=["Only recorded real simulator evaluation points are plotted; no interpolation beyond the observed steps.",
               "The old-world-model fallback is a separate policy branch from step 20.",
               "Policy step 0 has no local evaluation record and is not plotted."])
    json_file(output / "success_rates.json",payload)


def plot(output, main, branch, font):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    if font is None or not font.is_file():
        raise FileNotFoundError("Use --font with a complete Chinese TTF/OTF font")
    font_manager.fontManager.addfont(str(font))
    family=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family":family,"font.size":12,"axes.unicode_minus":False,
        "svg.fonttype":"path","pdf.fonttype":42,"savefig.facecolor":"white"})
    fig=plt.figure(figsize=(14.6,8.6),facecolor="white")
    fig.text(.075,.951,"LIBERO-Spatial 真实成功率",fontsize=23,weight="bold",color="#172B4D")
    fig.text(.075,.911,"每点 10 个任务 × 50 次 = 500 次真实仿真评估  ·  固定初始状态  ·  最大部署 240 步",
             fontsize=12,color="#536477")
    ax=fig.add_axes([.075,.365,.87,.46])
    last=max(row["global_step"] for row in main)
    xmax=max(44,last+4)
    ax.set_xlim(2,xmax);ax.set_ylim(58,84)
    ax.axvspan(2,20,color="#EAF2FC",zorder=0)
    ax.axvspan(20,40,color="#FFF2E7",zorder=0)
    ax.axvspan(40,xmax,color="#F1EDF8",zorder=0,alpha=.8)
    for boundary,color in [(20,COLORS["event"]),(40,COLORS["next"])]:
        ax.axvline(boundary,color=color,ls=(0,(5,4)),lw=1.6,zorder=2)
    ax.text(11,82.45,"初始世界模型阶段",ha="center",fontsize=12,color=COLORS["original"])
    ax.text(30,82.45,"第1轮重训后（当前主线）",ha="center",fontsize=12,color=COLORS["updated"])
    if last<=40:
        ax.text(42,66,"第2轮\n更新中\n暂无新点",ha="center",va="center",fontsize=10,color=COLORS["next"])
    groups=[([row for row in main if row["global_step"]<=20],"original"),
            ([row for row in main if 20<=row["global_step"]<=40],"updated"),
            ([row for row in main if row["global_step"]>=40],"next")]
    for rows,name in groups:
        if len(rows)>1:
            ax.plot([row["global_step"] for row in rows],[row["success_rate_pct"] for row in rows],
                    color=COLORS[name],lw=2.8,zorder=4)
    for row in main:
        step,y=row["global_step"],row["success_rate_pct"]
        name="original" if step<=20 else "updated" if step<=40 else "next"
        point=ax.scatter([step],[y],s=76,color=COLORS[name],edgecolor="white",linewidth=1.6,zorder=6)
        point.set_gid(f"policy-main-{step}")
        offset=(0,-14) if step in (10,15,35) else (-24,12) if step in (20,40) else (0,12)
        ax.annotate(f"{y:.1f}%\n{row['success']}/500",(step,y),xytext=offset,textcoords="offset points",
                    ha="center",va="top" if offset[1]<0 else "bottom",fontsize=11,
                    color="#243B53",bbox=dict(boxstyle="round,pad=.22",fc="white",ec="none",alpha=.9))
    if branch:
        start=next(row for row in main if row["global_step"]==20)
        rows=[start]+branch
        line,=ax.plot([row["global_step"] for row in rows],[row["success_rate_pct"] for row in rows],
                      color=COLORS["branch"],ls=(0,(4,4)),lw=1.8,zorder=3)
        line.set_gid("branch-line")
        for row in branch:
            point=ax.scatter([row["global_step"]],[row["success_rate_pct"]],marker="D",s=64,
                             color=COLORS["branch"],edgecolor="white",linewidth=1,zorder=5)
            point.set_gid(f"policy-branch-{row['global_step']}")
            label=ax.annotate(f"旧备用分支：第{row['global_step']}步\n{row['success_rate_pct']:.1f}%（{row['success']}/500）",
                (row["global_step"],row["success_rate_pct"]),xytext=(12,-4),textcoords="offset points",
                fontsize=10,color="#606D80",va="center",ha="left")
            label.set_gid(f"branch-label-{row['global_step']}")
    ax.set_xticks(list(range(5,last+1,5)));ax.set_yticks([60,65,70,75,80])
    ax.set_xlabel("策略 GRPO 步数",labelpad=10);ax.set_ylabel("LIBERO-Spatial 成功率（%）",labelpad=10)
    ax.grid(axis="y",color="#CAD4DF",lw=.7,alpha=.7)
    for side in ("top","right"):ax.spines[side].set_visible(False)
    for side in ("bottom","left"):ax.spines[side].set_color("#9AAAB9")
    ax.tick_params(colors="#425466")
    handles=[Line2D([0],[0],color=COLORS["original"],marker="o",lw=2.5,label="主线：初始世界模型"),
             Line2D([0],[0],color=COLORS["updated"],marker="o",lw=2.5,label="主线：重训世界模型后"),
             Line2D([0],[0],color=COLORS["branch"],marker="D",ls="--",lw=1.6,label="旧世界模型备用分支")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.07,.885),frameon=False,ncol=3,fontsize=11)
    cards=[(.075,.055,.415,.21,COLORS["event"],"第20步后｜采集数据／重训 #1（已完成）",
            "采集：第20步策略采集 500 条真实双相机轨迹\n重训：Wan 1400 步＋Cosmos 2000 步（修正版）\n接续：从第20步继续，测评第25／30／35／40步"),
           (.515,.055,.43,.21,COLORS["next"],"第40步后｜采集数据／重训 #2（进行中）",
            "数据：新采集 500 条＋旧数据 500 条，按1:1回放\n重训：Wan／Cosmos 各2000步；后续 GRPO 40→60\n截至本图生成时，第二轮更新后暂无新成功率测点")]
    for x,y,w,h,color,title,body in cards:
        fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.012",
            transform=fig.transFigure,fc="#FAFBFD",ec=color,lw=1.3))
        fig.text(x+.012,y+h-.042,title,fontsize=12.1,weight="bold",color=color)
        fig.text(x+.012,y+h-.08,body,fontsize=10.4,color="#34475E",va="top",linespacing=1.8)
    fig.text(.075,.015,"仅绘制真实测评记录；实测点直接连线，未平滑、未外推。采集成功率未混入测评曲线。",
             fontsize=9.5,color="#66788A")
    for suffix in ("png","svg","pdf"):
        fig.savefig(output / f"spatial_success_curve.{suffix}",dpi=210,bbox_inches="tight",pad_inches=.16)
    plt.close(fig)


def interactive_html(output, main, branch):
    namespace="http://www.w3.org/2000/svg"
    ET.register_namespace("",namespace)
    root=ET.fromstring((output / "spatial_success_curve.svg").read_text())
    point_lookup={f"policy-{'main' if row['series']=='main' else 'branch'}-{row['global_step']}":row for row in main+branch}
    for element in root.iter():
        gid=element.attrib.get("id","")
        if gid in point_lookup:
            row=point_lookup[gid]
            title=ET.SubElement(element,f"{{{namespace}}}title")
            title.text=f"{'当前主线' if row['series']=='main' else '旧世界模型备用分支'}｜GRPO {row['global_step']}步｜{row['success_rate_pct']:.1f}%｜{row['success']}/500\n来源：{row['source_metrics']}"
        if gid.startswith(("branch-","policy-branch-")):
            element.set("data-branch","true")
    svg=ET.tostring(root,encoding="unicode")
    rows="".join(f"<tr><td>{'主线' if row['series']=='main' else '旧备用分支'}</td><td>{row['global_step']}</td><td>{row['success_rate_pct']:.1f}%</td><td>{row['success']}/500</td><td>{html.escape(row['source_metrics'])}</td></tr>" for row in main+branch)
    document="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LIBERO-Spatial 成功率曲线</title>
<style>body{max-width:1450px;margin:24px auto;padding:0 16px;color:#243b53;font-family:system-ui,sans-serif}svg{width:100%;height:auto}label{display:inline-block;background:#f2f5f9;padding:10px 16px;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dce3ec;text-align:left}td:last-child{font:11px monospace;overflow-wrap:anywhere}a{color:#2463ad}</style>
<label><input type="checkbox" id="branch" checked> 显示旧世界模型备用分支（悬停测点查看来源）</label>SVG_CONTENT
<p><a href="spatial_success_curve.png">PNG</a> · <a href="spatial_success_curve.pdf">PDF</a> · <a href="spatial_success_curve.svg">SVG</a> · <a href="success_rates.csv">原始汇总CSV</a> · <a href="success_rates.json">来源与校验记录</a></p>
<table><thead><tr><th>分支</th><th>GRPO步数</th><th>成功率</th><th>成功数</th><th>原始测评记录</th></tr></thead><tbody>ROWS_CONTENT</tbody></table>
<script>document.getElementById('branch').addEventListener('change',e=>document.querySelectorAll('[data-branch]').forEach(x=>x.style.display=e.target.checked?'':'none'));</script></html>"""
    (output / "index.html").write_text(document.replace("SVG_CONTENT",svg).replace("ROWS_CONTENT",rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=PROJECT)
    parser.add_argument("--output",type=Path,default=PROJECT / "runtime_artifacts/reports/spatial_success_20261024")
    parser.add_argument("--font",type=Path,default=PROJECT / "runtime_artifacts/reports/assets/NotoSansSC_plot_subset.ttf")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    mainline,branch,events=gather(args.project)
    export_data(args.output,mainline,branch,events)
    plot(args.output,mainline,branch,args.font)
    interactive_html(args.output,mainline,branch)
    print(json.dumps(dict(output=str(args.output),mainline_points=len(mainline),other_branch_points=len(branch),
        values=[dict(step=row['global_step'],success_rate=row['success_rate_pct']) for row in mainline]),ensure_ascii=False))


if __name__=="__main__":main()
