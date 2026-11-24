# 八卡训练结束后的两个顺序看门狗

用户授权的执行顺序为：当前 GRPO 完成第 20 步后，先采集真实 LIBERO Spatial
双相机轨迹、更新 Wan/Cosmos，再使用新 world model 从第 20 步继续训练 20 步。
若第一条链任一阶段失败，第二个独立看门狗使用原 world model，从最近完整的
策略与 optimizer checkpoint 再训练 20 步。主链成功时备用路径退出。

`spatial_followup_watchdogs.py primary` 与 `fallback` 在两个独立 screen 中运行。
每个角色有自己的锁，另有一个共享工作锁保证两条路径不会同时启动 GPU 作业。
备用进程检查主进程 PID 和启动时间，因此主进程被强制终止、来不及写失败标记时
也能接手。它先清理带本次链和角色标记的残留进程，再选择完整断点，只执行一次。
训练启动仍检查 GPU 占用，其他作业占卡时等待；这不是调度器层的 GPU 独占。

正常路径复用 `queue_spatial_wm_update.py`：八卡采 20 条 pilot → 帧/动作检查 →
Wan 四卡和 Cosmos 四卡的完整更新探针 → 八卡采 500 条 → 固定训练验证清单 →
各四卡同时更新 2000 次、每 200 次保存 → 打包模型对 → GRPO 20→40。
每个模型的 micro batch=1、梯度累计 4 次、有效 batch=16。

续训使用原作业的 **resolved_config.yaml 快照**，保留当前批量、KIR、动作边界、
统计量、奖励分类器和共享显存池设置；重新生成私有 RPC 路径和输出位置。
`spatial_wovr_pair_launch.py` 为本次续训加载配置快照，并用校验和验证更新的
Wan/Cosmos 权重。Wan 使用原先冻结的 VAE 和 ResNet 奖励模型。
更新的模型对作为实验候选，GRPO 每五步仍在真实 LIBERO Spatial 上做 10×50 评估。

两条续训均每五步保存完整续训状态。备用断点必须包含策略权重、DCP 元数据和
八份非空分片，并且对应步数已完成日志记录。例如主链在第 34 步失败，最近完整
断点为 30，则备用使用原 world model 从 30 继续训练到 50，而不是只跑到 40。

新 GRPO 的日志和配置生成后会沿用现有邮件通知看门狗；邮件启动失败仅记录到
`notification_launches.jsonl`，不会因此终止正常训练。

另由 `spatial_followup_email.py` 在独立的 `spatial-next-results` screen 中观察
**两条看门狗的完整结果**：主链成功或失败、备用成功或失败、主链成功后备用无需
启动、主动撤销，均各发一封邮件。即使在重训 world model 阶段失败、尚未启动
GRPO，或者看门狗意外退出而未写终态，也会通知。它核对 PID 和启动时间，避免
PID 被复用掩盖退出，并重新检查完成标记以避免把正常退出误报为失败。

通知进程每两秒检查一次，不调用 GPU，也不参与训练调度。邮件事件保存在
`outcome_email/outbox/`，投递状态保存在 `outcome_email/delivery/`；SMTP 失败后
按 30、60、120、240、300 秒间隔重试，每次重新加载现有本地 SMTP 配置，日志
只记录错误类型与数字状态码。重启通知进程会继续待发邮件，已标记发送的结果
不会重复提交。同一事件使用固定 Message-ID；若 SMTP 接受邮件后通知进程恰好
退出、来不及记录发送结果，重试仍可能产生重复邮件。`sent` 表示 SMTP 中继接受，
不代表已经进入收件箱。两个角色的结果都提交成功后，通知进程才自动结束。

```bash
python=/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python
chain_root=/data41/qianb2608/dyk/grpo-world-model/runtime_artifacts/followup_chains/spatial_after20_20261021_node146
"$python" experiments/spatial_followup_watchdogs.py status --plan "$chain_root/plan.json"
cat "$chain_root/outcome_email/state.json"
```

结果通知可以独立重启，文件锁会阻止重复启动发送进程。首次启用加 `--send-armed`
发送一封明确标注为测试的启用邮件；已有同一计划的启用记录时不会重复发送。

```bash
"$python" experiments/spatial_followup_email.py --plan "$chain_root/plan.json" --send-armed
```

取消时应使用下面的入口，它先写撤销标记再关闭两个看门狗，避免主动停止主进程
被误认为失败而触发备用。当前旧 GRPO 在完成前不受这个撤销命令影响。

```bash
"$python" experiments/spatial_followup_watchdogs.py cancel --plan "$chain_root/plan.json"
```

`primary_state.json`、`fallback_state.json` 和各自 `*_events.jsonl` 记录运行状态；
`logs/` 保存阶段日志，`world_model_update/` 保存采集数据和更新后的模型。
CPU 测试覆盖主链成功、更新失败、主看门狗意外死亡、取消抑制备用触发、
断点完整性、配置继承、更新权重校验，以及两条链的结果通知、发送失败重试、
通知进程重启后的去重和磁盘满时继续投递。真实 GPU 更新探针在旧训练完成后执行。
