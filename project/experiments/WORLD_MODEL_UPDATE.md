# Spatial 双相机 world model 更新

这套入口沿用当前 π0.5（原生预测 10 步、实际执行 8 步）、Wan Spatial 和
Cosmos-WoVR wrist v3。支持四卡和八卡：采集使用全部指定 GPU，固定清单后，
Wan 和 Cosmos 各用一半 GPU 同时训练。四卡入口默认 Wan=0,1、Cosmos=2,3；
八卡入口默认 Wan=0,1,2,3、Cosmos=4,5,6,7。两者均为 DiT 全参数微调、
每卡 micro batch=1；四卡入口累计 8 次，八卡入口累计 4 次，每个模型的有效 batch 均为 16。
它实现 WoVR PACE / WMPO 的策略行为数据对齐，不改变现有 GRPO 算法。
WorldSample 的 Q-aware PPL 不在这个监督微调流程中。

## 代码与验证范围

- `wm_update_collect.py`：固定策略的真实 LIBERO 双相机采集，可按完整 episode 续采。
- `wm_update_data.py`：HDF5 真值格式、不可变 manifest、共享 13 帧窗口和两个数据适配器。
- `wm_update_wan.py`：原生 WoVR loss、两卡或四卡 FSDP、BF16 计算、FP32 主参数、梯度检查点。
- `wm_update_cosmos.py`：原生 Cosmos 动作条件 loss / DDP / DCP、梯度检查点。
- `cosmos_recipe_data.py` / `cosmos_recipe_train.py`：版本化的腕部 v3 配方重建与旧状态续训兼容。
- `wm_update_accept.py`：两个模型打包、相同验证轨迹上的新旧对照和假成功统计。

CPU 测试覆盖时间对齐、真实执行动作截断、分组划分、数据变更检测、
完整 optimizer/RNG 恢复和恢复后下一次更新的一致性。两个已有环境可执行
无 CUDA 的训练入口导入及配置检查。已完成真实 GPU pilot 采集及旧配方的正式更新。
新加入的显存预留路径和下述 Cosmos 重建配方已做 CPU 检查，
**新配方的真实 GPU 反向传播、更新及保存仍须通过 `probe` 实测。**
HDF5 编解码在内存中完成，磁盘使用顺序读写及原子提交，以避开已遇到的原生文件驱动 I/O 错误。

手动 GPU 命令启动前检查其分配 GPU 的占用，任一卡已使用超过 1024 MiB 即退出。
自动队列可用 `--wait-for-gpus` 持续等待；开启后即使显存不足 1 GiB，
只要仍有 CUDA 计算进程也不会启动。等待进程不加载 CUDA 模型，不驱逐其他作业。
多卡使用独立的 torchrun rendezvous，
采集无需 Ray。两组训练各自使用独立的 torchrun rendezvous、可见 GPU 和日志，
同时启动；任一组报错时监督器清理本次启动的两组进程。
`check`、`prepare`、`inspect`、`package`、`compare` 和 `--dry-run`
均不启动 GPU 作业。

## 八卡自动接续当前 GRPO

`queue_spatial_wm_update.py` 将指定的 GRPO run、配置、目标第 20 步策略和更新代码
固定到 `plan.json`。只有该 run 的 `exit_code=0`、日志完成第 20 步、最终权重存在、
launcher/learner 退出后才进入采集。失败退出不会回退到第 15 步或原始 π0.5。
在完成后清理本次 run 的残留 Ray workers，只接受 PID 与启动时间均匹配的本次进程
及其子进程，不清理其他作业。默认 GPU 未释放时记录失败；加 `--wait-for-gpus` 后持续等待。

自动流程依次为：八卡采 20 条 pilot → 校验统一帧 → 两组各四卡执行完整更新探针
并保存状态 → 八卡采 500 条真值 → 固定 80/20 清单 → Wan/Cosmos 各四卡同时微调
2000 次 optimizer update、每 200 次保存 → 打包新模型对。探针或任何阶段失败时停止，
状态与错误保存在 `state.json`，每阶段独立日志在 `logs/`。
新模型包不会自动切入下一次 GRPO；新旧视频指标和策略对照仍使用下方验收入口。

```bash
python=/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python
grpo_run=/data41/qianb2608/dyk/grpo_runs/spatial_wovr20_shared_pool_resume10_8gpu/20261021_061423_node146
update_root="$PWD/wm_updates/spatial_grpo20_20261021_061423_node146_8gpu"
watch_state="$PWD/runtime_artifacts/training_watchdog/20261021_061423_node146_4a64f91a/state.json"

# arm 只记录计划；run 在独立 screen 中等待目标训练成功结束。
"$python" experiments/queue_spatial_wm_update.py arm \
  --run-dir "$grpo_run" --watchdog-state "$watch_state" --output "$update_root"
screen -dmS spatial-wm-after-grpo20 env CUDA_VISIBLE_DEVICES= \
  "$python" experiments/queue_spatial_wm_update.py run --plan "$update_root/plan.json"
"$python" experiments/queue_spatial_wm_update.py status --plan "$update_root/plan.json"
```

也可在另一节点从已保存的指定策略启动，独立于原 GRPO 的结束状态：

```bash
"$python" experiments/queue_spatial_wm_update.py start \
  --policy-config /完整路径/resolved_config.yaml \
  --policy-checkpoint /完整路径/checkpoints/global_step_15 --policy-step 15 \
  --expected-node-ip 172.44.19.145 --wait-for-gpus \
  --output "$PWD/wm_updates/一个新的输出目录"
```

此模式绑定执行节点和策略权重 SHA-256；等待结束后再次校验权重，
不会清理原 GRPO 的任何进程。后台执行时将上述命令放入独立 screen。
失败阶段恢复时保留审查记录，仅复用 `events.jsonl` 中已有成功记录的阶段；
不会自动重新训练或覆盖已有训练目录。

需要手动使用八卡时，将下文的 `run_spatial_wm_update_4gpu.sh` 换为
`run_spatial_wm_update_8gpu.sh`，去掉或改写四卡 `WM_UPDATE_GPUS` 环境变量，
将显式 `--accumulation 8` 改为 `--accumulation 4`。
八卡池单模型续训仍给出完整 `--gpus 0,1,2,3,4,5,6,7`，各模型保持自己的四卡；
如只给出一组四卡，须明确加 `--ranks-per-model 4`，以区别原有四卡池的两卡组。
四卡与八卡训练状态具有不同 world size，不能直接互相续训。

## 1. 检查环境，选定策略

```bash
cd /data41/qianb2608/dyk/grpo-world-model
bash experiments/run_spatial_wm_update_4gpu.sh check --check-runtime
```

给采集器传入目标 GRPO run 的 **resolved_config.yaml** 或完整的
`tensorboard/config.yaml`，以及该策略的 `global_step_N` 目录或
`actor/model_state_dict/full_weights.pt`。不要用未展开 defaults 的训练 YAML。
采集会检查 horizon=10、执行 chunk=8，并保留配置中的原始 normalization stats。
配置文件会复制到采集目录，策略权重以 SHA-256 固定，恢复采集时重新核对。
策略配置应使用当前 Spatial 路线的 `libero_01_to_signed` 环境动作边界；
采集器只做一次对应的夹爪转换，并向策略输入零 proprio，与当前 GRPO 一致。

先填写下面两个路径，其他命令中的变量沿用同一 shell：

```bash
policy_config=/完整路径/目标实验/resolved_config.yaml
policy_checkpoint=/完整路径/目标实验/checkpoints/global_step_10
update_root="$PWD/wm_updates/spatial_round1"
export WM_UPDATE_GPUS=0,1,2,3
```

## 2. 先采 20 条检查，再采正式数据

```bash
bash experiments/run_spatial_wm_update_4gpu.sh collect \
  --policy-config "$policy_config" --policy-checkpoint "$policy_checkpoint" \
  --per-task 2 --output "$update_root/pilot" --dry-run

# 四卡空闲时执行：10 个任务，每个任务 2 条。
bash experiments/run_spatial_wm_update_4gpu.sh collect \
  --policy-config "$policy_config" --policy-checkpoint "$policy_checkpoint" \
  --per-task 2 --output "$update_root/pilot"

bash experiments/run_spatial_wm_update_4gpu.sh prepare \
  --collection "$update_root/pilot" --output "$update_root/pilot_manifest.json" --verify-rgb
bash experiments/run_spatial_wm_update_4gpu.sh inspect \
  --manifest "$update_root/pilot_manifest.json" --verify-files
```

确认记录后正式采 500 条，将 `--per-task` 改为 `50`，输出改为
`"$update_root/collection"`。建议用 screen 运行；命令失败后原样重跑即可复用
已完成 episode。每卡日志为 `collect_gpuN.log`；只有所有 shard 完整后才生成
`collection.json`。

每个 episode 保存：两路 RGB JPEG、实际执行的 7D 动作、仿真完整状态、9D proprio、
逐动作 reward/done/success、任务/初始状态编号/种子/策略指纹。成功和失败都保留。
时间约定为 **N 个动作、N+1 个观测，action[i] 把 observation[i] 推进到 observation[i+1]**。
提前成功或终止后不再执行剩余动作，最后不足 8 步的真实尾部仍保存，训练仅采完整窗口。

## 3. 固定训练/验证清单，可加入旧数据回放

```bash
bash experiments/run_spatial_wm_update_4gpu.sh prepare \
  --collection "$update_root/collection" \
  --output "$update_root/manifest.json" --verify-rgb

# 如有可复用的旧 Spatial 真值 HDF5，在上面的 prepare 命令中加：
# --replay-root /旧数据/all_episodes
```

约 80% / 20% 按完整 episode 划分；同一 task 的已知 initial_state_id 整组放入
同一集合，包括不同策略在该初始状态的重复轨迹。manifest 固定后不能替换内容。
旧数据没有可靠初始状态 ID 时按轨迹分组，无法保证与新数据的初始状态互斥；
也无法证明和模型原始预训练集互斥，这些限制记录在 split_status 中。
最终 LIBERO 10×50 策略评估应单独运行，明确记录初始状态重叠。

`config/spatial_wm_update.json` 控制采样。默认新旧各 50%（没有旧数据就全采新数据）、
任务均衡、成功/失败均衡；完整轨迹均匀窗口为主，并提高首次稳定夹爪闭合前 16 帧
附近窗口的采样概率。记忆不足时重复真实初始观测，未来目标不做填充。
Wan 的参考帧固定为 o0，记忆帧为最近四帧，未来为八个后继帧。
修正后的 Cosmos 使用 `[主相机0, 腕部0, 腕部t-2, 腕部t-1, 腕部t, 未来8帧]`；
前两帧动作在模型归一化空间中固定为 `[0,0,0,0,0,0,-1]`，
其余动作使用对应观测的 incoming action，即 action[k-1] 产生 observation[k]。
t=0 使用真实初始腕部帧重复填充历史，所有缺失前驱动作均使用同一 sentinel；
t=1、2 重采到 t>=3，过短 episode 则使用 t=0。未来目标不填充。
腕部中心裁剪、相机空间转换和动作归一化复用生产推理协议。

## 4. 一次反向传播检查，再正式训练

### Cosmos v3 重建配方

新启动的 Cosmos 默认 `--cosmos-recipe auto`，无 `--resume` 时选择
`v3-reconstructed`，版本标记为 `wrist-v3-reconstructed-v1`：

- 固定主相机参考与腕部初始锚点，补齐上述 13 帧和 incoming-action 排列。
- `memory_degrade_prob=0.5`：只对最近三张腕部历史图像做降采样、模糊和轻量像素噪声，
  保留主参考、腕部锚点和未来真值。这是重建的像素增强，不是扩散 loss 的加噪。
- `static_video_probability=0.05`：保留真实历史，将未来八帧保持为当前腕部图像，
  对应未来动作为零位姿增量并保持夹爪。这部分也是重建实现。
- 复用原生 `dataset_utils.apply_image_aug(stronger=True)`；一次裁剪、旋转和颜色变换
  使用同一组参数处理完整 13 帧。验证集禁用全部这些增强。
- 将设置中的 `weight_decay` 真正传入原生优化器，默认 0.01。
  图像增强随机数与训练的模型/扩散噪声 RNG 隔离，数据索引可重复恢复。

原生扩散 sigma/SDE、噪声采样、条件 latent 处理和 loss 保持原样；原生 trainer、
DCP、FP32 master weights、optimizer 和 rank RNG 的保存恢复路径继续使用。
action dropout 保留环境原值，并在训练 identity 记录实际值；不猜测原始发布版的值。
有效 batch 仍为显式 ranks × accumulation，默认每模型 16；发布摘要中的 batch=32
未说明是全局还是每卡，当前不宣称完整复刻该 batch 或所有原始训练超参。
任务/成功失败均衡与关键帧采样也仍由现有 `sampling` 配置控制。

发布摘要确认了概率和动作布局，但缺少原始增强实现，因此上述两项被明确标记为重建。
可在 settings JSON 的 `cosmos_recipe` 对象中覆盖 `memory_degrade_prob`、
`static_video_probability`、`use_image_aug`、`use_stronger_image_aug`、`weight_decay`。
若未覆盖，static_video_probability 和 weight_decay 会沿用 settings 顶层已有值。
有效配方写入 training plan、`recipe/rank_N.json` 和 checkpoint identity。

已有旧配方的 `--resume` 在 auto 模式下保持 `legacy`，保留其原始代码身份和
优化器设置。不能将旧 optimizer 状态直接标成新配方；显式跨配方 resume 会提前报错。
要以新配方从指定模型开始，应使用新的输出目录，不传 `--resume`，并设置
`COSMOS_WRIST_CHECKPOINT=/完整路径/iter_000020000`（或其他明确选定的模型目录）。
此时加载模型参数并初始化新优化器。原始 v3 发布包没有 optimizer，无法恢复其 Adam 历史。
启动前会在 CPU 上读取 `model/.metadata`，确认其引用的模型分片与字节范围均完整，
并在 recipe 日志记录实际加载目录和模型 metadata SHA-256。只有训练摘要不足以启动。
修正版额外检查原生模型实际请求的每个 tensor 名称和形状，缺少任何权重（包括
action_mlp1/2）均拒绝继续，避免原生 partial-load 行为保留部分随机初始化。
每卡加载记录在 `recipe/model_load_rank_N.json`，只有 native DCP 复制成功后才标记 loaded。

已有第 200 步模型可作为这次重训的初始化。设置
`COSMOS_WRIST_CHECKPOINT=.../checkpoints/iter_000000200`，使用
`--settings experiments/config/spatial_cosmos_v3_reconstructed.json` 和新输出目录，
不传旧 `--resume`。重训 step 从 0 计数，学习率和 Adam moments 重新开始；
后续本轮新 checkpoint 才通过 `--resume` 恢复完整优化器、scheduler、FP32 master 和 RNG。
修正版 DataLoader 使用独立的 CPU generator，创建或重建 iterator 不消耗模型的
CPU RNG；已用真实 DataLoader 验证连续更新与保存后恢复的下一次更新完全一致。
启动时根据空闲 GPU 选择 8/4/2 个 ranks，accumulation 分别为 2/4/8，
均保持每卡 micro batch 1、全局 batch 16。不会改变仍在运行的 GRPO。

### 持续预留显存

Wan/Cosmos 的 `train`、`probe` 和 Wan 四卡转八卡续训入口默认启用显存预留。
CUDA allocator 使用 native 与 expandable segments；不显式传入
`garbage_collection_threshold`，沿用默认关闭回收的值，避免本环境 PyTorch 拒绝显式 0.0。
Inductor 编译线程设为 1，避免额外编译进程在 GPU 0 创建 CUDA context。
Cosmos 在导入原生后端前先绑定本 rank 的设备并建立预留。
每张卡只有自己的训练 rank 预留，其模型、梯度、优化器和 checkpoint 操作
复用同一 CUDA allocator；两个模型仍使用不同 GPU 组。
目标为设备总容量减去 `3.5 GiB` 余量，80 GiB 卡对应约 76.5 GiB 的设备总占用。
模型加载、NCCL 与 FSDP/DDP 临时分配仍可能改变瞬时占用，真实可用峰值须实测。
训练 rank 启动时再次检查计算和图形进程，发现其他进程就退出，不占用对方的卡。

显式入口如下；基础入口也默认启用同一模式：

```bash
# 八张空闲卡：Wan 四卡 + Cosmos 四卡，各自 batch 16。
bash experiments/run_spatial_wm_update_reserved_8gpu.sh train \
  --gpus 0,1,2,3,4,5,6,7 --model both \
  --manifest "$update_root/manifest.json" \
  --output "$update_root/train_reserved" \
  --steps 2000 --save-interval 200 --accumulation 4

# 四张空闲卡：先做一次完整反向、优化器更新和保存。
bash experiments/run_spatial_wm_update_reserved_4gpu.sh probe \
  --gpus 0,1,2,3 --model both \
  --manifest "$update_root/pilot_manifest.json" \
  --output "$update_root/probe_reserved"
```

可用 `--memory-headroom-gib 4` 增加余量，或用
`--memory-target-gib 75` 设置较低的整卡目标；目标与余量之和不能超过卡容量。
`--no-reserve-memory` 用于明确要求不预留的测量。
单个已有八卡 Wan checkpoint 可加 `--model wan --ranks-per-model 8`，
同时保留原有 accumulation、学习率和数据参数续训。

每个模型输出的 `memory_reservation/rank_N.jsonl` 记录初始化、首次前向/反向、
优化器更新、checkpoint 与推理权重导出阶段的 allocated/reserved、整卡使用量、
余量和其他 PID。`policy_rank_N.json` 是最新预留配置，带 policy ID 的文件
保留历史配置和独立代码指纹。原训练代码、数据/优化配置与 checkpoint 身份不变。

缓存补充分配发生在原生训练峰值计数重置之前，临时占位峰值单独标记为
`reservation_transient_peak_allocated_gib`；训练 `peak_allocated_gib` 用于观察
实际张量峰值，`peak_reserved_gib` 会包含持续缓存。原生显存显式清空在该 worker
内被抑制，PyTorch 自身的分配和 OOM 回收机制保持可用。
新增预留模式已做 CPU 的反向、RNG/优化器一致性和 DCP 保存恢复检查，
尚未占用正在运行的 GRPO GPU 做完整模型实测；空闲卡上先执行上述 probe。

```bash
# 两组各两卡同时做一次完整 optimizer update（每卡八次微批反向），并保存完整状态。
# 这会实际占用指定四卡；没有空闲卡时退出。
bash experiments/run_spatial_wm_update_4gpu.sh probe \
  --manifest "$update_root/pilot_manifest.json" --model both \
  --output "$update_root/probe"

# 查看实际峰值与每次更新耗时。
cat "$update_root/probe/wan/metrics.jsonl"
cat "$update_root/probe/cosmos/metrics.jsonl"

# 正式微调：Wan 与 Cosmos 各两卡同时启动。
bash experiments/run_spatial_wm_update_4gpu.sh train \
  --manifest "$update_root/manifest.json" --model both \
  --steps 2000 --save-interval 200 --accumulation 8 --lr 1e-6 \
  --output "$update_root/train"
```

默认每卡一个窗口，每个模型的有效 batch = 2 卡 × 8 次累计 = 16。
两卡训练的实际峰值与总耗时需重新测量；Wan 的每卡参数和 optimizer 分片
比四卡时更大。`probe --model both` 会按正式的两卡＋两卡布局检查。
可以在 `train` 或 `probe` 命令末尾加 `--dry-run`，先查看两组分配和命令，
此时不启动作业。实际启动时计划保存在 `training_plan_both.json`。
日志中的 step 指 **optimizer update**，不等于 GRPO 外层 step。
峰值记录 CUDA allocator 的 allocated/reserved（跨 rank 取最大值），
不是整张卡的 NVML 总占用。训练期间的验证是固定随机噪声下的监督损失；
完整自回归生成和视频指标使用下一阶段的独立入口。

Wan：`train/wan/step_000200` 等目录保存分片模型、optimizer、各 rank RNG 和完成标记；
其中 `runtime/` 导出可供原有 Wan 推理直接加载的模型。
Cosmos：`train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints/iter_000000200` 等目录
使用原生 model/optim/scheduler/trainer 状态，额外保存 FP32 主权重 `master_weights.pt`、
各 rank RNG 与完成标记。FP32 主权重独立保存是因为原生 FusedAdam 的默认状态不包含它；
续训时显式恢复，避免由 BF16 参数重建而损失精度。
只有完成标记存在的 checkpoint 才可用于续训或模型打包。

```bash
bash experiments/run_spatial_wm_update_4gpu.sh train --model wan \
  --manifest "$update_root/manifest.json" --output "$update_root/train" \
  --steps 2000 --resume "$update_root/train/wan/step_000200"

bash experiments/run_spatial_wm_update_4gpu.sh train --model cosmos \
  --manifest "$update_root/manifest.json" --output "$update_root/train" \
  --steps 2000 \
  --resume "$update_root/train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints/iter_000000200"
```

单独续训时，`--gpus 0,1,2,3 --model wan` 使用 0,1，`--model cosmos` 使用 2,3，
只检查本组卡的占用。也可直接给单模型 `--gpus 2,3` 选择两张卡。
续训要求相同 GPU 数、manifest、代码、采样和优化设置；已存在的四卡训练状态
不能当作两卡状态直接续训。数据采样索引从 optimizer
step 推导，恢复后继续读取相同序列；验证不会改变训练 RNG。
可以增大总 `--steps`，Cosmos 学习率 schedule 不随总步数改变。
探针使用 pilot 数据，不能把它的 optimizer 状态直接恢复到正式数据集。

## 5. 两个模型成对打包和验收

```bash
bash experiments/run_spatial_wm_update_4gpu.sh package \
  --manifest "$update_root/manifest.json" \
  --wan-checkpoint "$update_root/train/wan/step_002000" \
  --cosmos-checkpoint "$update_root/train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints/iter_000002000" \
  --output "$update_root/pair_2000"

# 基线使用当前 WAN_SPATIAL_CHECKPOINT / COSMOS_WRIST_CHECKPOINT。
bash experiments/run_spatial_wm_update_4gpu.sh evaluate \
  --manifest "$update_root/manifest.json" --output "$update_root/eval_base"
bash experiments/run_spatial_wm_update_4gpu.sh evaluate \
  --pair "$update_root/pair_2000" --output "$update_root/eval_new"
bash experiments/run_spatial_wm_update_4gpu.sh compare \
  --baseline "$update_root/eval_base" --candidate "$update_root/eval_new" \
  --output "$update_root/comparison"
```

两个版本使用完全相同的 held-out episode、实际长度、5 步去噪和随机种子，
比较 FPS、LPIPS、FID、FVD 和 FloLPIPS。少于 16 帧的验证轨迹显式列为
视频指标 exclusions，仍保留在数据清单中。默认视觉检查允许 LPIPS / FloLPIPS
至多 5% 相对回退，FID/FVD 在小样本下作为辅助信息。

`outcome_metrics.json` 使用当前 Spatial 的固定 ResNet 奖励分类器与 0.9 阈值，
在 Wan **原生 256px** 生成视频和对应真实视频上分别评分，对照仿真成功标签计算
假成功率、漏检率及置信区间。不会用缩小后的指标图像给分类器评分。
分类器本身仍会出错；这项统计是诊断，腕部与主视角的物理一致性还应检查视频。
`comparison.json` 在数据或评测协议不同时拒绝比较，并记录假成功率是否上升。

模型包只创建版本化软链接和清单，不切换正在运行的 GRPO。
通过 world model 验收后，还要固定同一策略起点和 GRPO 预算，做新旧模型的续训对照，
并分别进行真实 LIBERO Spatial 10×50 策略评估。

## CPU 回归测试

```bash
env CUDA_VISIBLE_DEVICES='' /data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  -m unittest discover -s experiments/tests -p 'test_wm_update*.py' -v
```

参考：[WoVR](https://jzndd.github.io/assets/pdf/WoVR.pdf)、
[WMPO](https://wm-po.github.io/static/pdfs/WMPO_Seed.pdf)、
[WorldSample](https://xxreinsno.github.io/worldsample/)。
