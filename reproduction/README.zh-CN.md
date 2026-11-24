# LIBERO-Spatial 双相机 GRPO 复现

## 最新可续训权重：第 60 步

**最新完整策略断点为 `weights/policy/global_step_60/`。第 60 步之后的训练请从这份断点恢复。**
它包含完整模型权重、优化器状态及 8 个 DCP 分片；真实 LIBERO-Spatial
10 个任务 × 50 次评估成功率为 **77.2%（386/500）**。
`latest_policy.json` 记录最新断点、评估结果和续训入口。40、45、50 步用于历史对比。

**结果来源**：77.2% 是原单节点八卡训练在第 60 步执行的真实仿真验证，
使用 `runner.only_eval=false`、`runner.val_check_interval=5`，并非本发布包的独立
`portable.py eval` 入口得到。原训练日志记录于 2026-10-25 13:18:07；完整 500 条
结果见 `evidence/evaluations/global_step_60.json`，入口及验证范围见
`evidence/portability_fix_20261027.json`。

下载并执行下文的 `restore`、`check` 后，完整的下一轮实验使用以下入口：

```bash
screen -dmS spatial-cycle60 bash -c \
  'bash reproduction/run_world_model_cycle.sh "$PWD/runs/cycle60"'
screen -r spatial-cycle60
```

它依次执行：60 步策略采集 500 条真实双相机轨迹 → 更新 Wan/Cosmos →
打包本轮新模型 → 策略从 60 步继续到 80 步。完整配方见“后续世界模型更新”。

以下单条命令只续训 GRPO，不采集数据或更新世界模型；使用包内 Wan 1000/Cosmos 2000：

```bash
screen -dmS spatial-grpo60-to80 bash -c \
  'python3 reproduction/portable.py train --resume 60 --target 80 --output "$PWD/runs/resume60_to80"'
screen -r spatial-grpo60-to80
```

`--target 80` 表示训练至总步数 80，即新增 20 步；恢复优化器时使用完整断点目录。


本发布包对应 π0.5 + Wan 主视角 + Cosmos-WoVR 腕部视角，在另一台 Linux
服务器上启动八卡 GRPO。训练使用世界模型生成的观测；成功率是在真实 LIBERO
仿真器中，以固定初始状态、10 个任务 × 50 次试验统计的 success-once。

## 包中内容

- `source/`：本次源码快照，包含项目覆盖层及实际使用的 RLinf、Cosmos、OpenPI、
  DiffSynth、LIBERO 源码与仿真资源。
- `runtime/`：Python 3.11/3.10 解释器及两个环境实际使用的依赖层。恢复时保持层的
  顺序，不使用包含重复版本的 `pip freeze` 输出来重建环境。
- `weights/pi05/global_step_12000/`：匹配的 few-shot SFT 初始化权重和统计量。
- `weights/policy/global_step_40/`、`global_step_50/`、`global_step_60/`：可供原生 RLinf 续训的
  完整策略权重及 DCP 训练状态。`global_step_45/` 只有完整策略权重，可用于评估
  或模型初始化，不用于恢复优化器状态。
- `weights/wan_step1000/`、`weights/cosmos_step2000/`：本轮 GRPO 使用的世界模型
  推理权重。它们的世界模型训练优化器状态不在本发布包中。
- `assets/`：Wan 基础包及 VAE、奖励模型、Cosmos VAE、T5 缓存、统计量与 tokenizer。
- `data/`：KIR 使用的 500 条 Spatial 双视角源轨迹，以及 Wan 初始观测资源。
- `config/`：实际运行配置和原始 KIR 索引；恢复程序会重定位路径，核对大小和内容
  校验和，并更新本地文件指纹，不重新选择关键帧。
- `evidence/`：评估结果、模型配对元数据、环境版本清单与实验谱系。

参考检查点的真实 Spatial 成功率：40 步 365/500（73.0%），45 步
370/500（74.0%），50 步 349/500（69.8%），最新 60 步 386/500（77.2%）。40 步由前一轮世界模型训练得到；
40→60 的本轮续训使用新 Wan 1000 步和新 Cosmos 2000 步。

## 目标服务器

二进制环境适用于 Linux x86_64，参考系统为 Ubuntu 22.04 / glibc 2.35，
8 张 A800 80 GiB，驱动 595.45.04。另一种 CPU 架构、较旧 glibc、不同 CUDA
驱动或其他显卡需要额外验证。请预留发布包下载空间和约 45 GiB 的环境解压空间，
另为训练日志、视频与完整检查点预留至少 100 GiB。

系统依赖示例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential zstd screen ffmpeg libegl1 libgl1 libglfw3 \
  libglib2.0-0 libsm6 libxrender1 libxext6 libosmesa6
```

## 下载、恢复与检查

先用服务器已有的 Python 安装下载工具，下载公开仓库。下载凭据不属于发布包。

```bash
python3 -m pip install 'huggingface_hub>=0.36'
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('dyk111zn/GRPO-LIBERO-Spatial-Reproduction', local_dir='spatial-reproduction')"
cd spatial-reproduction
python3 reproduction/portable.py restore
python3 reproduction/portable.py check
# 在带 NVIDIA 驱动的目标服务器上，额外验证两个环境的首次 Triton 编译：
python3 reproduction/portable.py check --jit-smoke
```

`restore` 默认验证所有发布文件的 SHA-256。Python 3.10 的完整开发头文件由
`runtime/python310_headers.tar.zst` 补充，其中 `pyconfig.h` 已解析为 x86_64 的实际
配置，不依赖目标机安装 `python3.10-dev`。系统仍需 C 编译器与 libc 开发头文件。

`check` 使用恢复后的两个解释器编译并加载小型 Python C 扩展，然后检查源码导入、
配置、世界模型分片、策略、统计量、KIR 数据与相机/动作约定。`train`、`eval` 启动
服务前也执行 C 扩展检查，缺少头文件时立即失败。`check --jit-smoke` 还会用每次
新建的 Triton 缓存实际编译并加载 `cuda_utils`，不会被旧机器已有缓存掩盖。
日志在 `runs/compile_checks/`。这些检查不创建 CUDA 上下文，不执行 GPU 内核或
完整腕部服务前向，因此不等同于完整八卡 eval 通过。恢复后移动整个目录时，重新
执行 `restore` 以更新路径。

已下载旧版 102 文件发布包的用户，在发布包根目录重新下载增量并恢复即可：

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('dyk111zn/GRPO-LIBERO-Spatial-Reproduction', local_dir='.')"
python3 reproduction/portable.py restore
python3 reproduction/portable.py check --jit-smoke
```

相同 SHA-256 的旧归档会按恢复记录跳过，新头文件归档与源码修复层会自动解压。
此更新不修改策略/世界模型权重，也不要求重新下载相同的权重文件。

如果已经手改过 `sources/rlinf_backend/` 下的权重同步代码，先保留需要的改动，再用
下面的命令恢复随包后端及全部源码修复层；默认 `restore` 不重解压已有记录的旧源码。
这一步会覆盖源码文件中的手动修改，运行环境和权重归档仍按原记录跳过：

```bash
python3 reproduction/portable.py restore --refresh-source
```

## 启动 GRPO

历史复现示例：使用第 40 步完整策略及新世界模型，继续到第 60 步。
第 60 步之后的续训使用上面的最新 60 步断点。

```bash
screen -dmS spatial-grpo bash -c \
  'python3 reproduction/portable.py train --resume 40 --target 60 --output "$PWD/runs/resume40_to60"'
screen -r spatial-grpo
```

使用第 50 步完整断点继续到第 60 步：

```bash
python3 reproduction/portable.py train --resume 50 --target 60 \
  --output "$PWD/runs/resume50_to60"
```

从 SFT 12k 初始化，使用当前新世界模型开始一个新的 20 步实验：

```bash
python3 reproduction/portable.py train --steps 20 --output "$PWD/runs/new20"
```

新的 SFT→20 实验与历史策略 0→20 的世界模型和谱系不同，不能把两者当成相同实验。
原生 DCP 恢复模型、优化器等已保存状态；本发布不承诺跨服务器逐比特复现采样序列。

默认使用 0–7 号卡，每 5 步保存策略并做真实 Spatial 评估，每 10 步及最后一步
保存完整训练状态。启动前会检查 GPU 占用。计算进程保留可复用显存缓存，
稳态目标约 75 GiB，启动目标约 65 GiB，设备余量配置 3.5 GiB。
这些缓存会被计算复用；设备独占仍需服务器调度器保证。

## 评估发布的策略

```bash
python3 reproduction/portable.py eval --checkpoint 60 --output "$PWD/runs/eval60"
```

该入口只做真实 LIBERO 评估，不执行 GRPO 更新；沿用八卡启动器，因此仍会加载
世界模型与策略。内部使用 `runner.only_eval=false` 保留训练路径已经使用的完整
actor/rollout 初始化及双向权重同步，再由 `reproduction/eval_train.py` 将 runner 的
`run()` 替换为“同步一次 → 评估一次 → 退出”，不调用训练采样或优化器更新。
不能只跳过 rollout 一端的接收：actor 端仍发送时会留下不匹配的通信操作。
所选策略检查点记在 `runs/launch_eval.json`，报告步骤沿用策略步数。

2026-10-27 修复前的独立入口存在 `rollout.model.is_lora` 缺失和同步器未初始化的
问题，原 CPU 检查未覆盖，不能把历史 77.2% 作为该独立入口已经通过的证据。
修复的测试范围见上述 evidence 文件。H200 与 EFA 的完整 eval 仍需目标机验证；
原八卡参考运行未显式设置 `FI_PROVIDER=efa`，没有验证过 EFA 配置。不要仅凭
`Unsupported object type` 或 `gloo_send_0` 将故障归因于 GPU 或网络硬件。

另一个已确认的问题是后端 `_broadcast()` 捕获异常后只写日志，上层随后仍会读取
`torch.empty()` 的元数据缓冲区，产生负数 `Unsupported object type`。本修复在每次
启动的私有源码覆盖层中保留首次通信异常、完整 traceback、组名、rank、src 与张量
形状，不修改共用后端安装。这个修复不代表底层连接失败已定位或消除；需要检查第一条
`Broadcast failed ... with error:` 及其之前的 worker 错误，而非只看后续负数。
需要更多诊断时，可在下一次启动前设置 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 和
`RAY_DEDUP_LOGS=0`；不会自动更改网卡或启用 EFA。

`train --resume 60 --target 80` 默认在第 65 步首次验证，此时策略已经更新。
核对第 60 步的 77.2% 应使用上述 `eval --checkpoint 60`，而非第 65 步结果。
参考配置使用随机策略采样（`flow_sde`、`noise_level=0.5`、`temperature_eval=1.6`）；
训练内验证与独立评估此前消耗的随机数不同，不承诺每次恰好得到 386/500。

## 后续世界模型更新

完整入口为 `reproduction/run_world_model_cycle.sh`，从恢复好的发布包根目录执行，
需要八张可用 GPU。每个原生入口启动前都会检查占用；指定的卡有其他作业时退出。
采集和世界模型训练均使用实际计算进程持有的可复用显存缓存，保留 3.5 GiB 设备余量。

先查看命令计划，不启动 GPU 任务：

```bash
bash reproduction/run_world_model_cycle.sh --dry-run "$PWD/runs/cycle60"
```

正式执行上面的 `screen -dmS spatial-cycle60 ...` 命令。各阶段为：

1. **采集真实数据**：八卡使用 `weights/policy/global_step_60/`，在 LIBERO-Spatial
   的 10 个任务上各采集 50 条轨迹。保留成功和失败，`mode=train`，每条最多 240 步，
   每次执行 8 个动作；主相机、腕部相机、动作和仿真状态按同一帧索引保存。
   结果在 `runs/cycle60/collection/`。
2. **准备训练集**：新采集的 500 条轨迹与包内 `data/kir_sources/` 的 500 条公开
   Spatial 轨迹组成新数据/回放数据，验证比例 0.2；训练采样的新数据比例为 0.5。
   生成 `runs/cycle60/manifest.json` 并校验文件。历史每轮采集的原始数据未随包发布；
   此处回放来源是包内公开轨迹。
3. **更新两个世界模型**：Wan/Cosmos 各四卡并行，梯度累计 4 次，有效 batch 各为 16，
   学习率 `1e-6`，Cosmos 使用 `v3-reconstructed` 配方，目标 2000 步，
   显式设置 `--save-interval 1000`。从包内 Wan 1000/Cosmos 2000 的模型参数初始化，
   为本轮数据建立新的世界模型优化器；历史世界模型优化器状态不在发布包中。
4. **选择并打包新模型**：使用 `wm_update_until_cosmos.py` 控制两个训练组。
   Cosmos 完成 2000 步后，请求 Wan 在下一个完整优化器更新后保存并结束，
   不强制 Wan 达到 2000 步。读取 `train/selected_world_models.json`，打包生成
   `runs/cycle60/model_pair/pair.json`。视频指标可另行评估，不阻塞接续。
5. **继续 GRPO**：恢复第 60 步策略的完整 DCP 和优化器状态，用本轮新模型训练至
   总步数 80；第 70、80 步保存完整策略训练状态，每 5 步保存策略并做真实 Spatial
   10×50 评估。结果在 `runs/cycle60/grpo/`。

`reproduction/world_model.py` 为原生采集和更新工具设置本地恢复后的源码、解释器、
模型和数据路径。也可分阶段执行；例如本轮世界模型已训练完：

```bash
python3 reproduction/world_model.py package \
  --selection "$PWD/runs/cycle60/train/selected_world_models.json" \
  --output "$PWD/runs/cycle60/model_pair"

python3 reproduction/portable.py train --resume 60 --target 80 \
  --model-pair "$PWD/runs/cycle60/model_pair/pair.json" \
  --output "$PWD/runs/cycle60/grpo"
```

`--model-pair` 会同时切换 Wan 和 Cosmos，并复用冻结的 VAE、奖励模型和初始观测资源。
不用该参数时，继续使用发布包内的世界模型。完整入口使用种子 `20261021`；
通过 `WM_CYCLE_GPUS`、`WM_CYCLE_SEED` 可调整物理卡号和本轮随机种子。

原服务器的邮件配置、token、登录缓存、Ray 临时目录、活动进程和未完成检查点均
不属于发布包。新服务器需要邮件通知时，另行配置现有 `watch_training_email.py`。

## 来源与许可证

Built on NVIDIA Cosmos。每个上游组件和权重保留自身许可证，见 `LICENSES/`，
包括 NVIDIA 模型条款、Apache-2.0、Gemma 条款及 LIBERO 数据集的 CC BY 4.0。
没有给全部内容重新指定统一许可证。

来源包括 [Pi05 few-shot](https://huggingface.co/THU-SIGS-EILAB/Pi05_few_shot_libero)、
[Wan Spatial](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial)、
[Cosmos Predict2](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)、
[Cosmos Policy](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)、
[LIBERO-Cosmos-Policy 数据](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)
和 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。
