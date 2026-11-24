# WorldSample → WoVR-GRPO 迁移方案(配对组版)

## 0. 每迭代数据流(按用户想法)

每个 GRPO 迭代消费两组轨迹,组内独立归一化、两组同时回传:

1. **真实组**:8 条 LIBERO rollout(当前策略、真实动力学、仿真器 termination reward)
   → 组内归一化 advantage → GRPO 损失 L_A;
2. **合成组**:从真实组选 1 条轨迹,对它的动作做 WorldSample 式反事实加噪,
   生成 8 条 WM 轨迹(Wan 主视角 + Cosmos wrist、goal ResNet reward)
   → 组内归一化 advantage → GRPO 损失 L_B;
3. **策略更新**:L = L_A + ρ(H_t)·L_B。

这与论文 Eq.4 一一对应:L_Q(real) + ρ(H_t)·w_Q·L_Q(syn),只是把 TD loss 换成
GRPO 的组内 clipped surrogate。现状管线(纯 WM GRPO)对应「只有合成组」的特例。

## 1. 事实核对(代码证据)

| 结论 | 依据 |
| --- | --- |
| GRPO 损失是 clipped PPO surrogate,依赖 `rollout_logprobs` | `rlinf/algorithms/losses.py`:`compute_grpo_actor_loss_fn` → `compute_ppo_actor_loss` |
| advantage 组内均值/标准差归一化,无 critic | `rlinf/algorithms/advantages.py`:`compute_grpo_advantages` |
| flow-noise 采样 = `x_t_mean + ε·x_t_std`,logprob 用同一 `x_t_std` | `openpi_action_model.py:1122-1124` |
| 策略熵代码已算(flow_noise 高斯熵) | `openpi_action_model.py`:`get_log_prob_value` → `chains_entropy` |
| 训练 recompute 用存储的 chains + denoise_inds 重算 logprob | `workers/actor/fsdp_actor_worker.py`:`recompute_logprobs` |
| 流 A reward = 仿真器 termination;流 B reward = goal ResNet(阈值 0.65) | `libero_env.py`:`_calc_step_reward`;train config |
| WM reset 条件输入已打通,可从任意真实观察起步 | `cosmos_world_env.py`:`_reset_from_libero_state_bridge` |

## 2. 反事实加噪的可行形式(α 旋钮)

**论文附录 B.2 确实给了字面加噪公式**,围绕已执行动作做轨迹级扰动:

```
a'_t = Π_A(a_t ⊙ ξ + ε_t),  ξ_j ~ U(1−s, 1+s),  ε_t,j ~ N(0, σ²)
s = 0.20, σ = 0.05, inactive 维固定为 0,Π_A 投影回动作空间
```

**但 B.1 同时说明底层 learner 是 SAC/RLPD(off-policy actor-critic,带 Q 函数)。**
这解释了为什么字面加噪在论文里成立:

- 扰动动作只以 (s, a, s', r) 元组形式进入 critic 的 TD loss——Q-learning 对任意
  off-policy 动作都成立,**不需要动作密度**;
- actor 靠学到的 Q 以重参数化梯度更新,也从不要求扰动动作的密度;
- 论文的合成流本质上是 off-policy 数据,PPL(Q-aware selection + entropy scheduling)
  管理的正是这件事。

**当前管线是 critic-free 的 GRPO(on-policy likelihood-ratio)**:clipped surrogate 需要
π_θ(a|s)/π_old(a|s),而 flow-matching 对任意给定动作没有边际密度,代码里的 logprob 只是
「模型自身采样路径上的 per-step 条件高斯」。所以同一个 (s, a, s', r) 元组进不了 actor loss。
这不是「加噪不可行」,而是 **Q-learning 与 policy-gradient 的结构差异**。

### 2.1 三条路线

- **路线 1(忠实复刻附录 B.2)**:加 Q/critic。合成流用 B.2 原公式生成,按 Eq.4-9 训练
  critic(真实 TD + ρ(H_t)·w_Q·合成 TD),actor 经学到的 Q 更新。仓库已有 SAC policy
  worker、critic worker、pi05 value head 接口,主要是把 critic 训练接回 embodied runner。
- **路线 2(保持 critic-free,本方案 v1)**:把扰动写进采样 recipe(α 均值混合 + b 噪声
  放大),语义等价、公式不等价,改动最小,先回答「反事实放大有没有收益」。
- **路线 3(复用 NFT:更新时对保存链加噪 → WM → 新 reward)**:见 2.3,复用仓库已有
  NFT 的 resample + lagged rollout 记账,把重新去噪出的动作喂 WM 拿新 reward。

**合法实现:把「加噪」写进采样 recipe。** per denoising step 定义:

```
mean_t = α·a_real + (1−α)·x_t_mean(o_k)
std_t  = x_t_std(o_k)·b
x_t    = mean_t + ε·std_t
logp   = Gaussian(x_t | mean_t, std_t)
```

`get_logprob_norm` 只要求传入 (mean, std),所以这一步在现有代码里是参数替换。
两个极端:

- α=0:围绕「当前策略预测」加噪 → 状态空间反事实(纯 on-policy 放大噪声);
- α>0:采样均值向「真实执行动作」靠拢 → 动作空间反事实,越接近论文 q_σ,s
  「围绕真实动作做局部扰动」的语义。

**一致性约束(必须)**:rollout 采样与训练 recompute 必须用同一个 recipe,因此
`a_real、α、b` 要随 batch 一起存储,recompute 时按相同 (mean_t, std_t) 重算。
**闭环约束**:合成组策略在每个决策步观察 WM 生成帧(闭环)。纯开环「事后扰动整条动作序列」
的动作根本不是策略采样,无法进 surrogate。

### 2.2 路线 2 实现规格(代码落点)

- 采样:`openpi_action_model.py` 约 1122 行 `x_t = x_t_mean + noise·x_t_std`,
  改为 `mean_t = α·a_real + (1−α)·x_t_mean;std_t = b·x_t_std;x_t = mean_t + noise·std_t`;
- rollout logprob:同文件 1124 行 `get_logprob_norm(x_t, x_t_mean, x_t_std)` 同步传入
  `(mean_t, std_t)`;存储的 `chains` 即混合后的样本;
- recompute:`get_log_prob_value`(约 1391 行)按当前 θ 重算 mean/std 后,**必须再用同一
  `(α, b, a_real)` 混合**再算 `get_logprob_norm`;`a_real` 经 `forward_inputs` 传入
  `default_forward`(约 800 行);
- batch 字段:新增 `a_real`(合成组专属)、`alpha`、`noise_boost`;真实组 α=0 / a_real 置空,
  走原路径;
- 合成组 reset:WM env 以真实组选中轨迹的 o_0 与 context(帧、action)为条件,
  替代 LIBERO bridge 输出;轨迹长度对齐真实轨迹的已执行 chunk 数;
- 损失:`L = L_A + ρ(H_t)·L_B`,两组各自 `compute_grpo_advantages` 归一化,
  各自 clip;`ρ(H_t)` 用真实组状态上的 flow 熵,附录线性近似,ρ_max=0.30。

### 2.3 路线 3(更新时对保存链加噪 → WM → 新 reward)

仓库已有 NFT 机制(`fsdp_nft_policy_worker.py`),与用户描述的结构高度重合:

- rollout 保存 `nft_x0` / `nft_xnext` / chains,以及组归一化 advantage;
- update 时 `nft_xcur_source: resample` 把保存动作沿 flow 调度加噪:
  `xcur = (1−t)·x0 + t·randn`,注入分布显式为 `N((1−t)·x0, t²·I)`;
- 当前模型从 xcur 重新去噪;`nft_tau` 维护 lagged rollout 权重算 `v_old`,支持 off-policy;
- 损失为 advantage-weighted 的 DPO / MSE(softplus 或 `r·e_pos + (1−r)·e_neg`)。

用户方案 = NFT resample + 一步扩展:重新去噪出的反事实动作 ã 喂 WM → ResNet 新 reward
→ 合成组新 advantage → NFT loss(真实组继续 GRPO):

```
L = L_GRPO(真实组) + ρ(H_t)·L_NFT(合成组)
```

可行性:注入分布是显式高斯;重新去噪链的每步密度在 current 与 lagged rollout 权重下都可算
(NFT 的 `_recompute_v_old` 记账),因此 ã 的 proposal / target 密度完备。这是「利用保存链
在更新时加噪」的密度一致版本。**加噪必须沿 flow 调度做(xcur 插值),不能在最终动作上
直接加 ε——那会断开链与密度的对应。**

三个落地细节:

1. 合成组 loss 沿用 NFT 的 advantage-weighted DPO / MSE,不必硬塞回标准 clipped surrogate;
2. 8 条合成轨迹 = 8 条真实轨迹各 re-noise 1 次,或 1 条 × 8 次(不同 t、ε);
3. WM 生成不能堵在 actor update 上:在收集阶段异步生成,或 pipeline 一个迭代
   (与 WorldSample 的 async scheduling 一致)。

与路线 2 的关系:同族。路线 2 在采样时注入噪声(走标准 surrogate);路线 3 在更新时对
保存链加噪(走 NFT 记账)。路线 3 复用现有 NFT 代码更多,但 WM 时序必须异步化。

## 3. 两组损失与 PPL 对应

- 组内同质天然满足:两组各自独立归一化,不存在跨 reward 源的 σ 污染(此前组内混合的
  顾虑在此结构中自动消失);
- **Q-aware selection → 合成组的 reward-contrast selection**:0.65 阈值维护正负队列,
  约束组内正样本比例与真实组近期成功率差 ≤ δ、σ_group ≥ σ_min,σ≈0 的组丢弃重采样;
- **entropy scheduling**:H_t = 真实组状态上的 flow-noise 高斯熵(代码已算),
  ρ(H_t)=ρ_max/(1+κH^p),实现用论文附录线性近似,ρ_max=0.30,作用在「合成组系数」;
- **锚点选择**(哪条真实轨迹做反事实源):随机 / 组内最优 / 组内最差,对应论文正负样本
  队列的「传播成功」vs「学习恢复」,作为消融。

## 4. 失败模式对应

论文的 Bellman 累积 ε_syn → 我们的 ResNet 偏差经 advantage 放大 + 策略 reward-hack
WM 幻觉。对策:ρ_max 上限、早期熵抑制、正负平衡、真实 LIBERO eval 为最终裁决、
监控 σ_group / clip_fraction / KL drift / WM 帧漂移(复用
`evaluate_primary_wan_rollout_drift.py`、`eval_video_metrics.py`)。

## 5. 最小实现(v1)

1. **batch 扩展**:随轨迹存储 `a_real`(反事实源的真实动作)、α、b,与现有 chains /
   denoise_inds 对齐;
2. **采样与 recompute 双侧 blend+boost**:`openpi_action_model.py` 采样处与
   `fsdp_actor_worker.py` recompute 处同改(参数替换,各几十行);
3. **混合 env**:同一迭代跑 1 组 LIBERO + 1 组 WM(env mode mask;WM 组以选中真实轨迹
   的 o_0 及 context 为 reset 条件),当前 train 只有 WM env,这是最大改动;
4. **`_reset_from_replay_anchor`**:复用现有 conditioning 字段,把 bridge 输出换成
   选中轨迹的观测张量;
5. **selection + pacing + 锚点选择**;
6. **指标与配置**:H_t、ρ、σ_group、clip_fraction、两组各自成功率、WM 漂移;
   新增 `algorithm.ppl.*`(ρ_max、κ、p、δ、σ_min、α、b)。

## 6. 消融与验收

| 编号 | 设置 | 验证点 |
| --- | --- | --- |
| A | 纯真实组(LIBERO-only GRPO) | 真实基线 |
| B | 纯合成组(现状 WM GRPO) | 合成基线 |
| C | 配对组,ρ=0.3 固定,无 selection/pacing | 两组联合是否有净收益 |
| D | C + reward-contrast selection | σ_group 健康度、成功率 |
| E | D + entropy pacing(完整 PPL) | 早期稳定性、最终成功率 |
| F | α = 0 / 0.5 / 0.9 | 状态空间 vs 动作空间反事实 |
| G | b(噪声放大系数) | 反事实多样性 vs 稳定性 |
| H | 锚点选择(随机/最优/最差) | 成功传播 vs 失败恢复 |
| I | 分支深度 k(t=0 vs k>0) | 中段分支的覆盖增益 |

验收主指标:GOAL 任务 0/3/9 的真实 LIBERO eval 成功率;
稳定性:σ_group、clip_fraction、KL drift;反黑客:WM 帧漂移、ResNet 校准。

## 7. 与论文不一致处(诚实边界)

- 论文是真机闭环,我们的真实组是 LIBERO 仿真;预期收益是样本覆盖与多样性提升。
- 无 critic,Eq.3/5/7 只能同构迁移,不能逐字移植。
- 反事实噪声加在 flow 采样 recipe 上,而非「事后扰动已执行动作」;两者语义相近,
  但后者在 flow-matching 下没有可用的策略密度。
- 合成组 reward 与 WM 幻觉同源(同一条 ResNet),靠正负平衡、ρ 上限与真实 eval 缓解,
  而非彻底消除。
