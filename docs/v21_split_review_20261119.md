# V21 Muon/AdamW 优化器拆分 — 发车前代码审查

- 日期：2026-11-19
- 模式：preflight + review（代码/实验改动审查，含独立复现）
- 范围：`native/src/kernels.cu`、`native/src/muon.cu`、`native/model/ptcg_model.cu`、`native/model/ptcg_model4.cu`、`native/src/pufferlib.cu`、`native/src/bindings.cu`、`native/train_native.py`、`native/tools/native_weights.py`、`native/tests/test_adamw.cu`、`native/tests/muon_step_dump.cu`（工作区状态；repo 不是 git 仓库，无法 diff commit，只能按文件内容+工件审查）
- 规划基准：`docs/plan_v21_optimizer_20261119.md`（§3.1 参数分组 91-102、§3.2 超参与实现 104-111、§5 风险 141-148、§8.5 A0 清单 216-226、§9.2.8 工程细节 288、§11.1 状态/磁盘 410-414）
- 运行时基准：`logs/v21_split_smoke.log`、`logs/v21_sps_nosplit.log`、`logs/v21_sps_split.log`、`logs/v21_a0_lr1e4_aux3e4.log`、`logs/v21_a0_lr1e4_aux1e4.log`

## 0. 裁定摘要

| 对象 | 裁定 | 理由 |
|---|---|---|
| 代码实现（split 本身） | **GO（无 P0）** | AdamW 数学与 `torch.optim.AdamW` 对齐（独立复算 5 步误差 ≤7.5e-9）；split off 逐位一致（独立重编译对拍）；分组 85/35 与规划一致；gate/对齐/状态格式均有证据 |
| A0 四臂 smoke | **GO_WITH_NOTES（可继续）** | 代码不会使 A0 无效；但四臂在 KL guard 下 94–98% 的 optimizer step 被 veto，A0 结果不能直接当作“分组好坏”的判决（见 P1-1/P1-2） |
| B1（新池 + hybrid C1 1B/2B） | **NO_GO（暂不可发车）** | A0 预注册判据未通过（arm1 e10 ent 0.806/lgw 0.020；arm2 e10 ent 0.779/lgw 0.039；arm3 e1 lr 1e-5 仍 kl 0.098、klskip 124），且 aux lr 相对 Muon lr 的量级与所引 Muon 配方相差约 2 个数量级；split+cudagraph 路径从未运行；`adamw_lr` 未进日志。先补低 aux 臂 + capture smoke，再谈 B1 |

问题计数：**0 × P0，4 × P1，4 × P2，5 × P3（共 13 项；P3 中 2 项为记录性备注）**。

## 1. 审查方法与独立复现（全部为本轮实际执行）

### 1.1 split off 逐位一致（独立重编译，不信旧工件）

```
# 用 pristine（/root/autodl-tmp/v21_pristine/src，无 adamw 字样）与当前 src
# 分别编译同一个 muon_step_dump.cu，跑同一 8 步序列后比字节
nvcc -c -O2 -std=c++17 -arch=sm_89 -DPRECISION_FLOAT -I<pristine> muon_step_dump.cu -o dump_pristine.o
nvcc -c ... -I<repo>/native/src muon_step_dump.cu -o dump_current.o
./dump_pristine out_pristine.bin && ./dump_current out_current.bin
md5sum out_pristine.bin out_current.bin
→ f4bd07f75a957156b7db7908f9877d29  out_pristine.bin
→ f4bd07f75a957156b7db7908f9877d29  out_current.bin
cmp → 完全相同（"BIT-IDENTICAL"）
```

同时复跑了仓库已有工件 `build/dump_old` / `build/dump_new`，四份输出 md5 全为 `f4bd07f7…`，说明旧工件不是“顺手产物”，可复现。harness 覆盖 2-D tall/wide/square、1-D、对齐 gap、3 gated + 2 vetoed + 2 gated 步，输出 weights+momentum（`native/tests/muon_step_dump.cu:16-85`）。

全 run 逐位对拍不可行的原因也复核了：`experiments/native-v21_det_a/final.bin` 与 `native-v21_det_b/final.bin` 同 init、同配置，md5 分别为 `a41abf47…` / `0a36b0d8…`（不一致），确认引擎无种子（`ptcg/env.c:399`）导致两跑不可比。因此“split off 与旧 path 逐位一致”只能由单步 harness + 分支代码审查证明，不能由整跑 md5 证明。

### 1.2 AdamW 数学（与真实 torch 对拍，不只对手写 reference）

自建 GPU probe（`/tmp/v21_review/adamw_probe.cu`，用当前 `muon.cu`，单参数 3 元素，lr 0.003、betas (0.9,0.95)、eps 1e-8、wd 0.01、5 步常数梯度）：

```
kernel: 0.0849860683 -0.18497099 0.284955949
torch : 0.0849860758 -0.18497099 0.284955949   (torch.optim.AdamW, foreach=False, fused=False)
```

最大绝对误差 7.5e-9（float32 舍入级）。`build/test_adamw` 复跑 PASS：

```
test_adamw: PASS (torch-order AdamW over 5+1 steps, warm start, veto no-op, t counting, untagged m/v zero)
```

### 1.3 分组静态审计（直接审计 C 源码，不信 Python 镜像）

用 `nvcc -E` 预处理 `native/model/ptcg_model.cu`，从展开后的 `PT_PARAMS` 表逐项求值（相邻字符串字面量拼接、算术表达式求值），再套用 `pt_param_is_adamw` 的实际规则（`d.c == 0` + `ADAMW_NAMES`）：

```
groups: 120  entries 120  unique names 120
named 24  c0 rule present: True
C mask: adamw=85 muon=35 total=120
adamw elems 868680  muon elems 1437888  total 2306568
named not in table: none
muon non-2D: none
```

再把 C 表与 `native/tools/native_weights.py` 的 `param_order()/split_groups()` **逐 index** 比对（名称风格不同，按顺序+形状对齐）：

```
index-aligned mismatches (group or shape): 0
py adamw/muon 85 35   C 85 35
```

即：启动横幅的 85/35 不是“只信 Python 自报”，C 表与 Python 镜像当前确实一致。

### 1.4 对齐 / gate / 状态

- **对齐**：自建 probe（`/tmp/v21_review/align_probe.cu`）放一个 5 float 的未标记参数 + 一个 3 float 的 AdamW 参数，后者起始于 32B（跨 12B gap）：
  ```
  arena bytes 44 floats 11
  m: 0 0 0 0 0 0 0 0 0.001 -0.002 0.003
  v: 0 0 0 0 0 0 0 0 5e-06 2e-05 4.5e-05
  gap(m/v offs 5-7) clean=1 ; p1 offs 8-10 nonzero=1
  ```
- **gate/t**：`v21_split_smoke` 的 STA2 头可反推优化器步数：step 262144/epoch 2/t=6，step 524288/epoch 4/t=13；对应每 epoch 128 minibatch、klskip 125+125 与 124+125 → 6 与 13。A0 arm1 结束态：step 1310720/epoch 10/t=62 = 1280 − (125×6+124+122+119+103)=62。t 只计“被允许的 step”，且跨 STA2 resume 正确延续。
- **状态格式**：文件字节数与理论值精确相符：
  - STA1：`32 + 2×9,226,272 = 18,452,576`（`v21_sps_nosplit/state_latest.bin` 实测 18,452,576）；
  - STA2：`48 + 4×9,226,272 = 36,905,136`（`v21_split_smoke`、`v21_a0_*` 实测 36,905,136）。
- **旧代码读 STA2**：对 18:17 构建的旧 `.so`（`/root/autodl-tmp/v21_baseline/_C_baseline.so`，早于 18:19 的源码改动）反汇编 `load_state`：
  ```
  4736d: mov edx,0x20        ; fread 32 字节头
  47386: movabs $0x3141545347435450,%rax   ; "PTCGSTA1"
  47390: cmp %rax,0xa0(%rsp)
  47398: jne 47797           ; → " is not a PTCGSTA1 state file"
  ```
  即旧二进制会**明确拒绝** STA2（不会静默错读），行为安全。旧 `save_state` 同样是 32B 头，随后头、w、m 共 3 次 fwrite，再 fsync + rename。

### 1.5 环境与构建

- `python3 -m py_compile native/train_native.py native/tools/native_weights.py` → OK。
- `native/puffer_ptcg/_C.cpython-312-x86_64-linux-gnu.so`（19:01:04）与 `_C_new.so`（18:36:40）md5 相同（`2cfb4217…`），且都晚于全部源码改动（最晚 18:26），当前 A0 跑的是含本轮改动的 .so。
- `df -h / /root/autodl-tmp`：`/` 6.4G/30G，`autodl-tmp` 41G 可用；A0 三卡各 46,236 MiB 占用、GPU 0/1 100%，与 A0 三 rank 一致。

## 2. 实现证明表（预注册项 → 代码位置 → 运行时证据 → 状态）

| 预注册项 | 代码位置 | 运行时/独立证据 | 状态 |
|---|---|---|---|
| **a. `--optimizer-split` 默认 off 时旧 path 逐位一致**（allocator 布局/注册顺序、PT_PARAMS 顺序不变） | 默认：`train_native.py:290`（0）、`bindings.cu:762`（默认 0）；不注册状态：`pufferlib.cu:3018-3021`；walk 短路：`muon.cu:349`；注册只加元数据：`kernels.cu:455-462`；STA1 分支：`bindings.cu:338-347` | pristine vs 当前源码重编译 dump 逐字节一致（§1.1）；`v21_sps_nosplit` 正常；STA1 18,452,576 字节与旧格式一致；`diff pristine/current` 的 `kernels.cu` 只有 `use_adamw` 字段+marker，`muon.cu` 只有新增 kernel/字段/split 分支，walk 的旧分支仅 `if`→`else if` | **IMPLEMENTED** |
| **b. 分组与 §3.1 一致；120 项全覆盖** | `ptcg_model.cu:98-129`（表）、`144-161`（mask）；`ptcg_model4.cu:267-277`（注册时打标记） | 预处理 C 表静态审计：85 AdamW / 35 Muon / 120，`named not in table: none`，`muon non-2D: none`；C 与 Python 镜像逐 index 0 mismatch；启动横幅 85/35（`logs/v21_split_smoke.log:2`） | **IMPLEMENTED** |
| **c. AdamW 数学：decoupled wd + bias correction + eps 位置；torch 顺序** | `muon.cu:123-131`（prestep，t/inv）、`133-151`（step） | 真实 torch 对拍 5 步最大误差 7.5e-9（§1.2）；`build/test_adamw` PASS；62 步常数梯度 probe 得到 62×lr（无 10× 放大） | **IMPLEMENTED** |
| **d. gate 语义：veto 时 weights/m/v/t 全不动（branch 非 multiply）；prestep 与 per-param gate 一致** | `muon.cu:125`（prestep early return）、`138`（step early return）、`319-321`（每 step 一次 prestep）、`349-362`（per-param 同 `gate_ptr`） | STA2 头反推 t = minibatch − klskip（6/13/62 三处精确吻合，§1.4）；`test_adamw.cu:165-177` veto 后 w/t 不变；自造 multiply-gate 变体（/tmp）被单测以 `FAIL test_adamw.cu:176` 捕获 | **IMPLEMENTED**（m/v 断言缺口见 P2-3） |
| **e. arena offset 对齐（m/v 与 mb_puf 同一 `(offset+3)&~3` 游标；跨 16B gap 不错位）** | `muon.cu:339`（offset 对齐）、`357-358`（m/v 用同一 offset）、`kernels.cu:487-492`（alloc_create 16B） | 自建 gap probe：12B gap 后 AdamW m/v 精确落在 8-10，gap 5-7 全 0（§1.4）；`test_adamw` 的 OFF2=12 亦覆盖 | **IMPLEMENTED** |
| **f. DDP/NCCL：grads 全 arena allreduce + 全局 clip 在 walk 之前；m/v rank-local 一致性论证** | `muon.cu:301-304`（allreduce）、`307-312`（clip）、`318-322`（prestep 在 clip 后）、`349-362`；gate 的跨 rank 一致：`pufferlib.cu:2365-2374` | A0 三 rank 跑通（`v21_a0_*.log`），说明 NCCL 路径可用；**但没有任何 m/v 逐位/逐 rank 对照** | **UNVERIFIED**（代码论证充分，运行时未验证） |
| **g. cudagraph 捕获：adamw kernels 进 train graph 安全（gate/t 在 device）** | split 后 muon_step 仍在捕获区内：`pufferlib.cu:2283-2304`、`2376-2398`；新 kernel 全是 device-side：`muon.cu:123-151` | 所有已跑 split 证据（smoke/A0/SPS）全部 `--cudagraphs -1`；默认值是 10（`train_native.py:950`），split+capture **一次都没跑过** | **UNVERIFIED** |
| **h. state 兼容** | 头/保存/加载：`bindings.cu:272-353`、`355-457`；t 回读：`288-295`；STA2 仅在 split on 应用：`428-445` | STA1/STA2 尺寸精确（§1.4）；t 跨 resume 正确（6→13）；旧 .so 反汇编明确拒绝 STA2；STA1+split on 静默零 m/v（文档化的 warm start） | **IMPLEMENTED**（m/v 段 round-trip 未单测，见 P3-2） |
| **i. CLI 透传链** | `train_native.py:873-881` → `286-294`（build_args）→ `bindings.cu:762-766`（get_config_or）→ `pufferlib.cu:394-398`（HypersT）→ `3018-3021`（enable）→ `muon.cu:200-220`（字段/状态）→ `349-362`（使用） | 横幅 85/35 + aux 0.0003（Python 侧）；STA2 落盘证明 split 到了 C 侧；arm1 vs arm2 仅 aux 不同、轨迹不同（行为上说明 `--adamw-lr` 生效） | **IMPLEMENTED**（应用值未进日志，见 P1-4） |
| **j. 独立复算元素数/状态大小** | — | AdamW 868,680 + Muon 1,437,888 = 2,306,568；STA2 = 36,905,136 B；STA1 = 18,452,576 B（与实测文件逐字节数吻合） | **一致** |
| **k. 无法验证项** | — | 见 §5 | **UNVERIFIED** |
| **l. 规划外实现决策（oracle/aux 头归属）** | `ptcg_model.cu:144-161` 注释与名单 | 静态审计确认：`oracle_value_mlp` + `aux_*` → AdamW；`oracle_proj` + `critic_blocks.0`（`c0_`）→ Muon | **ALLOWED，需记录**（见 §4 D-20261119-01） |

## 3. 问题清单（按严重度）

### P0（阻断）
无。

### P1（会影响实验结论有效性，或默认路径无验证）

**P1-1 当前 A0 四臂在 KL guard 下 94–98% 的 optimizer step 被 veto，ent/lgw 结果不能当作“分组好坏”的判决。**
证据：arm1（`logs/v21_a0_lr1e4_aux3e4.log`）e1–e10 klskip 125/125/125/125/125/125/124/122/119/103，t=62（1280 minibatch），e10 ent 0.806、lgw 0.020；arm2（`logs/v21_a0_lr1e4_aux1e4.log`）e1–e10 klskip 119/120/120/120/121/121/121/117/109/85，e10 ent 0.779、lgw 0.039；arm3（`logs/v21_a0_lr1e5_aux3e4.log`，审查时 e1）lr 1e-5、aux 3e-4，e1 kl 0.0981、klskip 124。arm1/arm2 均已 FAIL 预注册判据（ent ≤0.5、lgw ≥0.65 @e10，规划 §8.5#2），且 arm1 比 V20 无 split 基线 #6（e10 ent 0.679、lgw 0.131）更差。原因是每次被放行的 AdamW step 很大（见 P1-2），guard 挡不住策略在 epoch 内的跳变。**影响**：四臂跑完也不能回答“hybrid 是否降低漂移”。**建议**（不改代码）：保留已完成臂作为“高 aux 负例”，补 aux ≤3e-5 的臂；日志里同时报 `klskip/128` 与 `t`；必要时对 A0 单开一臂 `--kl-skip 0` 只看纯优化器动力学（需用户拍板，因为会改变 guard 条件）。

**P1-2 aux lr 相对 Muon lr 的量级与所引 Muon 配方相差约 2 个数量级，AdamW 组在实测中明显主导。**
证据：规划 §1.2 引用的配方是 Muon 0.02 / aux 3e-4（比值 ≈67×），而 A0 是 Muon 1e-4 / aux 3e-4~1e-4（比值 0.33×~1×），B1 规划是 Muon 1e-3 / aux 3e-4（0.3×）。实测 arm1（t=62）init→final 分组位移：AdamW 参数加权平均 |Δ|=7.08e-4、最大 2.03e-2；Muon 加权平均 1.69e-4、最大 5.10e-3；`readout` 平均 |Δ|=1.37e-2、`lnf_w` 9.18e-3（62 步几乎走满 62×3e-4 的 sign 步）。arm3（Muon lr 1e-5 = arm1 的 1/10，aux 仍 3e-4）e1 的 kl 0.0981、klskip 124，比 arm1 的 kl 0.0682 更高：把 Muon 侧压小 10 倍并没有降低漂移，主导项是 AdamW 组。**影响**：分组效应与“aux 过大”混淆；即使 A0 某臂通过，也不能外推到 B1 的 1e-3 Muon lr。**建议**：B1 前把 aux 与 Muon lr 的比例纳入 sweep（例如 Muon 1e-4 时 aux 1e-5/3e-5；Muon 1e-3 时按比例再降），并把“1-D 是否留在 AdamW”作为后续单变量 A/B（1-D 的 sign 步对 LN/readout 很激进）。此为 NEEDS_USER（改变超参搜索空间）。

**P1-3 split + cudagraph 捕获路径从未运行，而默认 `--cudagraphs` 是 10。**
证据：`train_native.py:950` 默认 10；A0 脚本（`/root/autodl-tmp/tmp/a0_smoke.sh`）与 SPS 全是 `--cudagraphs -1`；`csrc` 侧新 kernel 均在捕获区内（`pufferlib.cu:2283-2304`、`2376-2398`），host 分支只依赖固定的 `optimizer_split`/`use_adamw`。**影响**：若 B1 忘记带 `-1`，会第一次在真实 2B 任务上走未验证的 capture 路径。**建议**：B1 前跑 2 epoch `--optimizer-split --cudagraphs 10` smoke；若失败，回退 `-1` 并在文档记录。

**P1-4 应用后的 `adamw_lr` 没有进入 console/wandb，`docs/training.md` 的 “logged as losses/adamw_lr” 与实现不符。**
证据：`bindings.cu:42-48` 把 `adamw_lr` 放在 log dict **顶层**；`train_native.py:767` 的 wandb `flat` 只取 `logs["lr"]`，`774` 只摊平 `logs["loss"]`；console 行（`train_native.py:701-713`）无 adamw 字段。`docs/training.md:297-299` 声称 “logged as `losses/adamw_lr`”。**影响**：A0/B1 无法从日志确认实际 aux lr（尤其余弦/暖启动时），只能靠启动横幅猜。**建议**：在 `flat` 加 `{"adamw_lr": logs.get("adamw_lr")}` 并在 console 行加一列；同步修正 `training.md`。

### P2（影响运维/复核，不影响当前数值正确性）

**P2-1 STA2 为全部 120 个参数存 m/v（Muon 的 1,437,888 个元素恒为 0），状态文件 36.9MB，是必要值的 ~1.45 倍。**
证据：`bindings.cu:311-320` 用 `arena_host_to_packed_blob` 收集整个 params 表（含未标记参数）；`save_state` 写 4 段 packed（w+m+am+av）。36,905,136 B vs 若只存 AdamW 元素约 32+9,226,272+6,949,440=16.2MB？按当前格式：w 9.23MB + muon m 9.23MB + am/av 18.45MB = 36.9MB；若 am/av 只存 868,680×2×4=6.95MB，可降到 ~25.4MB。`STATE_HISTORY` 仍是 10（`train_native.py:209`），split run 的 state 历史 = 369MB（规划 §11.1 的 10→3 尚未落地）。**影响**：磁盘预算；不影响正确性（Muon 的 am/av 不读）。**建议**：B1 前把 STATE_HISTORY 降到 3，或按 tagged 元素存稀疏段（需版本号 bump）。

**P2-2 “SPS −0.5%” 不是干净的 split 开销测量。**
证据：`v21_sps_nosplit` e1：SPS 3752、kl 0.0113、klskip 2；`v21_sps_split` e1：SPS 3732、kl 0.1049、klskip 124。两臂同 init（md5 `57f6076a…`）、同 lr，但 split 臂实际只做了 4 次 optimizer step（t=4，STA2 头验证），no-split 臂由 klskip=2 推断约 126 次（STA1 无 t 字段，无法直接读）；单 epoch、20 SPS 差异。**影响**：−0.5% 只能当“同配置下的墙钟指示”，不能当“split 本身开销”。**建议**：在 t 相近的两臂（或 `--kl-skip 0`）复测；跑 2–3 epoch 取稳定值。

**P2-3 单测没有直接断言 veto 后 m/v 不变；`docs/training.md:306` 与测试摘要的“weights/m/v/t untouched”略微夸大覆盖。**
证据：`test_adamw.cu:165-177` veto 后只比较 weights 和 t，未重读 `adamw_m_puf/adamw_v_puf`。我用 /tmp 的 multiply-gate 变体验证：单测会以 `FAIL test_adamw.cu:176`（weights 变化）捕获该错误，所以缺口很窄；但若未来只改 m/v 的写入路径，单测不会红。**建议**：veto 前后各 `cudaMemcpy` 一次 m/v 并 CHECK 相等（2 行）。

**P2-4 启动横幅的 85/35 只来自 Python 镜像，没有对 C 侧 tag 的运行时交叉校验。**
证据：`train_native.py:1041-1045` 调 `native_weights.split_groups()`；C 侧 `pt_param_is_adamw` 只在注册时打 tag，不打印。今天我用预处理的 C 表证明两者逐 index 一致（§1.3），但这是审查时的一次性静态审计；未来若 C 名单打错字，横幅不会发现，实验会静默换组。**建议**：启动时从 C 侧返回 tag 计数（或在 `ptcg_enc_reg_params` 里 assert 85/35），与 Python 镜像交叉校验。

### P3（记录/可维护性）

**P3-1** `muon_init` 的 `max_M/max_N` 扫描包含 AdamW-tagged 的 2-D 参数（如 `aux_hand_w` 1448×256），NS scratch 按 256×1448 分配；split 下这些参数不再走 Muon，属 ~1.5MB 的过度分配（`muon.cu:256-275`）。无正确性影响。

**P3-2** 没有 STA2 save/load 的单测；端到端 smoke 只证明了 t 的 round-trip（6→13）和文件尺寸，m/v 段的 round-trip 是代码对称性论证。

**P3-3** 两个新测试没有接进 `build_native.sh` 或 Makefile，只能按注释里的手动命令构建；复现依赖注释（`test_adamw.cu:1-11`、`muon_step_dump.cu:1-9`）。

**P3-4** 旧 `.so` 读 STA2 会报 `is not a PTCGSTA1 state file`（安全，已反汇编确认），但 `docs/training.md:213-223` 只写了“新代码兼容旧文件”，没写“旧二进制拒绝新文件”；建议补一句，避免有人用旧 .so 续 split run。

**P3-5** `adamw_eps` 硬编码 1e-8（`muon.cu:206`），与 torch AdamW 默认一致；未暴露 CLI。记录即可。

## 4. 偏差台账（规划 vs 实现）

| ID | 日期 | 阶段 | 规划文档/要求 | 实际实现 | 判定 | 证据 | 处理与文档落点 |
|---|---|---|---|---|---|---|---|
| D-20261119-01 | 2026-11-19 | A0/B1 | §3.1 表只列了 `score_mlp`/`value_mlp` 为输出头，未提 `oracle_value_mlp`、`aux_hand/aux_prize`、`oracle_proj`、`c0_` | `oracle_v1/v2`、`aux_hand`、`aux_prize` → AdamW；`oracle_proj`、`critic_blocks.0`（c0_）→ Muon | **ALLOWED**（按“输出头→AdamW、hidden 2D→Muon”的一致延伸；不是另一套假设） | `ptcg_model.cu:144-161`；静态审计 85/35；`docs/training.md:293-297` 已按此描述 | 在 V21 规划 §3.1 表下加一行“oracle/aux 头的归属按同一规则：loss 读出头→AdamW，critic 隐藏投影/块→Muon”；把“若改成 oracle/aux→Muon”记为另一个实验，不与本轮 A/B 混跑 |
| D-20261119-02 | 2026-11-19 | A0/B1 | §3.2/§6#2 未定“AdamW 是否与 Muon 共享 cosine” | 共享 schedule 形状：`adamw_lr = base × (applied_lr / base_lr)`，暖启动也乘进去 | **ALLOWED**（规划 §6 明确留待决定；实现选了“同形”） | `pufferlib.cu:2154-2162`；`docs/training.md:297-298` | 在 B1 预注册里写明 aux 也走 cosine+暖启动；若 B1 想固定 aux，需新开关（NEEDS_USER） |
| D-20261119-03 | 2026-11-19 | A0/B1 | §3.2 “state 增加 AdamW m/v 段（版本号 bump 或追加 section）” | 新 magic `PTCGSTA2` + 48B 头 + w/m/am/av/t；旧 magic 仍读；split off 写 STA1 | **ALLOWED**（规划允许“版本号 bump”） | `bindings.cu:272-353`；文件尺寸精确；旧 .so 拒绝 STA2 | 已在 `docs/training.md:213-223` 记录；补“旧二进制拒绝新格式”一句（P3-4） |
| D-20261119-04 | 2026-11-19 | A0/B1 | §9.2.8 要求“A0 单测逐项过” | 单测覆盖 math/warm start/veto/t/untagged；未覆盖 STA2 round-trip、NCCL、cudagraph | **PARTIAL**（可继续 A0，B1 前补） | `test_adamw.cu`；A0 全 `cudagraphs -1`；无多 rank m/v 对照 | A0 期间补 capture smoke；B1 前补 2-rank m/v 校验（可复用现有 dump harness 思路） |

## 5. UNVERIFIED / 缺什么证据

| 项 | 状态 | 缺的证据 | 建议的验证 |
|---|---|---|---|
| DDP 下 m/v 逐位一致 | UNVERIFIED | 没有两 rank、N 步后的 m/v/weights 对照 | 写一个 2-rank NCCL 小 harness（同 `muon_step_dump`，mpi/spawn 两进程），比较 rank0/rank1 的 m/v/weights md5；或在一轮 3-rank smoke 里加一个“每 rank 权重 checksum”日志 |
| split + cudagraph 捕获 | UNVERIFIED | A0/SPS 全 `-1` | 2 epoch `--optimizer-split --cudagraphs 10` smoke，检查无 capture 报错、SPS/VRAM/klskip 正常 |
| STA2 的 m/v 段真被恢复（不只 t） | PARTIAL | 只证明了 t（6→13）；m/v 靠代码对称 | 在 smoke 的 resume 前后比较 `adamw_m/v` 与“从零重算”的差异；或加一个 `load_state` 后 dump m/v 的 debug 入口 |
| PT_PARAMS 顺序“相对改动前”不变 | PARTIAL | 无 git/旧源码快照；只能从行为/镜像推断 | 用旧 `.so` 与新 `.so` 各加载同一 init.bin，`debug_forward` 同一 obs，比较输出（需在空卡上做，避免与 A0 抢显存） |
| full-run split off 逐位一致 | 不可验证 | 引擎无种子（`ptcg/env.c:399`）+ `libcg` 用 `std::random_device`，同 .so 两跑 final md5 已不同 | 已用单步 harness 替代；文档已说明，不再追求整跑 md5 |
| 旧代码读 STA2 的具体报错文本 | 已验证（反汇编+strings） | — | — |

## 6. 独立复算（load-bearing 数字）

| 数字 | 复算 | 对照 | 结论 |
|---|---|---|---|
| 总参数 | C 表逐项求和 = 2,306,568 | `final.bin` 9,226,272 B / 4 = 2,306,568；init 横幅 2,306,568 | 一致 |
| AdamW 元素 | 868,680（16 个 2-D + 69 个 1-D） | Python `split_groups()` 868,680 | 一致 |
| Muon 元素 | 1,437,888（35 个 2-D） | Python 1,437,888 | 一致 |
| arena floats | 2,306,581（含 52B padding） | `params->total_bytes/4` = 9,226,324/4 | 一致 |
| STA1 大小 | 32 + 2×9,226,272 = 18,452,576 | `v21_sps_nosplit/state_latest.bin` 18,452,576 | 一致 |
| STA2 大小 | 48 + 4×9,226,272 = 36,905,136 | `v21_split_smoke`/`v21_a0_*` 36,905,136 | 一致 |
| A0 arm1 优化器步数 | 1280 − 1218 = 62 | STA2 头 `0x3e` = 62 | 一致 |
| smoke resume t | 6（e2）→ 13（e4） | 头字段：262144/2/6、524288/4/13 | 一致 |
| SPS | 3752 vs 3732（−0.5%） | `v21_sps_nosplit/split.log` e1 | 数字一致，但口径受限（P2-2） |

## 7. 发车清单与建议

### 7.1 A0（已启动，可继续）

- [x] split off 逐位一致（单步 harness，独立重编译）
- [x] 分组 85/35（C 静态 + Python 镜像逐 index 一致）
- [x] AdamW 数学（真实 torch 对拍）
- [x] gate/t/对齐（单测 + 探针 + STA2 头反推）
- [x] STA1 旧格式可读、旧二进制拒绝 STA2
- [ ] KL guard 口径：把 `klskip/128` 与 `t` 打进每个 epoch 行（当前 t 不在 console，需要读 state 头才能算）
- [ ] 低 aux 臂（≤3e-5）与“1-D 留在 EMA”对照（NEEDS_USER：改变超参/分组）
- [ ] capture smoke（`--cudagraphs 10`）
- 结论：**GO_WITH_NOTES**；现有 arm1/arm2 已 FAIL、arm3 e1 仍 kl 0.098/klskip 124，四臂结果只能作负例/机制证据，不能作“split 有效”的判决。

### 7.2 B1（新池 + hybrid C1）

**NO_GO**，直到：
1. A0 出现一个 `ent ≤0.5 && lgw ≥0.65 @e10` 且 `klskip < 5%` 的臂（或用户明确接受“guard 全开”的 regime 作为新基线）；
2. aux lr 与 Muon lr 的比例按 §1.2 配方重估并预注册（NEEDS_USER）；
3. `--cudagraphs 10` smoke 通过（或 B1 明确用 `-1` 并写入 runbook）；
4. `losses/adamw_lr`（或 console 列）可观测；
5. 2-rank m/v 一致性校验通过。

在满足 1–2 之前发 B1，会在“未验证的 aux 量级 + 未验证的 capture 路径”上烧 24h/臂，且结果无法归因到“hybrid vs Muon”。

## 8. 本次文档改动

- 只新增本文件：`docs/v21_split_review_20261119.md`。
- 未修改代码、未改其他文档、未启动/停止任何进程、未 stage/commit（repo 非 git）。
