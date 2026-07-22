# `chunk_gated_delta_rule_fwd_h`：Triton 与 Ascend C 实现差异

本文对比以下两个前向状态扫描实现：

- FLA Triton：`fla/ops/common/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- `flash-linear-attention-npu` Ascend C：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/`

两者实现相同的 chunk 间状态递推，但数据布局、并行切分、门控约定、状态精度和硬件流水差异很大。NPU 实现不是 Triton Kernel 的逐行翻译，而是针对 Ascend Cube/Vector 架构进行的重构。

## 1. 算子在 GDN 中的位置

该算子不是完整的 Gated Delta Rule 前向，而是 chunk 间状态传播阶段。

```text
k / v / beta / gate
        |
        v
chunk 内 WY 变换
生成 w、u
        |
        v
chunk_gated_delta_rule_fwd_h
生成 chunk 起始状态 h、修正值 v_new 和最终状态
        |
        v
chunk_fwd_o
生成最终输出 o
```

其中 `u` 是 WY 变换后的 value，接口中的 `v` 或 `u` 均不是原始模型输入 `v`。

## 2. 共同的数学语义

对于第 `c` 个 chunk，定义：

- `H_c`：该 chunk 的起始状态，形状为 `[K, V]`
- `W_c`：WY 表示中的 `w`，形状为 `[BT, K]`
- `U_c`：WY 表示中的 `u`，形状为 `[BT, V]`
- `K_c`：该 chunk 的 key，形状为 `[BT, K]`

首先去除旧状态通过 `W_c` 产生的贡献：

```math
R_c = U_c - W_c H_c
```

输出的 `v_new` 是未施加 chunk gate 的 residual：

```math
V_{new,c} = R_c
```

对于标量 gate，令 `g_{c,last}` 为 chunk 内最后一个有效 token 的累计 gate：

```math
\widetilde{R}_{c,t} = R_{c,t} \exp(g_{c,last} - g_{c,t})
```

```math
H_{c+1} = \exp(g_{c,last}) H_c + K_c^T \widetilde{R}_c
```

两边都在更新前保存 `h[c] = H_c`，供后续输出 Kernel 计算该 chunk 对应的输出。

## 3. 总体差异

| 维度 | FLA Triton | NPU Ascend C |
|---|---|---|
| 输入主布局 | BTHD：`[B, T, H, D]` | BNSD：`[B, H, T, D]` |
| `h` 布局 | `[B, NT, HV, K, V]` | `[B, HV, NT, K, V]` |
| 执行结构 | 单个 Triton program 完成一个状态扫描流 | AIC Cube 与 AIV Vector 组成四阶段流水 |
| 时间维 | program 内顺序遍历所有 chunk | scheduler 保证同一 batch/head 的 chunk 顺序 |
| V 维切分 | `BV=32/64`，由 Triton autotune 选择 | Catlass V128/V256 tile 路径 |
| K 维切分 | 静态拆成最多四个 64 维块 | Cube tile 的 K 维矩阵乘 |
| 运行状态 | FP32 寄存器常驻 | GM 中的低精度 `h` 加 FP32 临时 workspace |
| 中间结果 | 主要留在 program 内 | 使用多个 per-core ping-pong workspace |
| 同步 | program 内顺序天然保证依赖 | Cube/Vector 之间使用跨核 event |
| 配置选择 | Triton autotune | host tiling、tiling key 和 scheduler |
| 状态转置布局 | 支持 `STATE_V_FIRST` | 当前仅支持 `transpose_state_layout=False` |
| 可选 `v_new` | 支持不保存 | 当前要求 `save_new_value=True` |
| 标量 gate 指数 | `exp2` | 自然指数 `exp` |

## 4. FLA Triton 实现

### 4.1 Grid 与任务粒度

Triton grid 为：

```python
(triton.cdiv(V, BV), N * HV)
```

一个 program 负责：

```text
一个 sequence
x 一个 value head
x 一个 V 维 tile
x 该 sequence 的所有时间 chunk
```

时间 chunk 不能独立并行，因为 `H_{c+1}` 依赖 `H_c`。Kernel 因此在一个 program 中顺序遍历 `NT`。

### 4.2 状态寄存器

K 维被静态拆成最多四段：

```text
b_h1: K[0:64]
b_h2: K[64:128]
b_h3: K[128:192]
b_h4: K[192:256]
```

每段状态使用 FP32 寄存器。每个 chunk 内依次计算：

```text
保存 h[c]
v_residual = u - w @ h
保存 v_new = v_residual
对 residual 和旧状态应用 gate
h = h + k.T @ gated_residual
```

这种设计减少了 chunk 间状态的显存读写，但 K/V 较大时会产生较高寄存器压力。当前 wrapper 限制 `K <= 256`。

### 4.3 Head 映射

Triton 实现支持 grouped value attention：

```text
key_head = value_head // (HV // H)
```

多个 value head 可以共享一个 K head，而 `w`、`u` 和状态仍按 value head 独立。

## 5. NPU Ascend C 实现

NPU 实现把每个 chunk 分成四个阶段。

### 5.1 Cube 1：计算 `W @ H`

```text
v_work = W_c @ H_c
```

Cube 将 FP32 累加结果写入 `vWorkspace`。

### 5.2 Vector 1：计算 residual 和 gate

```text
v_new = U_c - v_work
gated_v_new = v_new * exp(g_last - g)
```

Vector Core：

- 将未施加 gate 的 `v_new` 写入正式输出；
- 将施加 gate 的值写入 `vUpdateWorkspace`，供第二次 Cube GEMM 使用。

### 5.3 Cube 2：计算状态增量

```text
h_update = K_c.T @ gated_v_new
```

结果以 FP32 写入 `hWorkspace`。

### 5.4 Vector 2：更新状态

```text
H_next = exp(g_last) * H_current + h_update
```

更新后的状态写入下一个 chunk 对应的 `h`，或在最后一个 chunk 写入 `final_state`。

### 5.5 Scheduler

NPU scheduler 以 `(batch, value_head)` 为主要任务，并建立：

```text
key_head = value_head // (HV // H)
```

同一任务的 chunk 必须保持顺序，但不同 batch/head 任务可以分配到不同 core。实现还维护两个 ping-pong stream，使不同任务的 Cube/Vector 阶段能够交叠。

## 6. Gate 数值约定

这是直接对拍两个底层实现时最重要的差异。

### 6.1 标量 `g`

FLA GDN 在调用 Triton `fwd_h` 前，通常通过 `chunk_local_cumsum` 将自然对数 gate 乘以：

```math
1 / \ln 2
```

Triton Kernel 随后使用：

```text
exp2(g_last - g)
exp2(g_last)
```

NPU `fwd_h` 接口当前要求 `use_exp2=False`，Kernel 使用自然指数：

```text
exp(g_last - g)
exp(g_last)
```

若输入是从 Triton `fwd_h` 入口直接 dump 得到的 `g`，送入 NPU 前应转换：

```python
g_npu = g_fla * math.log(2)
```

若输入来自更上游的原始自然对数 gate，则应分别遵循两条完整调用链的 cumsum/scale 约定，避免重复转换。

### 6.2 `gk` 路径

两边 `gk` 都采用 base-2 语义：

```math
2^{gk_{last}}
```

NPU 通过先乘 `ln(2)` 再调用硬件 `Exp` 实现 `exp2`。

KDA 路径传入的 `k` 已经是：

```math
kg = k \odot 2^{gk_{last} - gk_t}
```

因此 `fwd_h` 只对旧状态应用 `2^{gk_last}`，不能再次对 `kg` 施加 chunk 内 decay。

## 7. 精度差异

### 7.1 Triton

- 运行状态 `b_h*` 使用 FP32。
- 状态在处理所有 chunk 期间一直保留在寄存器中。
- `h[c]` 按输入 dtype 写出，但下一 chunk 不会重新读取该低精度副本。
- `final_state` wrapper 固定分配为 FP32。
- GEMM 输入通常转换到 FP16/BF16，以使用 tensor core，累加保持 FP32。

### 7.2 NPU

- Cube GEMM 和主要 Vector 运算使用 FP32 workspace/累加。
- chunk 起始状态保存在 `h` GM 中，dtype 与主输入相同，通常为 FP16/BF16。
- 下一 chunk 的 `W @ H` 会重新读取低精度 `h`。
- `final_state` dtype 与 `initial_state` 相同；没有初始状态时默认为 FP32。

因此长序列下可能出现以下差异：

```text
Triton:
FP32 H_c -> FP32 H_(c+1) -> FP32 H_(c+2)

NPU:
FP32 update -> cast FP16/BF16 -> GM -> reload -> FP32 update
```

NPU 的 chunk 间量化误差可能逐步累积。这是比较 `h` 和 `final_state` 时需要重点区分的误差来源。

## 8. Workspace 与数据流

Triton 主要将中间值保留在 program 内。NPU 则为每个 AIC core 和两个 ping-pong stage 分配：

| Workspace | dtype | 用途 |
|---|---|---|
| `vWorkspace` | FP32 | 保存 `W @ H` |
| `vUpdateWorkspace` | 输入 dtype | 保存 gated residual |
| `kDecayWorkspace` | 输入 dtype | `gk` 路径保存已完成 decay 的 `kg` |
| `hWorkspace` | FP32 | 保存 `K.T @ gated_v_new` |
| sequence/chunk metadata | INT64 | varlen scheduler 使用 |

NPU 用更多 GM 流量换取 Cube/Vector 分工和流水并行；Triton 用更多寄存器占用换取较少的中间显存访问。

## 9. Dense 与 varlen

### Triton

- 输入使用 packed BTHD。
- `cu_seqlens` 提供每个序列的 `bos/eos`。
- wrapper 根据 `cu_seqlens` 生成 `chunk_offsets`。
- `h` 的 chunk 维在逻辑上压平，`boh` 指向每个序列的首个 chunk。
- 最后一个不完整 chunk 依靠 `boundary_check` 和有效 token mask 处理。

### NPU

- 输入使用 packed BNSD，varlen 时 shape batch 当前为 1。
- host tiling 和 scheduler 从 `cu_seqlens`、`chunk_indices` 建立 token/chunk offset 元数据。
- metadata 被放入 GM/UB workspace，scheduler 为每个逻辑序列建立顺序扫描流。
- 最后一个 chunk 通过 `blockTokens` 记录实际 token 数。

## 10. API 和支持范围

| 能力 | Triton | NPU 当前接口 |
|---|---|---|
| FP16/BF16 输入 | 支持 | 支持 |
| FP32 gate | 支持 | 支持 |
| FP32 initial state | 支持 | 支持 |
| GVA：`HV != H` | 支持，要求整除 | 支持，要求整除 |
| `state_v_first` | 支持 | 不支持 |
| 不保存 `v_new` | 支持 | 不支持 |
| 无标量 gate | 支持 | `g`/`gk` 至少一个存在；仅 `gk` 时 host 生成中性 `g` |
| `g + gk` | 支持 | 当前源码支持 |
| `K` | 最大 256 | README 与 host 实际校验范围存在差异 |
| `V` | 通过 `BV` 泛化，受资源限制 | host 要求 `V <= 256`，按 V128/V256 选择 tiling key |
| chunk size | wrapper 参数化 | README 声明 64/128，host 校验未完全体现该约束 |

NPU README、ACLNN 参数检查和 Kernel 实际泛化范围目前并非完全一致。边界形状应通过实际 NPU 测试确认，不能只依赖文档声明。

## 11. 性能瓶颈侧重点

### Triton 更可能受限于

- 长序列上的 chunk 串行依赖；
- 小 batch/少 head 时 program 数不足；
- K/V 较大时状态寄存器压力；
- K=256 时四个 FP32 状态块造成的 occupancy 下降；
- 每个 chunk 写出 `h` 带来的显存流量。

### NPU 更可能受限于

- `h`、`vWorkspace`、`vUpdateWorkspace` 和 `hWorkspace` 的 GM 流量；
- Cube/Vector 跨核事件等待；
- batch/head 任务数不足导致 AIC 利用率不高；
- 最后一个不完整 chunk 的 tile 利用率；
- V256 路径的 tile 和 UB/L1 资源压力；
- varlen metadata 读取和调度开销。

## 12. 跨仓对拍建议

建议按以下顺序比较：

1. 固定 `B=1`、`H=HV=1`、`K=V=128`、`chunk_size=64`。
2. 使用 FP32 `initial_state`，先仅运行两个 chunk。
3. 对齐布局：BTHD 与 BNSD 之间显式 transpose。
4. 对齐标量 gate：若使用 FLA `fwd_h` dump，执行 `g_npu = g_fla * ln(2)`。
5. 先比较第一个 chunk 的 `v_new`，它不包含 gate。
6. 再比较第一个状态更新 `h[1]`。
7. 分别比较 `W@H`、gated residual 和 `K.T@gated_residual`。
8. 对长序列误差区分一次 GEMM 误差与 chunk 间状态量化累积。
9. 最后扩展到 GVA、varlen、不完整 chunk、V256 和 `gk`。

不要只比较最终 `final_state`。最终状态混合了所有 chunk 的 GEMM、gate、cast 和状态递推误差，难以定位首个分歧点。

## 13. 结论

两个实现的数学目标一致，但采用了两种相反的硬件策略：

```text
Triton
  一个 program
    -> FP32 状态常驻寄存器
    -> 顺序处理所有 chunk
    -> 较少中间 GM 访问

Ascend C
  host tiling + scheduler
    -> Cube: W @ H
    -> Vector: residual + gate
    -> Cube: K.T @ residual
    -> Vector: decay + state update
    -> workspace 与跨核事件连接流水
```

跨实现比较时最关键的差异是：

1. BTHD 与 BNSD 布局不同。
2. 标量 `g` 分别使用预缩放后的 `exp2` 和自然指数 `exp`。
3. Triton 将 FP32 状态跨 chunk 保留在寄存器中，NPU 通常通过低精度 `h` GM 传递状态。
4. Triton 的性能风险主要是串行链和寄存器压力，NPU 的性能风险主要是 workspace 流量和 Cube/Vector 同步。
5. NPU 的公开约束、host 校验和 Kernel 实际支持范围仍需保持同步。

## 14. 源码位置

FLA：

- `fla/ops/common/chunk_delta_h.py`
- `fla/ops/gated_delta_rule/chunk.py`
- `fla/ops/kda/chunk_fwd.py`

`flash-linear-attention-npu`：

- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h.cpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/block/block_scheduler_gdn_fwd_h.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/block_epilogue_gdn_fwdh_update.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h`
