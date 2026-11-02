# TP8 MLA：DeepSeek-V4.1-Flash / 8×H20

在 vLLM 0.32.0 的原版 FlashMLA 上，通过 **head↔row 交换 + NVLink 最终布局直写**减少每卡 8 个有效 head 补齐到 64 head 的执行浪费。这里是独立 kernel / 函数级实验包，不包含完整 vLLM 源码，不会自动改写 `vllm serve`。

**真实 8×H20 96GB 已测。** ratio=2、32K 上下文、CUDA Graph 下：128 行 2.56×，288 行 3.24×，512 行 4.34×。1/8 行仍然更慢。数字是 attention 加双向通信的函数延迟，不是完整模型吞吐量、TTFT 或 TPOT。

## 模型与计算形状

已读取实际挂载 checkpoint 的 config、分片索引与 safetensors 文件头，未加载整套权重。模型为 `DeepseekV41ForCausalLM`：40 个主干层，hidden=5120，64 个 Q head、1 个 KV head、head_dim=512（448 NoPE + 64 RoPE），q_lora_rank=1280；SWA=128，index_topk=512。主干层 ratio0/2/1 分别为 2/18/20 层。

权重头验证：`wq_b=[32768,1280]` FP8，`wq_a=[1280,5120]` FP8，`wkv=[512,5120]` FP8，sink=[64] FP32。测试始终是 **TP8，每卡 8 个有效 head**，不是每卡 64 个有效 head。EP/MoE 不参与本次函数测试。

```text
原版：每卡 M 行 × 8 有效 head → 补齐 64 head → FlashMLA
优化：每卡 M 行 × 8 有效 head
      → Q peer push
      → 每卡 ceil(M/8) 行 × 64 有效 head → 同一个 FlashMLA
      → O peer push → 恢复本卡 M 行 × 8 head
```

原版没有 all-to-all。NCCL 对照是“打包→all-to-all→重排→attention→打包→all-to-all→恢复”；融合版把数据直接写入接收卡最终布局，去掉中间重排。搬运采用 SM 发出的 NVLink store，不是 copy engine；没有增加激活量化。每个 CTA 使用独立 128 字节信号区、两阶段跨卡同步，GPU 维护 epoch，支持 CUDA Graph 重放。

## 实测性能

2026-11-02，8×H20 96GB / NV18，vLLM 0.32.0、torch 2.13.0+cu130、Triton 3.7.1、**实际 NCCL 2.30.7**、nvcc 13.0。驱动 550.54.15，容器提供 CUDA compatibility 580.95.05。测试使用相同的环境开关：

```bash
NCCL_IB_DISABLE=1
NCCL_NVLS_ENABLE=0
TORCH_SYMM_MEM_DISABLE_MULTICAST=1
OMP_NUM_THREADS=1
```

默认 NCCL 初始化曾卡住；关闭 IB/NVLS 后 all-reduce 通过。PyTorch 对称内存 rendezvous 另有卡住，关闭其 multicast 后通过。这里没有进一步把 IB 与 NVLS 的影响逐个隔离；不能宣称它们在所有 H20 上都必须关闭。预装包未降级：`ncclGetVersion` 和 NCCL 日志确认 23007；`torch.cuda.nccl.version()` 返回的 2.29.7 与实际动态库不同。

下表为 ratio=2、32K 上下文、每 query SWA128+512 稀疏位置、Graph。取 8 个 rank 各自 P50 中的最大值；30 样本，每样本 20 次调用。NCCL 与融合方案按顺序分轮执行，原版列来自融合方案同轮对照。

| Query 行数 | 原版 μs | NCCL 交换 μs | 融合交换 μs | 融合 / 原版加速 |
|---:|---:|---:|---:|---:|
| 1 | 16.15 | 38.46 | 26.49 | 0.61× |
| 8 | 20.54 | 41.43 | 26.88 | 0.76× |
| 32 | 47.58 | 50.98 | 27.41 | 1.74× |
| 64 | 59.07 | 59.67 | 32.87 | 1.80× |
| 96 | 94.59 | 66.67 | 39.71 | 2.38× |
| 128 | 118.39 | 80.90 | 46.21 | 2.56× |
| 192 | 166.74 | 95.99 | 55.58 | 3.00× |
| 288 | 233.19 | 113.70 | 71.94 | 3.24× |
| 384 | 291.23 | 141.40 | 86.60 | 3.36× |
| 512 | 407.77 | 155.89 | 93.91 | 4.34× |

融合总时间包含 Q/O 双向交换、布局写入、FlashMLA 和结果恢复。512 行双向通信/布局：NCCL 90.62 μs，融合 32.99 μs；128 行：43.10→15.16 μs。独立测量项不能简单相加代替总函数耗时。

**适用范围：** ratio1 与 ratio2 的 Graph 结果接近；纯 SWA 的收益较小，96/128 行只有约 1–2%，不能据此认为稳定加速。eager 的 host 开销和波动更明显，例如 ratio2 的 32/64 行仍慢于原版；路径选择必须区分 attention 类型、行数与执行模式。完整 eager/Graph、ratio0/1/2 结果见 [CSV](results/20261102-h20-tp8/tp8-comparison.csv)。

输入和缓存是合成数据，KV 在所有卡复制且索引映射一致。workspace 初始化、sink 汇聚、metadata 准备、投影、indexer、KV 更新、MoE、调度器及整模型推理均不计时；planner/缓存预热、重复输入。没有完整模型质量或端到端性能结论。

## 正确性与实验边界

- 11 种行数：1/7/8/9/32/64/127/128/129/256/512；eager + Graph，8 rank，各 200 轮，共 **35,200 个 rank 往返，全部逐位一致**。
- 输入变化包含任意 BF16 位模式、NaN/Inf，以 int16 视图比较；覆盖尾部、空 rank、延迟 rank，以及各 rank 配置不一致时的集体拒绝。
- 30 种函数配置分别测试 NCCL 与融合方案，均通过反量化 FP32 attention 参考检查。两种交换方案相对原版最大相对 L2 都是 **0.218486%**。改变行数使 FlashMLA 的 split-KV 调度/归约顺序变化；通信复制本身没有数值损失。
- 额外 12 种变化窗口/空窗口及非整除行数配置通过。该批 repeat=3、inner=2，仅作为正确性证据，不用于性能选型。
- CPU 测试覆盖 TP2/4/8 布局和独立 attention 数学参考。
- 本次未运行 8 卡 Compute Sanitizer；不把历史 TP2 memcheck 当作 TP8 的结果。未测试完整服务生命周期、多流并发复用或真实模型质量。

## 与蚂蚁代码的关系

核心源自此前提交 [ab2f5a8](https://github.com/qiuuuuu622/vllm_dsv41_flash_opt/commit/ab2f5a8d5774a85c89b0a450f53632898525734c)，搬运与同步借鉴蚂蚁 SGLang 的固定提交 `380e2d5`，保留 [许可证及来源说明](vllm_dsv41_head_exchange/csrc/NOTICE.md)。

详见 [逐层差异与迁移判断](docs/ant-diff.md)。本轮另做 **6 组 grid × 7 种行数**消融；直接换成蚂蚁的 64 CTA × 512 threads 没有稳定明显收益，因此默认核心代码不变。该实验仅扩大 host grid 上限，未移植完整蚂蚁运行时，放在 `experiments/` 中单独复现。

## 复现

需要 Linux、8 张具备完整 NVLink/P2P 互联的 SM90 GPU，以及兼容的 CUDA 驱动/工具链。无须完整模型权重。在仓库根目录执行：

```bash
uv venv --python 3.12 --system-site-packages
# 若机器没有匹配的 torch：按 CUDA 13.0 官方 wheel 源安装 torch==2.13.0。
uv pip install --python .venv/bin/python numpy pytest
# kernel 实验只需官方 wheel 的 FlashMLA 扩展，不用启动完整服务。
uv pip install --python .venv/bin/python vllm==0.32.0 --no-deps
bash scripts/02-prepare.sh
bash scripts/01-verify.sh
bash scripts/03-function-ab.sh
# 可选：变化窗口检查及蚂蚁 grid 消融
bash scripts/06-additional-checks.sh
```

`--system-site-packages` 只为复用已验证的机器 CUDA/PyTorch 环境，不能让不兼容环境自动变兼容。上述安装命令没有重新解析整个服务依赖栈。源码与动态库要匹配，初始化和编译必须在 Graph capture 外完成。

Mac/CPU：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python torch pytest
.venv/bin/python -m pytest -c tests/kernels/head_exchange/pytest.ini tests/kernels/head_exchange -q
```

`FusedWorkspace(q, sink, group=tp_group)` 接收 BF16 `[M,1,64,512]`，仅前 8 head 有效；通过 `forward_q()`、FlashMLA 写入 `row_o`、`backward_o()` 使用。输入指针固定，metadata 更新后需刷新尾部副本；同一 workspace 串行使用，所有 rank 调用顺序一致，调用方必须保持正确 CUDA device。结果缓冲在下一次调用覆盖。独立 row scheduler、拓扑检查、一致 dispatch 与服务集成仍需实现，不能单 rank 自行回退。

## 文件

- `vllm_dsv41_head_exchange/`：与 ab2f5a8 一致的默认核心。
- `benchmarks/kernels/head_exchange/`：原版、NCCL、融合方案 A/B 和通信验证。
- `scripts/`：准备、TP8 测试与消融命令，输出到 `build/tp8-results/`。
- `experiments/`：64-CTA 上限实验，默认路径不导入。
- `results/20261102-h20-tp8/`：原始 JSON、日志、CSV、模型结构和环境证据。

本次未恢复已消失的模型副本；函数测试从 /root 工作目录执行，测试结束 GPU 进程已退出。仓库不含模型权重、共享库二进制或 SSH 凭据。
