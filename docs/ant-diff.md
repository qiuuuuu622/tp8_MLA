# 蚂蚁代码 diff：已移植什么，还能借鉴什么

蚂蚁参考：[380e2d5](https://github.com/antgroup/sglang/commit/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47)，父提交 `778bacdf3f0efe7eaa5ea23d3b11353b098b8548`。本仓库默认核心保持此前 `ab2f5a8`，本轮新增真实 TP8 证据及 grid 消融。

## 模型调用链的变化

蚂蚁新增 `deepseek_v4_head_a2a.py`，并向 `_compute_q_b`、`_forward_prepare`、`forward` 传递同一个 row plan。原实现每卡处理全部行、部分head并补齐；新实现把head汇聚到行所属卡，做完整head的attention，再把O送回head所属卡。

触发条件包含 decode/verify、TP>1、足够行数和支持的head形状；确定性模式等保留原路径。计划还要与共享indexer结果的consumer层一致，不能某层突然改成不同的行切分。

蚂蚁阈值 `_MIN_ROWS=96`、`_ROW_Q_MIN_ROWS=288` 来自H20-3e。本次H20 96GB上，ratio1/2的Graph在32/64行已有收益，eager在32/64行仍慢；纯SWA的96/128行只有微弱改善。因此不能照搬阈值。

来源：[模型行切分实现](https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/srt/models/deepseek_v4_head_a2a.py)。

## 通信 kernel 的主要机制已经借鉴

| 机制 | 蚂蚁 | 当前核心 |
|---|---|---|
| 搬运 | SM peer store直写对端最终布局 | 相同思路 |
| 布局 | 源peer/行、目标sender/行四个stride | 相同思路 |
| 同步 | 每CTA 128B信号、进入与发布两阶段 | 同一协议，另加10秒诊断超时 |
| 位模式 | 数据与信号分离 | 相同；NaN/Inf按int16比较 |
| 搬运粒度 | 16B向量，unroll=4 | 相同；固定8192B的8-head分段 |
| 分工 | 相邻warp轮转peer，各rank从下一peer开始 | 相同 |
| 参数与启动 | grid-constant参数结构、PDL支持 | 设备指针数组、普通launch |
| 检查 | 源/目标范围、对齐、容量 | 固定Python布局保证源范围，C ABI检查目标范围 |

进入阶段保证各卡已经结束上一轮读取，才允许覆盖；完成阶段先同步CTA写入线程，再用system release/acquire发布和观察写入。不能用单卡stream顺序代替跨卡可见性保证。PDL和参数结构是实现差异，尚无本仓库实测证明其单独收益。

来源：[segment_a2a](https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/kernels/jit/csrc/distributed/segment_a2a.cuh)、[Barrier/Semaphore](https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh)。

## 紧凑 buffer 比直接照搬 grid 更值得尝试

蚂蚁交换边界使用 `[M,8,512]`，row侧是 `[ceil(M/8),64,512]`，返回也是紧凑8head；预留最大batch容量，Q消费完后复用同一个区接收O，并支持容量检查和NCCL fallback。

当前输入已补齐为 `[M,1,64,512]`，Q/O两个plane独立，恢复O也保留64-head stride。只传8有效head，但分配空间和地址跨度大。M512时当前Q/O对称数据区合计36MiB（4+32）；紧凑单区约4MiB，均不计信号和独立attention输出。**减少9倍的是这部分缓冲容量，不能说NVLink传输量也减少9倍。** 应在投影输出和O投影之间接入，避免额外pack抵消收益。

蚂蚁使用64 CTA×512 threads，当前默认1–16 CTA×256。本次仅扩展实验host grid上限，保持payload、同步、布局不变：

| 行数 | 当前默认双向搬运 μs | 32×256 μs | 64×512 μs |
|---:|---:|---:|---:|
| 8 | 10.27 | 9.58 | 10.48 |
| 32 | 11.10 | 10.77 | 11.10 |
| 96 | 13.92 | 13.80 | 14.21 |
| 128 | 15.85 | 14.95 | 15.78 |
| 288 | 22.37 | 22.79 | 22.86 |
| 512 | 32.45 | 31.75 | 32.05 |
| 1024 | 55.65 | 54.15 | 55.01 |

这是transport-only Graph P50，取最慢rank，不是attention函数延迟。单轮、固定扫描顺序，百分之几差异不能认作稳定收益。默认与固定16×256在部分行数实际上相同，测量仍有波动。64×512没有明显普遍优势，因此不改默认grid。6组grid×7行数均完成并通过布局与往返检查。

来源：[SymmHeadA2A](https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/srt/distributed/device_communicators/head_a2a.py)、[本次消融原始结果](../results/20261102-h20-tp8/tp8-grid-scan.json)。

## 更进一步：省掉 Q 交换

蚂蚁在已有完整 `wq_b` 的路径，让每卡直接为本卡行计算全部head，省去Q交换；O仍回传。`wqkv_a`的Q分支也只算本卡行，KV分支仍算全部行，因为各卡保留完整KV。

普通vLLM TP8的 `wq_b` 是ColumnParallelLinear分片。`fused_wqa_wkv`设置`disable_tp=True`，不意味着`wq_b`完整复制。checkpoint完整`wq_b=[32768,1280]` FP8约40MiB，每卡分片约5MiB；复制完整权重将每层每卡增加约35MiB，40层约1.37GiB，尚未计scale和临时空间。

仅Q-b投影而言，每卡总FLOPs大体相同：从“全部行×1/8列”换成“1/8行×全部列”，每卡权重读取量却增大。收益来自省通信与更合适的GEMM形状，不能说Q-b计算减少8倍。小batch可能更慢，需将投影、归一化、RoPE和通信一起测。这项尚未在本仓库实现或实测。

## indexer 行切分和服务接入

蚂蚁将相同 `row_slice` 下传到indexer，只为本卡attention行打分和选top-k；consumer层须复用一致的行范围。切分可能穿过请求边界，row_offset不能当作request_offset。

本仓库提前生成稀疏索引，只测attention和交换，未包含indexer收益。建议顺序是：紧凑Q/O边界与动态metadata → indexer行切分 → 完整wq_b省Q交换，各项独立A/B。

服务还需一致dispatch、独立row scheduler、预分配、拓扑/P2P检查和Graph生命周期管理。当前模块要求调用方保持正确CUDA device，不能并发复用workspace，也不能某个rank单独回退。

另一个已知限制：原始`summarize.py`未拒绝不同`vary_lengths`输入。本次主性能两组均为false，变化窗口独立存放，未混合汇总。接口加固应独立修改并注明实测代码版本，不能将旧数据直接当成修改后实测。
