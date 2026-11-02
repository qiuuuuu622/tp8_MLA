# 2026-11-02 H20 TP8 实验

运行结果：completed。

默认核心固定来自 `vllm_dsv41_flash_opt@ab2f5a8`。8×H20 96GB，TP8每卡8有效head，实际运行NCCL2.30.7。`executed_scripts/`保存实际执行脚本；仓库`scripts/`是改为仓库相对路径后的复现入口。

- `tp8-stress.json`：22个行数×模式配置、176个rank记录、各200轮，共35200个rank往返。
- `tp8-nccl.json`、`tp8-fused.json`：各30配置，8rank，eager/Graph完整样本。
- `tp8-comparison.csv`：原版列来自融合方案同轮；NCCL另轮。
- `tp8-vary-lengths.json`：12个空窗口/变化长度/非整除行数正确性配置，少量计时不用于性能判断。
- `tp8-grid-scan.json`：6组grid×7行数；实验只扩展host grid上限，不替换默认核心。
- `model-structure.json`：实际config和checkpoint文件头，未加载完整权重。
- `initial-environment.json`、`test-environment.json`：环境、开关、源文件哈希。
- `evidence.json`：核心和原始结果SHA256。

前置失败：默认NCCL检查卡住，关闭IB/NVLS后通过；默认对称内存rendezvous卡住，关闭其multicast后通过。初始探针导入旧路径`vllm.model_executor.models.deepseek_v41`失败，实际v0.32模型在`vllm.models.deepseek_v41`；FlashMLA实际函数已成功运行。

驱动550.54.15，CUDA compatibility库580.95.05，nvcc13.0，未改驱动或预装NCCL。pip check并非全通过，不能宣称完整服务依赖栈已验证。模型复制约50%时目标目录消失，用户选择只做函数测试，未恢复复制；实验不依赖该副本。GPU测试进程已经退出。

未做8卡sanitizer、整模型准确率或服务吞吐测试。未将TP2历史数据混入TP8结果。
