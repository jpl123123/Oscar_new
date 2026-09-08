# CPU 回读与同步审计

状态：最近真机 FULL decode 启动阻塞。候选修改没有成功真机验收记录。
“可以 dispatch 到 graph”、kernel 提交返回、dummy capture 返回、CPU 测试通过，
都不能证明真实请求 graph replay、模型质量或性能合格。

| 位置 | 数据与 CPU 行为 | 范围与处理 |
|---|---|---|
| 原生 `gdn_attn.py:530` | 对设备查询长度执行 `.cpu()`，等待前序设备工作 | v0.2.8 外部 hook 使用已有 CPU 查询边界计算 host draft 分类，去掉这一 D2H；保留 NPU 接受 token 计数和原生 GDN build |
| `metadata.py` initial prefill | 对 `query_start_loc_cpu` 调用 `.tolist()`，提供原生 FIA 要求的 host 序列边界 | 输入本来就在 CPU，有设备类型检查；没有 NPU→CPU 回读，不处理 Q/K/V 数值 |
| `metadata.py` / `_prepare_tasks_kernel` | host 只读取原生 `max_seq_len` 整数上界；请求任务映射与可达页表列复制在 NPU 完成 | 固定任务表/页表地址；不把设备 query boundaries 读回 CPU，dummy 不再清整表 |
| `backend.py` / `kernels.py` 正常前向 | 未发现显式 `.cpu()`、`.item()`、`.numpy()`、`.tolist()` 或 CPU 量化回退；Triton workspace 在输入 NPU 设备分配 | 这是源码结论，不能证明下层原生算子没有 AiCPU 回退；需真机 profiler 验证 |
| `startup.py` 首次图外预热 | 对当前流显式 synchronize，分别排空前序工作和确认新 kernel 完成 | 诊断行为，不是性能优化；捕获内和正常请求不增加这些同步，`OSCAR_STARTUP_TRACE=0` 可关闭 |
| `rotations.py` / `prepare_rotations.py` | CPU 文件读取、矩阵形状/正交性和缓存归属验证，然后将 rotation 传入 NPU | 加载阶段的文件校验，不是每步 attention；没有写回或改损 `.pt` |
| `calibration_worker.py` | `.item()` 读取权重和标量；最终 rotation/eigenvalues `.cpu()` 后校验并保存 `.pt` | 校准阶段同步确实存在，最终文件序列化需要 host 数据；不在 serving 热路径 |
| `calibration_kernels.py` Jacobi 收敛 | v0.3.0 通过 Triton 汇总统计，每轮判断只 `.item()` 读取一个最终残差 scalar | 消除原来至多512个 scalar 的 D2H 与 CPU 汇总；host 停止条件仍有一次同步，不能称全流程零 CPU |
| vLLM/Ascend 原生 metadata | CPU 上的 `.item()`/`.tolist()`、host 调度、H2D 更新；还存在原生流同步 | 保留原生框架并不代表已经完成性能审计；需按实际 tensor 所在设备区分 host 计算与 D2H，不能靠函数名判定 |

本次 GDN capture 优化仅用于 dummy capture：runner 在 CPU 上构造查询边界并复制到
设备，此时 CPU 副本是权威来源。不会在真实 async decode 中用可能滞后的 CPU 长度
替代设备状态，也不会把实际 K/V、attention 或量化搬到 CPU。

dummy counts 为零是无请求占用 KV 页时的保护；它不能验证非空 history、MTP 回滚、
真实页表切换、完整图 replay 的数值正确性。即使 dummy capture 返回，也必须完成
真实请求、混合 GDN/FULL、MTP、长上下文及 graph replay 的对照才能宣称跑通。

必须保留的失败事实：现场同步等待尚无已验证的唯一 kernel 根因。
删除 D2H 只能消除该回读，不能让未完成的设备队列自动恢复。
