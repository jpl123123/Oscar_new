# 验证记录 — 2026-09-08

状态：**本地实现与检查完成；NPU / 端到端 / 精度 / 性能验收未完成**。

## 实际运行

环境：macOS arm64，Python 3.12，独立 `.venv`，CPU Torch 2.14.0。
没有 `torch_npu`、Triton Ascend 或可连接的 NPU 测试机配置。

| 检查 | 结果 |
|---|---|
| `python -m pytest -q --junitxml=artifacts/local-tests.xml` | **76 passed, 2 skipped**；skip 为 serving 和 calibration 两个真实 NPU 测试模块 |
| `ruff check oscar_ascend tests tools` | 通过 |
| `python -m compileall -q oscar_ascend tests tools` | Python 语法通过；不等于 Triton JIT 编译通过 |
| `bash -n scripts/serve.sh scripts/test_npu.sh scripts/bench.sh` | 通过 |
| native / native-prefix-off / oscar 启动脚本 dry-run | TP4、MTP3、async、FULL_DECODE_ONLY 参数及带空格模型路径检查通过 |
| 外部包 editable 安装和 entry point 查询 | `oscar_ascend.plugin:register` 已注册到 `vllm.general_plugins` |
| 目标源码指纹检查 | vLLM `0fc695fc`、Ascend `19e436985` 的关键接缝文件匹配 |
| 两个参考 Git 仓库的 `status --porcelain` | 均为空，没有改动原生文件 |
| `pytest tests/test_npu_kernels.py --require-npu -q` | **退出码4**：`torch_npu is missing; NPU acceptance did not run` |

CPU 测试覆盖数值 oracle、字节地址/隔离、环形缓存回滚、配置、插件路由、
启动参数和性能对照的缺结果/回退/NaN 拒绝逻辑。插件路由测试使用 mock，
没有在 Mac 上冒充加载完整 vLLM NPU 服务。

v0.2.0 新增：缺失自动生成/有效缓存复用/模型和校准数据指纹/生成失败不发布/
显式文件保留/不同run ID拒绝复用；两遍校准 RPC 顺序及释放 engine；
QQT/SST 和 Jacobi 的 CPU 数学 oracle；一键脚本成功与校准失败两条路径。
生成器/模型/一键启动的控制流程测试使用 fake，不代表真实模型已生成 `.pt`。
设备测试故意传入 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`，确认所有运行入口及子进程
强制使用 `4,5,6,7`，已添加 `AGENTS.md` 保存该限制。

v0.2.1 新增：现场完整版本串的归一化、预检和实际插件注册回归；拒绝错误 release；
源码差异作为审计信息、缺接口仍阻止启动；检查 base `5cb98caaa` 与额外 PR 参考树
在适配关键接缝文件上的一致性。带 `torch 2.10.0+cpu` / TorchNPU / Triton 的预检测试
使用 mock，仅验证判断流程，没有在 Mac 上执行真实 NPU API。

## 尚未运行的必要检查

- 实际 Triton Ascend 编译与执行；独立 NPU tests 包含旋转 stride、INT2 bytes、
  non-monotonic 页表、sink/recent 去重、long continuation、MTP 拒绝回滚、
  不同请求数/页表的 NPUGraph replay，以及 metadata 固定地址测试。
- 指定 Qwen3.5-27B-w8a8-mtp 模型的四卡启动、混合 GDN/FULL、MTP、async 和图模式。
- Qwen3.5-27B 专属离线校准矩阵的现场校验及端到端质量测试。
- 自动生成路径中 Triton 协方差、FP32 Jacobi、Hadamard/Pbr 和双遍真实模型运行。
- 同请求集、相同输出长度、相同硬件的三轮配对 benchmark 和 NPU profiler。

没有生成或填充任何虚构 NPU 性能数据。“不得更慢”尚未证明。
prefix caching 关闭、原生 page 容量不缩小、额外 current-token scratch 的开销，
均需在现场验收中计入；不能用理论 slot 压缩率代替实际吞吐或显存收益。
