# OSCAR INT2 for Qwen3.5-27B / vLLM Ascend

外部适配实现，目标为本仓库参考树的 **vLLM Ascend 0.23.0 + PR #12607
（19e436985）**，配套 vLLM 0.23.0。方案、公式、流程图和实施点见
[设计文档](docs/design.md)。参考源码没有修改，扩展不导入 `references/`。

实现包括 Triton Ascend rotation、clip/INT2 store、分页 split-KV attention、
BF16 sink/recent/current、LSE merge 和 vLLM general plugin。
主模型 FULL 层使用 OSCAR，GDN 和 MTP draft 层保持原生；MTP target verification
的多个候选 query 在同一 Triton program 内复用 KV tile。

**当前为待真机验证的实现。** 本地 CPU 检查不代表 NPU 编译、端到端启动、图回放、
精度或“不慢于原生”已经通过。没有提供目标机器连接或 Qwen3.5-27B 的专用 rotation。

## 真机一键启动

先进入已安装上述 Ascend/vLLM、torch_npu 和 Triton Ascend 的 NPU 容器或虚拟环境。
从 GitHub 下载后执行下面一组命令，最后一条会自动安装扩展、预检并启动服务：

```bash
git clone --branch codex/oscar-ascend https://github.com/jpl123123/Oscar_new.git
cd Oscar_new
VLLM_OSCAR_K_ROTATION_PATH=/absolute/path/qwen35_27b_k_rotation.pt \
VLLM_OSCAR_V_ROTATION_PATH=/absolute/path/qwen35_27b_v_rotation.pt \
bash scripts/serve.sh
```

这两个路径必须替换为 **Qwen3.5-27B D256 校准矩阵**；包含全局层号
3、7、…、63，格式兼容 OSCAR PR。缺层、D128 或非正交矩阵会报错，
不会以 identity 冒充校准结果。作者当前公开的
[RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo/tree/main)
列出了 Qwen3.5-4B/35B-A3B，不能据此视为已提供 27B 的矩阵。

脚本仅安装当前外部包并校验环境，不安装/替换 vLLM、torch_npu 或 Triton。
`PYTHON_BIN` 可指向已有虚拟环境 Python；`MODEL`、`PORT` 可覆盖默认值。
默认保留用户的 TP4、262144 context、MTP3、async 和 FULL_DECODE_ONLY 参数。
只预览命令：`DRY_RUN=1 bash scripts/serve.sh`。

默认模型目录为 `/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp`，
服务监听 `0.0.0.0:8989`，模型服务名为 `qwen3.5`。如现场 Python 环境不是
默认的 `python3`，在启动命令前加入 `PYTHON_BIN=/absolute/path/to/python`；
模型目录不同时加入 `MODEL=/absolute/path/to/model`。

GitHub 仓库提供外部适配代码，原生参考树用
`oscar_ascend/upstream_fingerprints.json` 固定版本，不作为子模块分发。
这条命令不负责安装 CANN、驱动、原生 Ascend/vLLM 或生成校准矩阵。

本版本有两个明确限制：

- **关闭 prefix caching**。BF16 ring 以请求首个物理页为 owner，不能让不同请求
  共享后并发覆盖；检测到 prefix sharing 会拒绝运行。
- **保留原生页容量与分配机制**。INT2 减少 history 的读取字节，空余空间作为 padding；
  不能声称此版本释放了 6.4 倍显存或提高了可分配 context 容量。
  实际页数由原生 memory profiling 决定，当前步 scratch 也会占用显存预算。

如果需要排查图问题，可临时使用 `OSCAR_ENFORCE_EAGER=1`，但 eager 结果不等于
用户要求的 FULL_DECODE_ONLY 性能验收通过。

## 测试与 A/B 对照

```bash
# CPU 和接口检查，NPU 测试缺环境时明确 skip
python -m pytest -q

# 真机：缺 NPU 直接失败，不允许以 skip 冒充通过
bash scripts/test_npu.sh

# 分别启动三个模式，须先自行停止前一个服务，脚本不杀进程
MODE=native bash scripts/serve.sh
MODE=native-prefix-off bash scripts/serve.sh
# OSCAR 模式使用上面带 rotation 路径的启动命令

# 每个模式启动后，在另一个终端执行，对应修改 LABEL
LABEL=native bash scripts/bench.sh
LABEL=native-prefix-off bash scripts/bench.sh
LABEL=oscar bash scripts/bench.sh

# 与原生命令对比，任何必要指标回退或结果缺失均返回非零
python tools/compare_benchmarks.py artifacts/native artifacts/oscar \
  --output artifacts/performance-vs-native.json
# 分离 prefix caching 设置的影响
python tools/compare_benchmarks.py artifacts/native-prefix-off artifacts/oscar \
  --output artifacts/performance-vs-prefix-off.json
```

基准覆盖短/长 prompt、continuation、并发和 MTP，每场景默认三次。
比较相同 workload 的 output throughput、TTFT/TPOT/P99 和 MTP 接受率。
该检查不代替质量评估；还需在相同模型、相同固定题集上测 GSM8K/长上下文任务，
以及 NPU profiler 的 AiCPU、同步和内存流量。用户提供的瞬时日志不能替代配对 A/B。

开发机测试依赖：`python -m pip install -e '.[test]'`。
真机已有 Torch/NPU 环境只需补充 pytest；不要用 CUDA Triton wheel 替换 Triton Ascend。

## 文件

| 文件 | 作用 |
|---|---|
| `docs/design.md` | 方案、地址公式、MTP/生命周期、流程图、性能验收 |
| `oscar_ascend/plugin.py` | 幂等注册、FULL 路由、保留 native draft |
| `oscar_ascend/backend.py` | serving 前向调度 |
| `oscar_ascend/metadata.py` | 固定地址 NPU 元数据 |
| `oscar_ascend/kernels.py` | 实际 Triton kernel 实现 |
| `oscar_ascend/layout.py` | 三条内存上的字节地址与容量检查 |
| `oscar_ascend/rotations.py` | 严格 rotation 文件加载 |
| `oscar_ascend/check.py` | 源码指纹、版本、模型、NPU 预检 |
| `scripts/serve.sh` | 一键启动 |
| `scripts/test_npu.sh` / `scripts/bench.sh` | 真机测试 / 配对基准 |
| `tests/` | CPU oracle、隔离测试、真实 NPU kernel/graph 测试 |

上游数值依据为 [OSCAR PR #46774](https://github.com/vllm-project/vllm/pull/46774)，
实现针对 D256 单组：K68 + V68 = 136 bytes，外层对齐到160 bytes。
