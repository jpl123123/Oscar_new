# OSCAR INT2 for Qwen3.5-27B / vLLM Ascend

外部适配实现，开发参考为 **vLLM Ascend 0.23.0 + PR #12607（19e436985）**，
配套 vLLM 0.23.0。运行时允许 Ascend **0.23.0 / 0.23.1**，包括开发版和本地
构建后缀，并检查适配所需接口；例如现场的
`vllm-ascend 0.23.1.dev0+g5cb98caaa.d20260822` 与 `vllm 0.23.0+empty`。
方案、公式、流程图和实施点见
[设计文档](docs/design.md)。参考源码没有修改，扩展不导入 `references/`。

实现包括 Triton Ascend rotation、clip/INT2 store、分页 split-KV attention、
BF16 sink/recent/current、LSE merge 和 vLLM general plugin。
主模型 FULL 层使用 OSCAR，GDN 和 MTP draft 层保持原生；MTP target verification
的多个候选 query 在同一 Triton program 内复用 KV tile。

**当前为待真机验证的实现。** 本地 CPU 检查不代表 NPU 编译、端到端启动、图回放、
精度或“不慢于原生”已经通过。v0.2.0 已实现缺失 rotation 时的自动校准流程，
该流程的实际 NPU 编译和模型运行也需要现场验证。

## 真机一键启动

先进入已安装上述 Ascend/vLLM、torch_npu 和 Triton Ascend 的 NPU 容器或虚拟环境。
从 GitHub 下载后执行下面一组命令，不需要手动填写 `.pt` 路径：

```bash
git clone --branch codex/oscar-ascend https://github.com/jpl123123/Oscar_new.git
cd Oscar_new
bash scripts/serve.sh
```

**所有脚本固定执行 `export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`。**
服务、校准子进程和真机测试都只允许使用物理后四卡，外部同名环境变量不会覆盖它。
限制生效后，进程内逻辑 NPU 0..3 对应物理卡4..7。

启动脚本依次执行：

1. 安装当前外部扩展，检查目标运行环境与后四卡可见性。
2. 按模型指纹和校准数据配置查找 K/V `.pt`，校验模型归属、层号、D256 和正交性。
3. 文件不存在或自动缓存失效时，加载同一 W8A8 模型，使用原生 BF16 KV 做两遍校准，
   在 NPU 上计算 QQT/SST、特征分解和 `U H Pbr`，生成 K/V `.pt`。
4. 验证生成结果、保存缓存，释放校准模型与进程的显存。
5. 用生成或复用的 `.pt` 启动 OSCAR 服务。

默认缓存位于 `artifacts/rotations/<模型指纹-校准配置指纹>/`。
首次运行需要额外校准时间，后续启动会直接复用有效缓存。
默认校准使用随包提供的16段中英混合文本，每段最多1024 tokens，无需下载数据集。
这是一套启动用的校准数据，**不等于模型质量验收通过**；也可通过
`OSCAR_CALIBRATION_DATA=/path/to/prompts.jsonl` 使用实际业务文本。
完整方法、配置和失败处理见 [自动校准说明](docs/calibration.md)。

已有外部矩阵时，仍可选择设置 `VLLM_OSCAR_K_ROTATION_PATH` 和
`VLLM_OSCAR_V_ROTATION_PATH`。显式指定的已有文件如果无效会报错并保留原文件；
自动缓存损坏则重新校准。生成失败或矩阵校验失败时，脚本停止，不启动服务。

脚本仅安装当前外部包并校验环境，不安装/替换 vLLM、torch_npu 或 Triton。
`PYTHON_BIN` 可指向已有虚拟环境 Python；`MODEL`、`PORT` 可覆盖默认值。
默认保留用户的 TP4、262144 context、MTP3、async 和 FULL_DECODE_ONLY 参数。
只预览命令：`DRY_RUN=1 bash scripts/serve.sh`。

默认模型目录为 `/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp`，
服务监听 `0.0.0.0:5656`，模型服务名为 `qwen3.5`。如现场 Python 环境不是
默认的 `python3`，在启动命令前加入 `PYTHON_BIN=/absolute/path/to/python`；
模型目录不同时加入 `MODEL=/absolute/path/to/model`。

GitHub 仓库提供外部适配代码，原生参考树用
`oscar_ascend/upstream_fingerprints.json` 记录开发参考指纹，不作为子模块分发。
运行时源码差异记录在预检报告的 `source_audit` 中，不因构建标签或整文件哈希不同
直接拒绝。v0.2.3 的接口预检只用 AST 读取已安装源码中的方法参数与元数据字段，
不导入原生 attention/device/ops 模块，不主动触发 Ascend patch 初始化。
实际类参数在挂钩时校验，KV geometry 在绑定原生 Tensor 时校验。
`--sources-only` 保留严格哈希审计，仅用于维护本地只读参考树。
Triton Ascend 驱动的目标名称 `npu`（例如 `Ascend910B4, warp_size=0`）是有效值；
预检也兼容使用 `ascend` 名称的发行构建。`warp_size=0` 不按 CUDA 的 warp 规则拒绝。
这条命令负责安装外部扩展和准备校准矩阵；CANN、驱动、原生 Ascend/vLLM
和 Triton Ascend 使用现场已有环境。

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
# OSCAR 模式会自动准备 rotation
bash scripts/serve.sh

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
| `oscar_ascend/prepare_rotations.py` | 缓存校验、缺失自动生成、配对发布 |
| `oscar_ascend/calibrate.py` / `calibration_worker.py` | 原生 TP4 两遍模型校准 |
| `oscar_ascend/calibration_kernels.py` | NPU 协方差、Jacobi 特征分解与矩阵组合 |
| `oscar_ascend/check.py` | 源码指纹、版本、模型、NPU 预检 |
| `oscar_ascend/source_interfaces.py` | 无导入副作用的源码接口检查 |
| `scripts/serve.sh` | 一键启动 |
| `scripts/test_npu.sh` / `scripts/bench.sh` | 真机测试 / 配对基准 |
| `tests/` | CPU oracle、隔离测试、真实 NPU kernel/graph 测试 |
| `AGENTS.md` | 后四卡的使用限制及后续维护规则 |

上游数值依据为 [OSCAR PR #46774](https://github.com/vllm-project/vllm/pull/46774)，
实现针对 D256 单组：K68 + V68 = 136 bytes，外层对齐到160 bytes。
