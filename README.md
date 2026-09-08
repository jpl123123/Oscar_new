# OSCAR INT2 for Qwen3.5-27B / vLLM Ascend

外部适配实现，开发参考为 **vLLM Ascend 0.23.0 + PR #12607（19e436985）**，
配套 vLLM 0.23.0。运行时允许 Ascend **0.23.0 / 0.23.1**，包括开发版和本地
构建后缀，并检查适配所需接口；例如现场的
`vllm-ascend 0.23.1.dev0+g5cb98caaa.d20260822` 与 `vllm 0.23.0+empty`。
方案、公式、流程图和实施点见
[设计文档](docs/design.md)。参考源码没有修改，扩展不导入 `references/`。

实现包括 Triton Ascend rotation、clip/INT2 store、分页 split-KV attention、
BF16 sink/recent/current、LSE merge 和 vLLM general plugin。
主模型 FULL 层使用 OSCAR，GDN 和 MTP draft 算子保持原生；MTP target verification
的多个候选 query 在同一 Triton program 内复用 KV tile。

**最近真机结果：FULL decode 启动阻塞，graph 验收未通过。** 后续候选变更尚无
成功的真机启动、真实请求 graph replay、精度或性能记录。本地 CPU 测试只能证明
其覆盖的控制流程与数学性质，不能将本项目称为已跑通、已优化完成或不慢于原生。
用户已反馈 `.pt` 生成成功；这不等于服务启动或模型质量验收成功。
CPU 回读与同步的逐项记录见 [CPU/同步审计](docs/cpu-audit.md)。

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
所有入口同时固定 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，使校准和服务的
EngineCore/TP worker 启动新解释器，避免多线程父进程 fork 后继承失效线程池。
校准使用 Python LLM API，不能依赖正式 `vllm serve` CLI 才设置的 spawn 默认值。

启动脚本依次执行：

1. 安装当前外部扩展。
2. 按模型指纹和校准数据配置查找 K/V `.pt`，校验模型归属、层号、D256 和正交性。
3. 文件不存在或自动缓存失效时，加载同一 W8A8 模型，使用原生 BF16 KV 做两遍校准，
   在 NPU 上计算 QQT/SST、特征分解和 `U H Pbr`，生成 K/V `.pt`。
4. 验证生成结果、保存缓存，释放校准模型与进程的显存。
5. 用生成或复用的 `.pt` 启动 OSCAR 服务。

从 v0.2.5 起，启动脚本不再调用 `oscar_ascend.check`，直接准备矩阵并启动服务。
同时修复正式加载模型时默认 BF16 导致的 `Float did not match BFloat16`：
rotation 正交性比较显式使用矩阵的 FP32 dtype 与 CPU 设备，再转为 NPU BF16。
已有有效 `.pt` 可继续复用，更新本版本不改变缓存指纹，无需删除或重做校准。

v0.2.6 修正图启动路径：dummy 预热/捕获通过设备端计数屏蔽 KV 访问，
避免私有 slot 快照错过原生后续的 `-1` 屏蔽。裁剪选择循环不再强制展开，
rotation 的 token 数改为运行时参数，避免每种 capture size 重编同一旋转算法。
保留 FULL_DECODE_ONLY；启动打印每组预热/捕获与首次 kernel launch 阶段。

v0.2.7 进一步处理设备执行等待：attention 的空任务/空 split 执行一个全掩码
tile，避免整个 Cube/Vector 分支或循环被跳过，掩码继续阻止无效数据读写。
图外预热时每个首次出现的 kernel 配置等待设备完成，日志会分别显示
`launch returned`（提交返回）和 `device complete`（设备完成）；正式推理和图内
不增加同步。捕获期间每120秒将各 worker 的 Python 栈分别写入
`artifacts/startup/worker-<pid>-stacks.log`，完成后取消，避免四卡输出交错。
`OSCAR_STARTUP_TRACE=0` 可关闭这些诊断，`OSCAR_STARTUP_LOG_DIR` 可指定日志目录。
这些更新继续复用已有 `.pt`，不改变量化参数或矩阵文件。

v0.2.8 消除 GDN 捕获 metadata 的一次冗余 NPU→CPU 回读：接受 token 数继续在
设备上计算，CPU 侧的 draft 分类直接使用 runner 已有的 CPU 查询边界副本。
这是外部 metadata hook，未改动 GDN/conv/SSM 算子；它消除该同步点，不代表
前序设备工作阻塞已经解决。启动日志仅报告 `capture_model returned`，不宣称
真实请求 graph replay 或精度验收通过。

v0.3.0 的执行与启动优化：

- 在 NPU 上按批次建立紧凑任务表，供所有 history/raw split 使用；attention program
  循环处理任务，避免每个 program 重新扫描全部请求。默认 program budget 为32，
  512-token、8-split history 的 launch grid 从2048个 program 降为32个。
- INT2 每向量读取64个 packed bytes，然后在寄存器解出256个值，避免重复字节地址。
- 页表复制融合进任务准备 kernel，仅复制原生 host 长度上界可达的列，保留固定
  buffer 地址/stride；dummy 不再反复清零最大上下文的整张页表。
- Jacobi 残差在 NPU 汇总，host 每轮判断仅回读一个 scalar，已有 `.pt` 不重算。
- OSCAR 启动配置使用 `mode=0`、`enable_npugraph_ex=false`，跳过现场重复的 FX/AOT
  编译，运行时保持 `FULL_DECODE_ONLY`。外部 hook 同时修正原生 `_use_aclgraph`
  对 FX 的依赖，使图参数初始化保持启用；显式 eager 仍关闭 graph。

这些是已实现并完成本地回归的候选优化。program 数、字节地址和复制范围的减少
不等于已经测得端到端加速；直接 FULL graph 的真机捕获/请求 replay 仍需现场验证。
`OSCAR_ASCEND_PROGRAM_BUDGET` 可选择8/16/32/64，默认32；实际最佳值需 profiler 测量。

默认缓存位于 `artifacts/rotations/<模型指纹-校准配置指纹>/`。
首次运行需要额外校准时间，后续启动会直接复用有效缓存。
默认校准使用随包提供的16段中英混合文本，每段最多1024 tokens，无需下载数据集。
这是一套启动用的校准数据，**不等于模型质量验收通过**；也可通过
`OSCAR_CALIBRATION_DATA=/path/to/prompts.jsonl` 使用实际业务文本。
业务 JSONL 中不足32 token 的短记录会被跳过并打印告警；数据文件缺失、为空、
格式不可用或全部过短时，自动回退到内置文本并打印告警。直接
`bash scripts/serve.sh` 即可完成校准和启动，无需手动设置环境变量；
`data/calibration_bootstrap.jsonl` 只是同一套文本的文件形式，供需要显式
指定文件路径时使用。
完整方法、配置和失败处理见 [自动校准说明](docs/calibration.md)。

已有外部矩阵时，仍可选择设置 `VLLM_OSCAR_K_ROTATION_PATH` 和
`VLLM_OSCAR_V_ROTATION_PATH`。显式指定的已有文件如果无效会报错并保留原文件；
自动缓存损坏则重新校准。生成失败或矩阵校验失败时，脚本停止，不启动服务。

脚本仅安装当前外部包，不安装/替换 vLLM、torch_npu 或 Triton。
`PYTHON_BIN` 可指向已有虚拟环境 Python；`MODEL`、`PORT` 可覆盖默认值。
默认保留用户的 TP4、262144 context、MTP3、async 和 FULL_DECODE_ONLY 参数。
`MODE=oscar` 使用直接运行时图路径；两个 native 对照模式保留原来的 FX 配置。
只预览命令：`DRY_RUN=1 bash scripts/serve.sh`。

默认模型目录为 `/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp`，
服务监听 `0.0.0.0:5656`，模型服务名为 `qwen3.5`。如现场 Python 环境不是
默认的 `python3`，在启动命令前加入 `PYTHON_BIN=/absolute/path/to/python`；
模型目录不同时加入 `MODEL=/absolute/path/to/model`。

GitHub 仓库提供外部适配代码，原生参考树用
`oscar_ascend/upstream_fingerprints.json` 记录开发参考指纹，不作为子模块分发。
单独运行可选诊断工具 `oscar_ascend.check` 时，源码差异记录在报告的 `source_audit` 中，
不因构建标签或整文件哈希不同直接拒绝。
v0.2.3 的接口预检只用 AST 读取已安装源码中的方法参数与元数据字段，
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
| `oscar_ascend/runtime_env.py` | 导入前固定后四卡与 spawn 进程策略 |
| `oscar_ascend/startup.py` | 原生 dummy 作用域、启动阶段日志和超时栈 |
| `oscar_ascend/capture_metadata.py` | 复用 CPU 调度副本，避免 GDN 捕获时冗余 D2H |
| `scripts/serve.sh` | 一键启动 |
| `scripts/test_npu.sh` / `scripts/bench.sh` | 真机测试 / 配对基准 |
| `tests/` | CPU oracle、隔离测试、真实 NPU kernel/graph 测试 |
| `AGENTS.md` | 后四卡的使用限制及后续维护规则 |

上游数值依据为 [OSCAR PR #46774](https://github.com/vllm-project/vllm/pull/46774)，
实现针对 D256 单组：K68 + V68 = 136 bytes，外层对齐到160 bytes。
