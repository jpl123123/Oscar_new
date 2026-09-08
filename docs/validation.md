# 验证记录 — 2026-09-08

状态：**最近真机启动阻塞；NPU graph / 端到端 / 精度 / 性能验收未通过**。
后续候选变更尚无成功的真机记录；本地测试结果不覆盖这些验收结论。

## 实际运行

环境：macOS arm64，Python 3.12，独立 `.venv`，CPU Torch 2.14.0。
没有 `torch_npu`、Triton Ascend 或可连接的 NPU 测试机配置。

| 检查 | 结果 |
|---|---|
| `python -m pytest -q --junitxml=artifacts/local-tests.xml` | **174 passed, 2 skipped**；skip 为 serving 和 calibration 两个真实 NPU 测试模块 |
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

v0.2.2 新增：使用现场 `backend=npu, arch=Ascend910B4, warp_size=0` 的预检回归，
同时兼容 `ascend` 名称并拒绝 CUDA/HIP/CPU 后端；校准采用 worker extension 的
字符串 RPC，控制消息可由默认序列化编码；核对所有校准 LLM 参数都存在于同版
LLM/EngineArgs 接口。核对官方
[NPUDriver.get_current_target](https://github.com/Ascend/triton-ascend/blob/main/third_party/ascend/backend/driver.py#L156)
返回 `npu` 和 `warp_size=0`；核对本地上游 `vllm/v1/serial_utils.py` 默认拒绝函数对象。
未连接真机，未声称上述本地检查等同于端到端真机成功。

v0.2.3 新增：在阻止所有 vLLM/Ascend 导入的条件下，读取真实参考源码完成整个接口预检，
该测试没有 mock 接口检查函数；不兼容构造参数仍被拒绝。用相同依赖边的最小 Python
包复现 DeviceOperator 循环导入，验证正常包初始化顺序可完成导入；服务注册延迟读取
Attention 类、跳过尚未初始化完的模块；校准注册不安装 hook，begin 阶段才安装。
这些测试覆盖导入行为，并非完整原生 NPU 模块或模型执行。

v0.2.4 新增：所有入口及子进程在导入 Torch/vLLM 前强制 spawn，覆盖继承 fork 的
情况；读取同版源码验证 EngineCore 与 TP executor 均通过 get_mp_context 选择上下文。
实际运行 CPU 两层进程测试：父进程、模拟 EngineCore 均先运行 Torch/autograd 并建立
后台线程，再创建子进程；两层均以 spawn 启动，父进程标记没有被继承，子进程 Torch
梯度计算结果正常。该测试证明进程隔离行为，不等于在目标 TorchNPU/CANN 上复现并消除断言。

v0.2.5 基于 `207d2d4`：用户反馈真机已成功生成 `.pt`，正式服务加载 rotation 时出现
`Float did not match BFloat16`。参考源码 `model_loader/base_loader.py` 的
`process_weights_after_loading` 在 `set_default_torch_dtype(model_config.dtype)` 内执行，
导致没有显式 dtype 的 `torch.eye` 继承 BF16，与加载的 FP32 rotation 冲突。
本地先用同一 BF16 默认上下文复现相同异常，再显式指定单位矩阵的 dtype 和设备。
回归覆盖 FP32/BF16 默认 dtype、CPU/meta 默认设备、无效矩阵拒绝以及已有 K/V 文件
字节不变且不再次生成。移除 `serve.sh` 的两次环境预检调用，脚本控制流程测试确认
安装、准备矩阵、启动服务的顺序，仍覆盖继承错误卡号与 fork 的情况。
这些结果验证此次 dtype 修复与控制流程；没有登录 node93 验证修复后的完整服务启动。

v0.2.6 基于 `aacaf2a`，处理用户反馈的首组 FULL decode capture 停留超过5分钟：

- 读取真实参考 `_dummy_run`，确认先构建 metadata 再将原生 slot mapping 设为 -1。
  CPU 回归运行真实 OSCAR builder（仅替换 native imports 和设备标签），复现私有
  快照仍保留0而原生源已为-1；外部 dummy scope 修复后 counts=[0,0]、slots=-1，
  捕获不访问 KV。随后真实 build 将有效元数据写回相同地址。
- `_clip_vec` 移除 `tl.static_range`，保持原有选择与插值顺序。
  使用 Torch 对应 TL 操作执行实际函数体，对随机值、全零、并列值以及
  0/0.875/0.92/0.96/1.0 百分位与 `torch.quantile` 对照通过。
  [Triton 文档](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html)
  明确说明 static_range 会引导积极展开；本地未测量 Ascend 编译耗时。
- rotation 的 token 数移出 constexpr，按运行时参数传入并禁止值特化。
  CPU AST 回归检查这两个条件，不冒充真实 Triton 编译结果。
- hook 仅在原生 runner 导入完成后安装，重复安装幂等；异常恢复 dummy scope，
  捕获结束/异常取消超时栈定时器。每个 worker 输出启动阶段与首次 kernel 提交耗时，
  捕获未完成时每120秒输出 Python 栈，覆盖 native synchronize 的等待位置。
- 扩展真实 NPU 测试，覆盖 counts=0 捕获不改 KV、更新 counts 后 replay 处理真实请求。
  该 NPU 测试在本地明确跳过。保留后四卡、5656、MTP3、FULL_DECODE_ONLY 和无预检启动。

这些改动修正已发现的 dummy 访问与编译结构问题；现有等待告警无法证明唯一现场根因，
也没有在本地复现 node93 的 NPU 挂起。修复后的完整图捕获、推理与性能仍待真机验证。

v0.2.7 基于 `6533773`：用户补充的日志显示各 OSCAR kernel 首次提交均返回，
四个 worker 的512-token预热约57～59秒返回；后续超时栈都停在
`gdn_attn.py:530` 的 `(num_accepted_tokens - 1).cpu()`。
这是捕获 metadata 构建期间对前序设备队列的同步等待，不能从这个栈单独断言
GDN 自身出错或某个 OSCAR kernel 已完成。保留原生 GDN 路径与编译配置。

- attention 去掉 runtime `if live`，空任务/空 split 至少运行一个全掩码 tile，
  保持 QK/PV dot 阶段可到达，屏蔽实际内存访问。这是对零计数混合核执行路径的
  兼容性修正，尚未在目标编译器上证明它是本次设备挂起的唯一原因。
- CPU 测试直接执行实际 attention 函数体，使用带越界检查的地址模型与 Torch
  对应 TL 操作，覆盖 poisoned dummy metadata、额外空任务、空 history、
  当前 token 因果 attention。检查空任务两次 dot、无 Q/K/V/页表/输出访问，
  有效空 history 为0/-inf，正常 raw 输出与独立 dense softmax 对照通过。
  这不模拟 NPU 的核间同步，也不替代真实编译与执行测试。
- 首次图外预热按 head/stride/旋转布局或 split 配置验证 device completion；
  回归确认提交前后同步的顺序、捕获内禁止同步、设备失败不记作完成。
- 超时栈保存到每个 pid 独立文件，回归验证异常时取消定时器并关闭文件。
  日志目录无法写入时回退 stderr，不因此阻止服务启动。

本地总计129项通过，两个 NPU 模块跳过。没有修改 `.pt` 或校准指纹；
未声称本地测试已经验证 node93 挂起消失。

v0.2.8 消除 GDN capture metadata 的冗余 D2H，并纠正验收表述：

- 外部 hook 使用已有 CPU query boundaries 计算 host draft 分类，接受 token 数
  仍通过 NPU query boundaries 计算，再交给原生 GDN build。
  原生 GDN/conv/SSM 算子和普通 async decode metadata 路径没有替换。
- CPU 测试禁止设备对象调用 `.cpu()`/`.item()`/`.tolist()`/`.numpy()`，
  执行实际参考 capture 函数可复现 D2H，执行适配函数不触发回读。
  四种查询边界（含零长度 padding）的参数与原公式一致；缺失/不匹配 CPU
  副本不回退到 D2H。该测试不是实际 NPU 性能测试。
- hook 延后至原生 GDN module 完成初始化才安装，保持原生 build 和其他平台的
  自有 capture override。日志仅报告 capture_model 返回，不据此声称 graph 验收成功。
- 用户最新 e148925 现场日志已包含各类首次 kernel 的 `device complete`。
  该证据证明这些首次提交的设备工作完成，日志仍停在512-token capture 开始；
  不能外推为全部层、所有capture档位、真实请求replay或精度已通过。

当前状态仍为真机启动验收未通过。本地137项通过，两个 NPU 模块跳过。
详细 CPU 边界、尚未优化的校准回读与诊断同步见 [CPU/同步审计](cpu-audit.md)。

v0.3.0 基于 `aeadd39`，优化调度/数据搬运并使能直接运行时 FULL graph：

- 紧凑任务表在 NPU 一次生成，history/raw 使用固定 task buffer 与 task_count。
  program 以固定步长循环任务；CPU 执行真实 Triton 函数体，覆盖空请求、零长度
  padding、129请求容量、截断 active tokens、QT=1/2/4 和跨 program 的任务唯一归属。
  多请求 ragged raw attention 与独立因果 dense reference 一致。
- 默认512-token/8-split场景 launch program 从2048降到32，单独测试该数量变化。
  它不能被解释为64倍吞吐提升。INT2 解包实际源函数体与 packed bytes 对照逐值相等，
  读取64字节数据与4字节元信息，不构造256个重复 packed-byte load 地址。
- 页表复制融合进 NPU 任务准备 kernel，按 native host 长度上界限制列范围；
  回归检查128/129 token边界、尾部列不被无谓覆盖、固定地址与真实 build 更新。
  dummy 仅清 active counts，不重复清完整最大长度的页表。
- 校准残差汇总源函数体与旧 host reduction 对照相等，NaN/Inf 仍拒绝；
  host 每次收敛判断仅回读一个 scalar。现有 rotation/cache 指纹不变。
- 启动配置为 mode=0、FULL_DECODE_ONLY、enable_npugraph_ex=false，关闭FX/AOT，
  保留原生运行时 ACLGraph。两个 native 对照模式保留原配置。
  测试执行真实参考 `_use_aclgraph` predicate，复现mode=0下False，验证外部hook
  使直接FULL模式返回True且显式eager仍返回False；核对原生FULL wrapper、capture
  入口仅依据cudagraph_mode，CANN jit_compile=False设置只执行一次。
- NPU测试代码适配任务表并保留真实graph replay、MTP回滚、非顺序页表等场景；
  这些NPU测试在本地仍跳过。170项CPU回归、ruff、shell语法和diff检查通过。

没有可报告的真机延迟、吞吐或完整graph成功记录。关闭FX可能失去某些fusion收益，
新program budget的最佳值也需要实际profiler。当前不能宣称所有慢操作已消除。

v0.3.1 基于 `620f698`，撤销造成 UB 对齐膨胀的三维解包优化：

- 用户提供的 compiler IR 中，逻辑 `32×64×4` BF16 被放进 `32×64×16` UB allocation，
  单个临时缓冲为64 KiB，另有同样大小的转置缓冲。附件从IR中段开始，缺少原始
  `error:`/CompilationError诊断，因此不把该片段当作完整的UB overflow诊断。
- `_load_vec` 恢复二维地址和位运算；AST与 `e148925` 中同名函数完全一致。
  该历史版本有用户提供的首次设备完成记录，但这不证明当前整个persistent kernel
  已在真机编译通过。紧凑任务、融合页表复制、直接FULL graph配置不变。
- 实际解包函数体的CPU回归覆盖4/16/32行、无效行越界地址屏蔽、FP16元信息与BF16
  数值；增加静态约束防止再引入窄尾轴三维展开。173项通过，两个NPU模块跳过。

本次恢复重复字节地址，明确撤回从源代码load宽度推断其一定更快的优化判断。
不改动 `.pt`、量化公式、位序或其他用户的设备。仍缺当前版本完整真机编译和graph验收。

v0.3.2 基于 `b5cd96c`，根据更精确的设备完成日志撤销持久化任务循环：

- 四个 worker 的 rotation/store 均报告 device complete，history `(splits=8,groups=4)`
  在约12秒编译/提交返回后停在设备完成等待。已进入 kernel 执行等待，不再笼统
  描述成正常编译慢；没有该 kernel 完成或后续graph成功的证据。
- 移除 v0.3.0 加入的外层多任务循环与 TASK_GROUPS 参数。一个 program 处理一个
  task_id，仍通过 NPU 紧凑表取 request/query offset；没有跳过history或替换输出。
  从head/query加载开始的attention数学主体，与有过现场设备完成记录的
  `e148925` 对应主体 AST 一致，二维解包也保持该历史实现。
- 移除 program_budget 配置；旧 OSCAR_ASCEND_PROGRAM_BUDGET 不再改变执行。
  512-token/8-split grid恢复2048个program，不再用32个program作为性能承诺。
  任务表避免重复请求扫描、页表融合复制和直接FULL graph等独立改动仍保留。
- 回归覆盖顺序/逆序/交错program执行下的ragged因果attention、任务唯一归属、
  无效任务屏蔽，并约束attention内只保留KV循环。174项CPU测试通过，两个NPU模块跳过。

该回退针对新增持久化执行路径；未在本地证明它就是硬件等待的唯一原因。
整体NPU启动、graph replay、质量及性能仍未验收通过。

## 尚未运行的必要检查

- 实际 Triton Ascend 编译与执行；独立 NPU tests 包含旋转 stride、INT2 bytes、
  non-monotonic 页表、sink/recent 去重、long continuation、MTP 拒绝回滚、
  不同请求数/页表的 NPUGraph replay，以及 metadata 固定地址测试。
- 指定 Qwen3.5-27B-w8a8-mtp 模型的四卡启动、混合 GDN/FULL、MTP、async 和图模式。
- Qwen3.5-27B 专属离线校准矩阵的现场校验及端到端质量测试。
- 自动生成路径的独立 NPU 测试：用户已反馈真机生成 `.pt` 成功，
  本地没有执行 Triton 协方差、FP32 Jacobi、Hadamard/Pbr 或双遍真实模型运行。
- 同请求集、相同输出长度、相同硬件的三轮配对 benchmark 和 NPU profiler。

没有生成或填充任何虚构 NPU 性能数据。“不得更慢”尚未证明。
prefix caching 关闭、原生 page 容量不缩小、额外 current-token scratch 的开销，
均需在现场验收中计入；不能用理论 slot 压缩率代替实际吞吐或显存收益。
