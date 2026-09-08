# 自动生成 rotation：v0.2.0

在已有目标 NPU 环境中，`bash scripts/serve.sh` 完成安装扩展、检查 `.pt`、
缺失时校准、校验及启动。无需手动提供 rotation 路径。
**所有运行入口固定 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`**，并在 NPU 库导入前生效。
校准使用 TP4，进程内逻辑 rank0 使用物理卡4；物理卡0..3不属于本项目。

## 查找与复用

默认文件为：

```text
artifacts/rotations/<model-key>-<profile-key>/k_rotation.pt
artifacts/rotations/<model-key>-<profile-key>/v_rotation.pt
```

model-key 由模型绝对路径、config、权重索引内容、权重文件大小与修改时间计算，
不重复读取数十GB权重。模型移动到另一路径也会保守地重新生成。
profile-key 包含生成器版本、校准文本内容和每段 token 上限。
`.pt` 内记录完整指纹、校准 run ID、方法、样本量及收敛情况。
只有本模型、同一次校准的有效 K/V 配对才自动复用，不把权重 `.pt` 当作 rotation。

有自动缓存但只剩一边、模型指纹改变、矩阵损坏或 run ID 不一致时，重新生成配对。
生成过程使用临时目录，两个文件都验证通过才发布。
并发启动同一模型/校准配置时使用文件锁串行准备，避免重复校准和半成品读取。
异常退出时保留原有文件，不以 identity/Hadamard-only 代替缺失的校准结果。

用户可显式提供已有 `VLLM_OSCAR_K_ROTATION_PATH` / `VLLM_OSCAR_V_ROTATION_PATH`。
这时兼容上游缺少指纹的 checkpoint，但仍检查层号、维度和正交性；
已有有效文件保持不变，缺少另一边时生成另一边。显式文件无效会拒绝启动，
不会覆盖可能属于用户的其他 `.pt`。

## 校准方法

一手数值参考为 OSCAR 的
[compute_kv_rotation.py](https://github.com/FutureMLS-Lab/OSCAR/blob/41ebcdba3db5f0ce1339c3727caea80df575d437/rotation/compute_kv_rotation.py)
中 `qqt_sst` 与 `r_h_pbr`，不运行其 CPU Q/K/V dump 路径。

1. 独立进程以 `quantization=ascend` 加载同一权重，KV 使用原生 BF16。
   校准阶段 eager、无 prefix caching、无 speculative decoding，使用同一 RoPE 配置。
2. 随包提供16段中英混合主题文本，默认每段1024 tokens，作为两遍相同输入。
   也支持本地 JSONL，每行是字符串或包含 `text`/`prompt` 的对象。
   文本只参与校准，不作为性能/质量测评结果。生产质量应在独立题集上验证。
3. 临时外部 hook 观察主模型16个 FULL 层的 post-RoPE Q/K/V；GDN/vision/draft不处理。
   v0.2.3 在模型初始化完成后的 begin RPC 中才安装 hook；插件发现和模型 profiling
   阶段都不安装 hook。挂钩只访问已完成初始化的原生 attention 模块。
   校准 worker 通过 `worker_extension_cls` 注册三个命名方法，`collective_rpc`
   只传方法名字符串及基础数据类型，兼容 vLLM 默认消息序列化，不跨进程传 Python 函数。
4. 第一遍在每个 TP rank 对自己的6个 Q heads 累计：

   `Cq = sum(QᵀQ) / (N * 6)`。

5. 第二遍重跑相同文本。对该 rank 的1个 KV head，计算：

   `w[t] = max(K[t] Cq K[t]ᵀ, 0)`；
   `Cv = sum(w[t] V[t]ᵀ V[t]) / sum(w[t])`。

   这与上游先把 w 归一化到和为 N、再除以 N 的 SST 公式相同。
   两遍方法保留全样本 Cq，避免按 chunk 各自归一化改变 SST 定义。
6. 对四个 rank 的 Cq/Cv 做 NPU all-reduce 后平均，得到每层单个 K/V 旋转目标。
7. Triton FP32 cyclic Jacobi 同时分解32个256×256对称矩阵。
   每 sweep 包含255轮互不重叠的旋转配对；列/行更新由不同 kernel 隔开，避免数据竞争。
   每两 sweep 检查一次 `max(abs(offdiag))/max(abs(diag)) < 1e-6`，默认最多12 sweep。
   不收敛直接报错，不切到 CPU EVD。
8. 特征向量按特征值升序排列；Triton butterfly 计算归一化 Hadamard，再应用
   特征值降序加 bit-reversal 的 Pbr，得到 `R = U H Pbr`。
9. 生成包含全局层号3、7、…、63的 K/V 文件，验证后发布。
   校准进程关闭 vLLM engine、释放模型及 workers 后，启动脚本才进入服务阶段。

Q/K/V、协方差累计、SST 权重、矩阵特征分解和组合都在 NPU 上完成。
CPU 只负责文本 tokenization、调度、少量收敛诊断和最终约8MiB矩阵文件的校验/序列化。
不把原始 Q/K/V 拷回 CPU 或保存完整 activation dump。
FP32 NPU 分解与上游 FP64 CPU 分解允许特征向量符号/退化子空间不同；
数值重构和正交性有测试，但模型质量仍需真机评估。

## 可选参数

默认直接运行，无需这些参数；如需改变校准输入：

| 环境变量 | 默认 | 用途 |
|---|---|---|
| `OSCAR_ROTATION_DIR` | 项目 `artifacts/rotations` | 持久化缓存根目录 |
| `OSCAR_CALIBRATION_DATA` | 内置16段文本 | 业务 JSONL |
| `OSCAR_CALIBRATION_TOKENS` | 1024 | 每段 token 上限，允许256..8192 |
| `OSCAR_CALIBRATION_MEMORY` | 0.75 | 校准进程 memory utilization |
| `OSCAR_CALIBRATION_SWEEPS` | 12 | Jacobi 上限，允许2..32 |
| `OSCAR_RECALIBRATE` | 0 | 设为1重算自动缓存，不覆盖显式指定文件 |

`ASCEND_RT_VISIBLE_DEVICES` 固定为 `4,5,6,7`，不提供覆盖参数。
从已克隆的目录更新并运行：

```bash
git pull --ff-only
bash scripts/serve.sh
```

真机测试：`bash scripts/test_npu.sh`，同时覆盖 serving 和 calibration kernels。
本地 CPU/mock 测试只证明准备流程和数学 oracle，不代表 NPU 编译或校准已经跑通。
