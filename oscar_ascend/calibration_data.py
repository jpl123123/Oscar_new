"""Bundled, deterministic bootstrap prompts; optional real workload JSONL overrides."""

import json
from pathlib import Path

PASSAGES = [
    "Explain how a relational database uses indexes, transactions, isolation and recovery. "
    "A warehouse stores orders, customers and stock movements. Readers need consistent snapshots "
    "while writers update quantities. Discuss duplicate keys, missing values, query plans and tests.",
    "请分析城市公共交通的运营数据。早高峰客流集中在住宅区至办公区，晚高峰方向相反。"
    "周末游客分布不同，雨天换乘需求增加。比较发车间隔、换乘等待、可靠性与乘客体验，"
    "解释如何设计对照实验并避免用相关性代替因果关系。",
    "Review a Python service with async workers, bounded queues and retries. A request can fail "
    "before or after its database transaction commits. Explain idempotency, cancellation, backoff, "
    "observability and shutdown. Include small examples and think through concurrency races.",
    "某工厂生产甲乙两种产品，每件产品使用不同数量的材料和工时。设库存和交付时间为约束，"
    "建立线性规划模型，并讨论需求预测误差、设备维护与计划调整。要求列出变量的含义、"
    "单位和边界条件，逐步检查计算结果是否满足原始约束。",
    "Read a scientific account of water transport in plants. Roots absorb water, vessels carry it "
    "through stems, and leaves exchange gases through stomata. Temperature, light and humidity "
    "affect these processes. Separate measurement, hypothesis, mechanism and uncertainty.",
    "请把一份较长的会议记录整理成行动计划。讨论涉及用户反馈、界面可访问性、产品测试、"
    "交付排期和跨团队依赖。有些行动已经确定负责人，有些仍需核实。保留明确的事实，"
    "标记尚未解决的问题，不要凭空补充日期、数字或参与者的意见。",
    "A numerical simulation solves a diffusion equation on a finite grid. Compare explicit and "
    "implicit time stepping, stability, truncation errors and boundary conditions. Explain what "
    "changes as the grid is refined and how conservation laws help detect implementation bugs.",
    "解释一段网络故障排查过程：客户端请求经过域名解析、路由、TLS握手和应用处理。"
    "不同时间段的错误率不一致，部分请求超时而服务仍返回成功日志。请按证据定位问题，"
    "区分连接失败、排队延迟和业务错误，提出可以验证假设的检查步骤。",
    "Design an algorithm to process a stream of events with timestamps that may arrive out of order. "
    "Discuss window boundaries, watermark assumptions, duplicate suppression and state cleanup. "
    "Use clear pseudocode and describe expected behavior at exact boundary values.",
    "请为初学者讲解概率与统计中的抽样误差。比较均值、中位数、方差和分位数，讨论异常值"
    "以及样本量对结果稳定性的影响。用掷骰子、排队时间和商品尺寸给出具体例子，说明"
    "为什么应报告观察条件，而不能只展示一个看起来很好的数字。",
    "Translate a technical product guide into plain language. The guide describes account setup, "
    "document editing, version history, exports and recovery. Preserve technical meaning, remove "
    "unnecessary jargon and keep warnings connected to the action they concern.",
    "一家图书馆希望改进借阅服务。历史记录包含借阅时间、归还时间、预约队列与书籍分类。"
    "记录中存在重复条目、编码不一致以及节假日闭馆的特殊情况。设计数据清理流程，"
    "说明哪些指标适合评估改进，并考虑读者隐私与长期维护成本。",
    "Compare merge sort, quicksort and heap sort for data stored in memory and on disk. Explain "
    "stability, worst-case behavior, space complexity and cache locality. Walk through ties, "
    "empty inputs and partially sorted inputs without assuming every benchmark is comparable.",
    "请描述从需求分析到软件发布的完整过程。一个版本需要迁移配置格式，同时兼容老用户"
    "的文件。讨论如何制定接口约定、提供迁移工具、编写验收案例和保存回滚信息。"
    "用清晰的语言说明成功条件以及发生错误时系统应当呈现的具体行为。",
    "Analyze an energy system with solar generation, storage and variable demand. Follow the "
    "units in every calculation. Discuss peak demand, seasonal variation, losses and uncertainty. "
    "Build a simple model, state assumptions and explain how observations could falsify them.",
    "请比较历史资料中的多种解释。不同作者可能引用同一事实，却提出不同原因。梳理时间"
    "顺序、证据来源与推论之间的关系，保留重要的分歧。输出结构清楚的说明，避免把"
    "作者的主张当成已经证明的结论，并给出下一步应查找的材料类型。",
]


def load_texts(path=None):
    if path is None:
        return PASSAGES, "builtin-bootstrap-v1"
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        value = item if isinstance(item, str) else item.get("text", item.get("prompt"))
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Calibration JSONL needs strings or objects containing text/prompt")
        records.append(value)
    if not records:
        raise ValueError("Calibration JSONL is empty")
    return records, "user-jsonl"


def token_prompts(tokenizer, texts, tokens=1024, builtin=False, minimum=32):
    # Real workload JSONL routinely contains short queries; skip them instead of
    # aborting a full calibration run that already loaded the TP4 model.
    prompts = []
    skipped = []
    for index, text in enumerate(texts):
        if builtin:
            # Deterministic long contexts, no downloads and no hidden benchmark dataset.
            text = "\n".join(f"Section {i + 1}: {text}" for i in range(48))
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            ids = tokenizer.encode(text, add_special_tokens=True)
        ids = list(ids)[:tokens]
        if len(ids) < minimum:
            skipped.append(index)
            continue
        prompts.append({"prompt_token_ids": ids})
    if skipped:
        print(
            f"[OSCAR calibration] Skipped {len(skipped)} short prompt(s) below "
            f"{minimum} tokens at record indexes {skipped}",
            flush=True,
        )
    if not prompts:
        raise ValueError(
            f"All {len(texts)} calibration prompts are below {minimum} tokens; "
            "fix OSCAR_CALIBRATION_DATA or unset it to use the builtin bootstrap texts"
        )
    return prompts
