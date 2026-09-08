"""Read interface declarations without executing vLLM/Ascend package initializers.

    device_op -> ops.__init__ -> fused_moe -> device_op

is order-sensitive in the target source tree. A preflight must not activate this
graph just to inspect signatures. Real tensor geometry remains checked at bind.
"""

import ast
from pathlib import Path


def validate_source_interfaces(roots):
    trees = {}

    def cls(package, relative, name):
        path = Path(roots[package]) / relative
        if path not in trees:
            try:
                trees[path] = ast.parse(path.read_text(), filename=str(path))
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"Cannot inspect interface source {path}: {exc}") from exc
        for node in trees[path].body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise ValueError(f"Required class {name} not found in {path}")

    def fields(node):
        return {
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        }

    checks = {"inspection_mode": "source_ast_no_imports"}
    methods = (
        (
            "vllm-ascend",
            "vllm_ascend/platform.py",
            "NPUPlatform",
            "get_attn_backend_cls",
            ("selected_backend", "attn_selector_config", "num_heads"),
        ),
        (
            "vllm",
            "vllm/model_executor/layers/attention/attention.py",
            "Attention",
            "__init__",
            ("prefix", "attn_backend"),
        ),
        (
            "vllm",
            "vllm/model_executor/layers/attention/attention.py",
            "Attention",
            "process_weights_after_loading",
            ("act_dtype",),
        ),
        ("vllm", "vllm/entrypoints/llm.py", "LLM", "collective_rpc", ("method", "args", "kwargs")),
        (
            "vllm",
            "vllm/v1/attention/backend.py",
            "AttentionMetadataBuilder",
            "build_for_cudagraph_capture",
            ("common_attn_metadata",),
        ),
        (
            "vllm-ascend",
            "vllm_ascend/attention/attention_v1.py",
            "AscendAttentionBackendImpl",
            "forward",
            ("layer", "query", "key", "value", "kv_cache", "attn_metadata", "output"),
        ),
        (
            "vllm-ascend",
            "vllm_ascend/attention/attention_v1.py",
            "AscendAttentionBackend",
            "get_kv_cache_shape",
            ("num_blocks", "block_size", "num_kv_heads", "head_size"),
        ),
        (
            "vllm-ascend",
            "vllm_ascend/attention/attention_v1.py",
            "AscendAttentionBackend",
            "get_supported_kernel_block_sizes",
            (),
        ),
    )
    for package, relative, class_name, method_name, required in methods:
        owner = cls(package, relative, class_name)
        method = next(
            (
                node
                for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ),
            None,
        )
        label = f"{class_name}.{method_name}"
        if method is None:
            raise ValueError(f"Required method {label} not found in {relative}")
        params = {
            arg.arg for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs
        }
        if missing := set(required) - params:
            raise ValueError(
                f"Incompatible {label}: missing parameters {', '.join(sorted(missing))}"
            )
        checks[label] = {"parameters": sorted(params), "source_line": method.lineno}

    parallel = cls("vllm", "vllm/config/parallel.py", "ParallelConfig")
    if "worker_extension_cls" not in fields(parallel):
        raise ValueError("ParallelConfig.worker_extension_cls is required for calibration RPC")
    checks["ParallelConfig.worker_extension_cls"] = "declared"
    metadata_fields = fields(cls("vllm", "vllm/v1/attention/backend.py", "CommonAttentionMetadata"))
    metadata_fields |= fields(
        cls("vllm-ascend", "vllm_ascend/attention/utils.py", "AscendCommonAttentionMetadata")
    )
    required_fields = {
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "block_table_tensor",
        "slot_mapping",
        "num_reqs",
        "num_actual_tokens",
        "max_query_len",
        "attn_state",
        "causal",
    }
    if missing := required_fields - metadata_fields:
        raise ValueError(
            f"AscendCommonAttentionMetadata missing fields: {', '.join(sorted(missing))}"
        )
    checks["AscendCommonAttentionMetadata"] = "required fields declared"
    checks["native_cache_geometry"] = "checked_when_native_tensors_are_bound"
    return checks
