"""Deployment preflight. Run in the target Ascend environment, not on the Mac."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import sys

from .config import OscarConfig
from .layout import CacheLayout


def verify_sources(roots):
    manifest = json.loads(
        pathlib.Path(__file__).with_name("upstream_fingerprints.json").read_text()
    )
    problems = []
    for package, info in manifest.items():
        root = pathlib.Path(roots[package])
        for relative, expected in info["files"].items():
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                problems.append(f"{package}: source mismatch: {path}")
    if problems:
        raise ValueError("\n".join(problems))
    return {package: info["commit"] for package, info in manifest.items()}


def check_model(path):
    model = json.loads((pathlib.Path(path) / "config.json").read_text())
    cfg = model.get("text_config", model)
    expected = {
        "head_dim": 256,
        "num_key_value_heads": 4,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "model_type": "qwen3_5_text",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Expected model {key}={value}, found {cfg.get(key)}")
    layer_types = cfg.get("layer_types")
    if layer_types is None:
        interval = cfg.get("full_attention_interval", 4)
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(64)
        ]
    full = [i for i, value in enumerate(layer_types) if value == "full_attention"]
    if full != list(range(3, 64, 4)):
        raise ValueError(f"Unexpected FULL layer pattern: {full}")
    return full


def main():
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument(
        "--sources-only",
        type=pathlib.Path,
        help="Read-only local source check: directory containing vllm and vllm-ascend",
    )
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument(
        "--skip-rotations",
        action="store_true",
        help="Check the base NPU runtime before automatic rotation generation",
    )
    args = parser.parse_args()
    report = {"status": "not_run", "npu_acceptance": "not_run"}
    try:
        if args.sources_only:
            roots = {
                "vllm": args.sources_only / "vllm",
                "vllm-ascend": args.sources_only / "vllm-ascend",
            }
            report["source_commits"] = verify_sources(roots)
        else:
            if not args.model:
                raise ValueError("--model is required for deployment preflight")
            import torch
            import torch_npu
            import triton

            report["versions"] = {
                name: importlib.metadata.version(name) for name in ("vllm", "vllm-ascend")
            }
            report["versions"].update(
                torch=torch.__version__, torch_npu=torch_npu.__version__, triton=triton.__version__
            )
            report["physical_npu_devices"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
            for package in ("vllm", "vllm-ascend"):
                if report["versions"][package].split("+")[0] != "0.23.0":
                    raise ValueError(
                        f"Unsupported {package} version: {report['versions'][package]}"
                    )
            roots = {
                name: pathlib.Path(
                    importlib.util.find_spec(name.replace("-", "_")).origin
                ).parent.parent
                for name in ("vllm", "vllm-ascend")
            }
            report["source_commits"] = verify_sources(roots)
            if not torch.npu.is_available() or torch.npu.device_count() != 4:
                raise ValueError(
                    "Expected exactly 4 visible NPUs after restricting physical devices to 4,5,6,7"
                )
            target = triton.runtime.driver.active.get_current_target()
            report["triton_target"] = str(target)
            if target.backend != "ascend":
                raise ValueError(f"Expected Triton Ascend, got {target.backend}")
            full_layers = check_model(args.model)
            cfg = OscarConfig.from_env()
            from .rotations import get_rotation

            if not args.skip_rotations:
                for layer in full_layers:
                    for path in (cfg.k_rotation_path, cfg.v_rotation_path):
                        get_rotation(path, f"model.layers.{layer}.self_attn.attn", "cpu")
            report["rotation_validation"] = (
                "pending_generation" if args.skip_rotations else "passed"
            )
            report["full_layers"] = full_layers
            report["example_geometry"] = {
                "physical_block_tokens": 768,
                "stripe_page_bytes": CacheLayout(768, config=cfg).stripe_page_bytes,
                "actual_geometry_checked_by_metadata_builder": True,
            }
            report["prefix_caching"] = "disabled by launch script"
            report["performance_acceptance"] = "not_run"
        report["status"] = "preflight_passed"
    except (Exception, SystemExit) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    return 0 if report["status"] == "preflight_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
