"""Reuse validated model-specific rotations or generate them before serving."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from .check import check_model


def model_fingerprint(model):
    model = Path(model).resolve()
    digest = hashlib.sha256(b"oscar-qwen35-model-v1\0")
    digest.update(str(model).encode())
    digest.update((model / "config.json").read_bytes())
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (model / name).is_file():
            digest.update((model / name).read_bytes())
    # Do not reread tens of GB of weights just to start the server.
    for path in sorted(set(model.glob("*.safetensors")) | set(model.glob("pytorch_model*.bin"))):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def profile_fingerprint():
    from .calibration_data import PASSAGES

    digest = hashlib.sha256(b"oscar-qqt-sst-r-h-pbr-v1\0")
    digest.update(os.getenv("OSCAR_CALIBRATION_TOKENS", "1024").encode())
    data_path = os.getenv("OSCAR_CALIBRATION_DATA")
    if data_path:
        with Path(data_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    else:
        digest.update(json.dumps(PASSAGES, ensure_ascii=False).encode())
    return digest.hexdigest()


def validate_file(path, layers, kind, fingerprint, profile=None, strict=True):
    import torch

    from .rotations import get_rotation, load_checkpoint

    if not Path(path).is_file():
        return False, "missing", None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        info = obj if isinstance(obj, dict) else {}
        if strict:
            if info.get("model_fingerprint") != fingerprint:
                return False, "different/unknown model fingerprint", None
            if info.get("calibration", {}).get("profile_fingerprint") != profile:
                return False, "different/unknown calibration profile", None
            expected = "qqt_r_h_pbr" if kind == "k" else "sst_r_h_pbr"
            if info.get("objective") != expected:
                return False, "unexpected calibration objective", None
        elif info.get("model_fingerprint", fingerprint) != fingerprint:
            return False, "different model fingerprint", None
        load_checkpoint.cache_clear()
        for index in layers:
            get_rotation(str(path), f"model.layers.{index}.attn", "cpu")
        return True, "valid", info.get("calibration", {}).get("run_id")
    except (Exception, KeyError) as exc:
        return False, str(exc), None


def run_generator(model, output_dir, fingerprint, profile):
    env = dict(
        os.environ,
        OSCAR_ASCEND_ENABLED="0",
        OSCAR_ASCEND_CALIBRATING="1",
        ASCEND_RT_VISIBLE_DEVICES="4,5,6,7",
    )
    if env.get("VLLM_PLUGINS") and "oscar_ascend" not in env["VLLM_PLUGINS"].split(","):
        env["VLLM_PLUGINS"] += ",oscar_ascend"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "oscar_ascend.calibrate",
            "--model",
            str(model),
            "--output-dir",
            str(output_dir),
            "--model-fingerprint",
            fingerprint,
            "--profile-fingerprint",
            profile,
        ],
        env=env,
        check=True,
    )


def atomic_copy(source, destination, overwrite=True):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        shutil.copyfile(source, temporary)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # A file supplied via an explicit path might have appeared while
            # another process calibrated. Publish without replacing that file.
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(model, cache_root, k_path=None, v_path=None, generator=run_generator, force=False):
    layers = check_model(model)
    fingerprint, profile = model_fingerprint(model), profile_fingerprint()
    cache = Path(cache_root).resolve() / f"{fingerprint[:16]}-{profile[:12]}"
    cache.mkdir(parents=True, exist_ok=True)
    explicit = {"k": k_path is not None, "v": v_path is not None}
    paths = {
        "k": Path(k_path).resolve() if k_path else cache / "k_rotation.pt",
        "v": Path(v_path).resolve() if v_path else cache / "v_rotation.pt",
    }
    if paths["k"] == paths["v"]:
        raise ValueError("K and V rotation paths must be different files")
    with (cache / ".prepare.lock").open("a") as lock:
        print(f"[OSCAR] Checking rotations in {cache}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        valid, runs = {}, {}
        for kind, path in paths.items():
            valid[kind], reason, runs[kind] = validate_file(
                path,
                layers,
                kind,
                fingerprint,
                profile,
                strict=not explicit[kind],
            )
            if explicit[kind] and path.exists() and not valid[kind]:
                raise ValueError(
                    f"Explicit {kind.upper()} rotation is invalid; kept unchanged: {path}: {reason}"
                )
            print(f"[OSCAR] {kind.upper()}: {reason} ({path})", flush=True)
        coherent = any(explicit.values()) or (runs["k"] is not None and runs["k"] == runs["v"])
        if not force and all(valid.values()) and coherent:
            return {
                "k": str(paths["k"]),
                "v": str(paths["v"]),
                "status": "reused",
                "model_fingerprint": fingerprint,
            }
        if force and any(explicit.values()):
            raise ValueError(
                "Automatic recalibration does not overwrite explicitly supplied rotation files"
            )
        print(
            "[OSCAR] A usable K/V pair is missing. Starting native TP4 model calibration.",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="calibration-", dir=cache) as work:
            generator(model, Path(work), fingerprint, profile)
            generated_runs = []
            for kind in ("k", "v"):
                ok, reason, run = validate_file(
                    Path(work) / f"{kind}_rotation.pt",
                    layers,
                    kind,
                    fingerprint,
                    profile,
                    strict=True,
                )
                if not ok:
                    raise RuntimeError(f"Generated {kind.upper()} rotation is invalid: {reason}")
                generated_runs.append(run)
            if generated_runs[0] is None or generated_runs[0] != generated_runs[1]:
                raise RuntimeError(
                    "Generated K/V artifacts do not belong to the same calibration run"
                )
            # Validate BOTH staged files before publishing either. Default cache
            # files are replaceable; an existing explicit valid file is preserved.
            for kind in ("k", "v"):
                if not explicit[kind] or not valid[kind]:
                    atomic_copy(
                        Path(work) / f"{kind}_rotation.pt",
                        paths[kind],
                        overwrite=not explicit[kind],
                    )
        result = {
            "k": str(paths["k"]),
            "v": str(paths["v"]),
            "status": "generated",
            "model_fingerprint": fingerprint,
            "profile_fingerprint": profile,
        }
        manifest = cache / "pair.json"
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, manifest)
        return result


def main():
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.model,
        args.cache_root,
        os.getenv("VLLM_OSCAR_K_ROTATION_PATH") or None,
        os.getenv("VLLM_OSCAR_V_ROTATION_PATH") or None,
        force=os.getenv("OSCAR_RECALIBRATE", "0") == "1",
    )
    args.output_json.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[OSCAR] Rotations {result['status']}; ready for final preflight.", flush=True)


if __name__ == "__main__":
    main()
