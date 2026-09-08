import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend import prepare_rotations as prep
from oscar_ascend.calibration_data import load_texts, resolve_texts, token_prompts


@pytest.fixture
def model(tmp_path, monkeypatch):
    for key in ("OSCAR_CALIBRATION_DATA", "OSCAR_CALIBRATION_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    folder = tmp_path / "model with spaces"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "head_dim": 256,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 64,
                    "full_attention_interval": 4,
                }
            }
        )
    )
    (folder / "model.safetensors").write_bytes(b"test fixture, not model weights")
    return folder


def write_pair(directory, fingerprint, profile, run="test-only", bad_v=False):
    directory.mkdir(parents=True, exist_ok=True)
    for kind, objective in (("k", "qqt_r_h_pbr"), ("v", "sst_r_h_pbr")):
        matrix = torch.eye(256) if not (bad_v and kind == "v") else torch.zeros(256, 256)
        payload = {
            "objective": objective,
            "model_fingerprint": fingerprint,
            "calibration": {"profile_fingerprint": profile, "run_id": run},
            "layers": {i: {"rotation": matrix} for i in range(3, 64, 4)},
        }
        torch.save(payload, directory / f"{kind}_rotation.pt")


def test_missing_generates_then_reuses_without_model_loading(model, tmp_path):
    calls = []

    def generate(model_path, output, fingerprint, profile):
        calls.append(model_path)
        write_pair(output, fingerprint, profile)

    first = prep.prepare(model, tmp_path / "cache", generator=generate)
    second = prep.prepare(model, tmp_path / "cache", generator=generate)
    assert first["status"] == "generated" and second["status"] == "reused"
    assert first["k"] == second["k"] and first["v"] == second["v"]
    assert len(calls) == 1


def test_missing_one_default_file_regenerates_coherent_pair(model, tmp_path):
    calls = []

    def generate(_, output, fingerprint, profile):
        calls.append(1)
        write_pair(output, fingerprint, profile, run=str(len(calls)))

    first = prep.prepare(model, tmp_path / "cache", generator=generate)
    Path(first["v"]).unlink()
    result = prep.prepare(model, tmp_path / "cache", generator=generate)
    assert result["status"] == "generated" and len(calls) == 2
    for kind in ("k", "v"):
        assert torch.load(result[kind], weights_only=True)["calibration"]["run_id"] == "2"


def test_generator_failure_never_publishes_a_partial_pair(model, tmp_path):
    def failed(_, output, fingerprint, profile):
        write_pair(output, fingerprint, profile)
        raise RuntimeError("NPU compilation failed")

    with pytest.raises(RuntimeError, match="NPU compilation"):
        prep.prepare(model, tmp_path / "cache", generator=failed)
    assert not list((tmp_path / "cache").rglob("*.pt"))


def test_invalid_generated_pair_never_published(model, tmp_path):
    def bad(_, output, fingerprint, profile):
        write_pair(output, fingerprint, profile, bad_v=True)

    with pytest.raises(RuntimeError, match="Generated V"):
        prep.prepare(model, tmp_path / "cache", generator=bad)
    assert not list((tmp_path / "cache").rglob("*.pt"))


def test_explicit_valid_file_preserved_when_other_side_missing(model, tmp_path):
    supplied = tmp_path / "supplied_k.pt"
    torch.save({i: torch.eye(256) for i in range(3, 64, 4)}, supplied)
    original = supplied.read_bytes()

    def generate(_, output, fingerprint, profile):
        write_pair(output, fingerprint, profile)

    result = prep.prepare(model, tmp_path / "cache", k_path=supplied, generator=generate)
    assert result["status"] == "generated"
    assert supplied.read_bytes() == original
    assert Path(result["v"]).is_file()


def test_invalid_explicit_pt_is_not_overwritten(model, tmp_path):
    supplied = tmp_path / "not-a-rotation.pt"
    supplied.write_bytes(b"must be preserved")
    with pytest.raises(ValueError, match="kept unchanged"):
        prep.prepare(
            model,
            tmp_path / "cache",
            k_path=supplied,
            generator=lambda *args: pytest.fail("Must not generate"),
        )
    assert supplied.read_bytes() == b"must be preserved"


def test_model_and_data_changes_get_new_cache_keys(model, tmp_path, monkeypatch):
    before = prep.model_fingerprint(model)
    with (model / "model.safetensors").open("ab") as stream:
        stream.write(b"new checkpoint")
    assert prep.model_fingerprint(model) != before
    profile = prep.profile_fingerprint()
    data = tmp_path / "prompts.jsonl"
    data.write_text(json.dumps({"text": "User workload example"}) + "\n")
    monkeypatch.setenv("OSCAR_CALIBRATION_DATA", str(data))
    assert prep.profile_fingerprint() != profile


def test_mismatched_run_ids_are_regenerated(model, tmp_path):
    def generate(_, output, fingerprint, profile):
        write_pair(output, fingerprint, profile)

    first = prep.prepare(model, tmp_path / "cache", generator=generate)
    value = torch.load(first["v"], weights_only=True)
    value["calibration"]["run_id"] = "different"
    torch.save(value, first["v"])
    assert prep.prepare(model, tmp_path / "cache", generator=generate)["status"] == "generated"


def test_data_is_offline_and_user_jsonl_is_supported(tmp_path):
    builtin, kind = load_texts()
    assert len(builtin) == 16 and kind == "builtin-bootstrap-v1"
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join([json.dumps({"prompt": "Prompt A"}), json.dumps("Prompt B")]))
    assert load_texts(path) == (["Prompt A", "Prompt B"], "user-jsonl")
    path.write_text("{}")
    with pytest.raises(ValueError, match="JSONL"):
        load_texts(path)


def test_prompt_token_budget_is_bounded():
    tokenizer = SimpleNamespace(
        chat_template=None, encode=lambda text, **kw: list(range(len(text)))
    )
    prompts = token_prompts(
        tokenizer, ["A long enough test passage about a subject."], 1024, builtin=True
    )
    assert len(prompts[0]["prompt_token_ids"]) == 1024


def test_short_user_prompts_are_skipped_not_fatal(capsys):
    tokenizer = SimpleNamespace(
        chat_template=None, encode=lambda text, **kw: list(range(len(text)))
    )
    usable = "A long enough test passage about a subject."
    prompts = token_prompts(tokenizer, ["hi", usable, "ok"], 1024)
    assert len(prompts) == 1
    assert "Skipped 2" in capsys.readouterr().out
    with pytest.raises(ValueError, match="below 32 tokens"):
        token_prompts(tokenizer, ["hi", "ok"], 1024)


@pytest.mark.parametrize(
    "apply_chat_template",
    [
        lambda *a, **kw: [1, 2, 3],  # renders any text to a handful of ids
        lambda *a, **kw: "not even a token list",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("template error")),
    ],
)
def test_broken_chat_template_falls_back_to_plain_encoding(apply_chat_template):
    tokenizer = SimpleNamespace(
        chat_template="<broken-jinja>",
        apply_chat_template=apply_chat_template,
        encode=lambda text, **kw: list(range(len(text))),
    )
    prompts = token_prompts(
        tokenizer, ["A long enough test passage about a subject."], 1024, builtin=True
    )
    assert len(prompts[0]["prompt_token_ids"]) == 1024


def test_resolve_texts_falls_back_when_data_unusable(tmp_path):
    builtin = load_texts(None) + (True,)
    assert resolve_texts() == builtin
    assert resolve_texts(tmp_path / "missing.jsonl") == builtin
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    assert resolve_texts(empty) == builtin
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps("A real workload prompt line") + "\n")
    assert resolve_texts(good) == (["A real workload prompt line"], "user-jsonl", False)


def test_missing_data_file_fingerprints_as_builtin(monkeypatch):
    monkeypatch.delenv("OSCAR_CALIBRATION_DATA", raising=False)
    builtin = prep.profile_fingerprint()
    monkeypatch.setenv("OSCAR_CALIBRATION_DATA", "/nonexistent/prompts.jsonl")
    assert prep.profile_fingerprint() == builtin


def test_calibration_falls_back_when_all_prompts_too_short(model, tmp_path, monkeypatch, capsys):
    from oscar_ascend import calibrate

    data = tmp_path / "short.jsonl"
    data.write_text(json.dumps({"text": "hi"}) + "\n")
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
            )

        def get_tokenizer(self):
            # "hi" encodes to 2 ids; builtin passages repeat to thousands of ids.
            return SimpleNamespace(
                chat_template=None, encode=lambda text, **kw: list(range(len(text)))
            )

        def collective_rpc(self, method, **kwargs):
            calls.append(method)
            return [{"published": True}]

        def generate(self, *args, **kwargs):
            calls.append("generate")

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw)
    )
    # calibrate.main() mutates these directly; pre-set so monkeypatch restores them.
    monkeypatch.setenv("OSCAR_ASCEND_ENABLED", "1")
    monkeypatch.setenv("OSCAR_ASCEND_CALIBRATING", "0")
    monkeypatch.setenv("OSCAR_CALIBRATION_DATA", str(data))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--model",
            str(model),
            "--output-dir",
            str(tmp_path),
            "--model-fingerprint",
            "model-id",
            "--profile-fingerprint",
            "profile-id",
        ],
    )
    calibrate.main()
    assert calls == [
        "oscar_calibration_begin",
        "generate",
        "oscar_calibration_second_pass",
        "generate",
        "oscar_calibration_finish",
        "shutdown",
    ]
    assert "falling back to the builtin bootstrap texts" in capsys.readouterr().out


def test_calibration_runs_two_native_passes_and_shuts_down(model, tmp_path, monkeypatch):
    from oscar_ascend import calibrate

    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
            assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
            assert kwargs["enforce_eager"] and not kwargs["enable_prefix_caching"]
            assert kwargs["tensor_parallel_size"] == 4 and kwargs["quantization"] == "ascend"
            assert (
                kwargs["worker_extension_cls"]
                == "oscar_ascend.calibration_worker.CalibrationWorkerExtension"
            )
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
            )

        def get_tokenizer(self):
            return SimpleNamespace(chat_template=None, encode=lambda text, **kw: list(range(1024)))

        def collective_rpc(self, method, **kwargs):
            # Functions fail vLLM's default utility-message serialization.
            # Our control plane must use strings plus plain serializable data.
            assert isinstance(method, str)
            json.dumps((method, kwargs))
            calls.append(method)
            return [{"published": True}]

        def generate(self, *args, **kwargs):
            calls.append("generate")

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw)
    )
    monkeypatch.setenv("OSCAR_ASCEND_ENABLED", "1")
    monkeypatch.setenv("OSCAR_ASCEND_CALIBRATING", "0")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--model",
            str(model),
            "--output-dir",
            str(tmp_path),
            "--model-fingerprint",
            "model-id",
            "--profile-fingerprint",
            "profile-id",
        ],
    )
    calibrate.main()
    assert calls == [
        "oscar_calibration_begin",
        "generate",
        "oscar_calibration_second_pass",
        "generate",
        "oscar_calibration_finish",
        "shutdown",
    ]


def test_named_worker_rpcs_delegate_without_serializing_functions(monkeypatch):
    from oscar_ascend import calibration_worker as worker

    seen = []
    for name in ("begin", "second_pass", "finish"):

        def callback(*args, _name=name):
            seen.append((_name, args))
            return {"phase": _name}

        monkeypatch.setattr(worker, name, callback)
    extension = worker.CalibrationWorkerExtension()
    assert extension.oscar_calibration_begin([3, 7], 1024) == {"phase": "begin"}
    assert extension.oscar_calibration_second_pass() == {"phase": "second_pass"}
    assert extension.oscar_calibration_finish("/output", {"key": "value"}, 12) == {
        "phase": "finish"
    }
    assert seen == [
        ("begin", (extension, [3, 7], 1024)),
        ("second_pass", (extension,)),
        ("finish", (extension, "/output", {"key": "value"}, 12)),
    ]


def test_calibration_kwargs_exist_in_pinned_vllm_api():
    import ast

    root = Path(__file__).resolve().parents[1]
    api = root / "references/vllm/vllm/entrypoints/llm.py"
    engine = root / "references/vllm/vllm/engine/arg_utils.py"
    if not api.exists():
        pytest.skip("Read-only source references are not shipped to deployment")
    ours = ast.parse((root / "oscar_ascend/calibrate.py").read_text())
    call = next(
        n
        for n in ast.walk(ours)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LLM"
    )
    supplied = {kw.arg for kw in call.keywords}
    llm_class = next(
        n
        for n in ast.parse(api.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "LLM"
    )
    initializer = next(
        n for n in llm_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    accepted = {arg.arg for arg in initializer.args.args + initializer.args.kwonlyargs}
    args_class = next(
        n
        for n in ast.parse(engine.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "EngineArgs"
    )
    accepted |= {
        n.target.id
        for n in args_class.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    assert supplied <= accepted, supplied - accepted
    assert "worker_extension_cls" in supplied


def test_calibration_child_always_uses_authorized_physical_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    seen = []
    monkeypatch.setattr(
        prep.subprocess, "run", lambda command, **kwargs: seen.append(kwargs["env"])
    )
    prep.run_generator("model", tmp_path, "model-hash", "profile-hash")
    assert seen[0]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
    assert seen[0]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert seen[0]["OSCAR_ASCEND_ENABLED"] == "0"
    assert seen[0]["OSCAR_ASCEND_CALIBRATING"] == "1"


def test_every_shell_launcher_pins_cards_before_python():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "scripts").glob("*.sh"):
        lines = path.read_text().splitlines()
        pin = lines.index("export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7")
        assert pin < next(i for i, line in enumerate(lines) if line.startswith("PYTHON_BIN="))
        start = lines.index("export VLLM_WORKER_MULTIPROC_METHOD=spawn")
        assert start < next(i for i, line in enumerate(lines) if line.startswith("PYTHON_BIN="))


@pytest.mark.parametrize("calibration_fails", [False, True])
def test_one_command_installs_prepares_and_serves_in_order(tmp_path, calibration_fails):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "fake-python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
args = sys.argv[1:]
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
with open(os.environ["FAKE_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:2] == ["-m", "oscar_ascend.prepare_rotations"]:
    if os.environ.get("FAKE_FAIL") == "1":
        sys.exit(17)
    path = args[args.index("--output-json") + 1]
    with open(path, "w") as stream:
        json.dump({"k": "/cache with spaces/k.pt", "v": "/cache with spaces/v.pt"}, stream)
elif args[0] == "-c":
    sys.argv = ["-c"] + args[2:]
    exec(args[1])
elif args[:2] == ["-m", "vllm.entrypoints.cli.main"]:
    assert os.environ["VLLM_OSCAR_K_ROTATION_PATH"] == "/cache with spaces/k.pt"
    assert os.environ["VLLM_OSCAR_V_ROTATION_PATH"] == "/cache with spaces/v.pt"
"""
    )
    fake.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake),
        MODEL="/fake model",
        MODE="oscar",
        FAKE_LOG=str(log),
        FAKE_FAIL=str(int(calibration_fails)),
        TMPDIR=str(tmp_path),
        ASCEND_RT_VISIBLE_DEVICES="0,1,2,3",
        VLLM_WORKER_MULTIPROC_METHOD="fork",
    )
    for key in ("DRY_RUN", "VLLM_OSCAR_K_ROTATION_PATH", "VLLM_OSCAR_V_ROTATION_PATH"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(root / "scripts/serve.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode == (17 if calibration_fails else 0), result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    modules = [args[1] for args in calls if args[0] == "-m"]
    expected = ["pip", "oscar_ascend.check", "oscar_ascend.prepare_rotations"]
    if not calibration_fails:
        expected += ["oscar_ascend.check", "vllm.entrypoints.cli.main"]
    assert modules == expected
    assert not list(tmp_path.glob("oscar-rotation-paths.*"))
