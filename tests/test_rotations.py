import pytest
import torch

from oscar_ascend.rotations import get_rotation, layer_index, load_checkpoint


@pytest.mark.parametrize(
    "name,expected",
    [("model.layers.3.self_attn.attn", 3), ("model.language_model.layers.63.self_attn.attn", 63)],
)
def test_layer_name(name, expected):
    assert layer_index(name) == expected


def test_missing_layer_or_wrong_dimension_is_not_identity(tmp_path):
    path = tmp_path / "rot.pt"
    torch.save({"layers": {3: {"rotation": torch.eye(256)}}}, path)
    torch.testing.assert_close(
        get_rotation(str(path), "model.layers.3.attn", "cpu"), torch.eye(256).bfloat16()
    )
    with pytest.raises(ValueError, match="Missing"):
        get_rotation(str(path), "model.layers.7.attn", "cpu")
    with pytest.raises(ValueError, match="D128"):
        get_rotation(str(path), "model.layers.3.attn", "cpu", 128)
    with pytest.raises(ValueError):
        load_checkpoint("")


def test_nonorthogonal_rotation_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({3: torch.ones(256, 256)}, path)
    with pytest.raises(ValueError, match="orthogonal"):
        get_rotation(str(path), "model.layers.3.attn", "cpu")


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("default_device", ["cpu", "meta"])
def test_rotation_loading_inside_model_loader_defaults(
    tmp_path, default_dtype, default_device
):
    # vLLM's post-load hook runs inside set_default_torch_dtype(model_config.dtype).
    # A meta default device also exposes accidental placement without needing an NPU.
    generator = torch.Generator(device="cpu").manual_seed(7)
    matrix, _ = torch.linalg.qr(
        torch.randn(256, 256, generator=generator, dtype=torch.float32, device="cpu")
    )
    path = tmp_path / "calibrated.pt"
    torch.save({"layers": {3: {"rotation": matrix}}}, path)
    original = path.read_bytes()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        with torch.device(default_device):
            result = get_rotation(str(path), "model.layers.3.attn", "cpu")
            assert torch.get_default_dtype() == default_dtype
            assert torch.empty(0).device.type == default_device
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.testing.assert_close(result, matrix.bfloat16())
    cached = load_checkpoint(str(path))[3]
    assert cached.dtype == torch.float32 and cached.device.type == "cpu"
    assert path.read_bytes() == original


def test_nonorthogonal_rotation_still_rejected_with_bf16_default(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({3: torch.ones(256, 256, dtype=torch.float32, device="cpu")}, path)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with pytest.raises(ValueError, match="orthogonal"):
            get_rotation(str(path), "model.layers.3.attn", "cpu")
    finally:
        torch.set_default_dtype(previous_dtype)
