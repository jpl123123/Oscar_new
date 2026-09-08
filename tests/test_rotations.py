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
