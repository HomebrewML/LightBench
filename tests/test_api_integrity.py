from pathlib import Path

import pytest
import torch

from lightbench import utils


def test_resolve_dtype_accepts_common_inputs():
    assert utils._resolve_dtype("float32") is torch.float32
    assert utils._resolve_dtype(torch.float16) is torch.float16
    assert utils._resolve_dtype(["float64"]) is torch.float64
    assert utils._resolve_dtype([]) is None
    assert utils._resolve_dtype(None) is None


def test_resolve_dtype_rejects_bad_inputs():
    with pytest.raises(ValueError):
        utils._resolve_dtype(["float16", "float32"])

    with pytest.raises(ValueError):
        utils._resolve_dtype("not_a_dtype")

    with pytest.raises(TypeError):
        utils._resolve_dtype(object())


def test_lightbench_modules_do_not_index_singleton_cli_lists():
    import lightbench

    lightbench_dir = Path(lightbench.__file__).resolve().parent
    offenders: list[tuple[str, str]] = []
    for path in lightbench_dir.glob("*.py"):
        text = path.read_text()
        for needle in ("opt[0]", "dtype[0]", "dtypes[0]"):
            if needle in text:
                offenders.append((path.name, needle))

    assert offenders == []
