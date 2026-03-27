import re
from pathlib import Path

import pytest
import torch

import lightbench
from lightbench import _EXCLUDED

_pkg = Path(lightbench.__file__).resolve().parent
_utils = (_pkg / "utils.py").read_text()
_runner = (_pkg / "run_all_benchmarks.py").read_text()


# --- resolve_dtype (pure, no heavy deps) ------------------------------------

class TestResolveDtype:
    def test_string(self):
        assert lightbench.resolve_dtype("float32") is torch.float32

    def test_bfloat16(self):
        assert lightbench.resolve_dtype("bfloat16") is torch.bfloat16

    def test_torch_dtype(self):
        assert lightbench.resolve_dtype(torch.float16) is torch.float16

    def test_single_element_list(self):
        assert lightbench.resolve_dtype(["float64"]) is torch.float64

    def test_single_element_tuple(self):
        assert lightbench.resolve_dtype(("float64",)) is torch.float64

    def test_empty_and_none(self):
        assert lightbench.resolve_dtype([]) is None
        assert lightbench.resolve_dtype(None) is None

    def test_rejects_multiple(self):
        with pytest.raises(ValueError):
            lightbench.resolve_dtype(["float16", "float32"])

    def test_rejects_bad_name(self):
        with pytest.raises(ValueError):
            lightbench.resolve_dtype("not_a_dtype")

    def test_rejects_non_dtype_attr(self):
        with pytest.raises(ValueError):
            lightbench.resolve_dtype("cuda")

    def test_rejects_arbitrary_object(self):
        with pytest.raises(TypeError):
            lightbench.resolve_dtype(object())


# --- Registry ---------------------------------------------------------------

def test_available_excludes_helpers():
    assert _EXCLUDED.isdisjoint(lightbench.available())


def test_available_has_real_benchmarks():
    names = set(lightbench.available())
    assert len(names) > 40
    for expected in ("beale", "rosenbrock", "rastrigin", "MNIST", "CIFAR10_wide",
                     "powers", "spiral", "SVHN", "Tolstoi_RNN"):
        assert expected in names, f"missing benchmark: {expected}"


def test_available_backed_by_source():
    for name in lightbench.available():
        assert (_pkg / f"{name}.py").exists()


def test_runner_derives_from_available():
    assert "from lightbench import available" in _runner
    # no hardcoded benchmark filenames
    for name in ("beale.py", "rosenbrock.py", "rastrigin.py"):
        assert name not in _runner


# --- CLI entrypoint ----------------------------------------------------------

def test_cli_exists():
    src = (_pkg / "cli.py").read_text()
    assert "from lightbench.run_all_benchmarks import app" in src


# --- Structural invariants (source-level) ------------------------------------

def test_no_singleton_list_indexing():
    for path in _pkg.glob("*.py"):
        text = path.read_text()
        for needle in ("opt[0]", "dtype[0]", "dtypes[0]"):
            assert needle not in text, f"{path.name} contains {needle}"


def test_no_raw_getattr_torch_dtype():
    """No benchmark should resolve dtypes via raw getattr(torch, ...).

    The canonical path is lightbench.resolve_dtype().  The only legitimate
    use of getattr(torch, ...) is inside resolve_dtype itself (__init__.py).
    """
    for path in _pkg.glob("*.py"):
        if path.name == "__init__.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"getattr\(torch,\s*\w+\)", line):
                pytest.fail(f"{path.name}:{i} uses raw getattr(torch, ...): {line.strip()}")


def test_runner_passes_scalars():
    m = re.search(r"arguments\s*=\s*\{.*?\}", _runner, re.DOTALL)
    assert m
    block = m.group(0)
    assert '"dtype": dtype' in block
    assert '"opt": opt' in block
    assert "[dtype]" not in block
    assert "[opt]" not in block


def test_success_metric():
    assert "Successfully found the minimum" not in _runner
    assert "Win: Yes" in _runner
    assert "Win: Yes" in _utils


def test_failure_threshold_removed():
    trial_sig = re.search(r"^def trial\(.*?\):", _utils, re.DOTALL | re.MULTILINE)
    assert trial_sig and "failure_threshold" not in trial_sig.group(0)
    obj_init = re.search(r"class Objective:.*?def __init__\(.*?\):", _utils, re.DOTALL)
    assert obj_init and "failure_threshold" not in obj_init.group(0)


def test_prev_best_tracked():
    assert "nonlocal prev_best" in _utils
    assert "prev_best = out" in _utils


def test_callback_results_attribute():
    assert 'hasattr(self, "test_accuracies")' not in _utils
    assert 'hasattr(self, "callback_results")' in _utils


def test_device_parameterized():
    for cls in ("FailureCounter", "Validator"):
        m = re.search(rf"class {cls}:.*?(?=\nclass |\Z)", _utils, re.DOTALL)
        assert m
        for line in m.group(0).split("\n"):
            if line.lstrip().startswith("def "):
                continue
            assert 'device="cuda"' not in line, f"{cls} hardcodes cuda: {line.strip()}"


# --- Packaging ---------------------------------------------------------------

def test_runtime_deps():
    toml = _pkg.parent / "pyproject.toml"
    if not toml.exists():
        pytest.skip("not in source tree")
    text = toml.read_text()
    for dep in ("torchvision", "requests"):
        assert dep in text, f"missing: {dep}"
