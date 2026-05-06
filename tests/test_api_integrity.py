import json
import re
from pathlib import Path

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

import lightbench
from lightbench import _EXCLUDED, utils
from lightbench.run_all_benchmarks import _is_dtype_gated
from lightbench.run_all_benchmarks import app as runner_app
from lightbench.utils import _resolve_optim_callable, difficulty_to_kwargs

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
    assert len(names) > 30
    for expected in (
        "powers",
        "absolute_varying_scale",
        "gradient_delay",
        "wide_linear",
        "xor_digit",
    ):
        assert expected in names, f"missing benchmark: {expected}"
    for excluded in (
        "beale",
        "rosenbrock",
        "rastrigin",
        "spiral",
        "discontinuous_gradient",
        "MNIST",
        "CIFAR100",
        "SVHN",
        "Tolstoi_RNN",
    ):
        assert excluded not in names


def test_available_backed_by_source():
    for name in lightbench.available():
        assert (_pkg / f"{name}.py").exists()


# --- Difficulty mapping ------------------------------------------------------


def test_difficulty_to_kwargs_single_param():
    spec = {"param": "size", "lo": 4, "hi": 64, "scale": "log", "dtype": "int"}
    assert difficulty_to_kwargs(spec, 0.0)["size"] == 4
    assert difficulty_to_kwargs(spec, 1.0)["size"] == 64
    assert difficulty_to_kwargs(spec, 0.5)["size"] == 16


def test_difficulty_to_kwargs_multi_param():
    spec = {
        "params": {
            "a": {"lo": 1, "hi": 5, "scale": "linear", "dtype": "int"},
            "b": {"lo": 0.1, "hi": 0.9, "scale": "linear"},
        }
    }
    res = difficulty_to_kwargs(spec, 0.5)
    assert res["a"] == 3
    assert abs(res["b"] - 0.5) < 1e-8


def test_difficulty_to_kwargs_matches_endpoints():
    spec = {
        "params": {
            "classes": {"lo": 2, "hi": 2048, "scale": "log", "dtype": "int"},
            "add_class_every": {"lo": 128, "hi": 4, "scale": "log", "dtype": "int"},
        }
    }
    lo = difficulty_to_kwargs(spec, 0.0)
    hi = difficulty_to_kwargs(spec, 1.0)
    assert lo["classes"] == 2
    assert lo["add_class_every"] == 128
    assert hi["classes"] == 2048
    assert hi["add_class_every"] == 4


def test_runner_derives_from_available():
    assert "from lightbench import available" in _runner
    # no hardcoded benchmark filenames
    for name in ("beale.py", "rosenbrock.py", "rastrigin.py"):
        assert name not in _runner


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
    assert '"dtype":' in block
    assert '"opt":' in block
    assert "[dtype]" not in block
    assert "[opt]" not in block


def test_success_metric():
    assert "Successfully found the minimum" not in _runner
    assert '"success": bool(r["did_win"])' in _runner
    assert "Win:" in _utils
    assert "'Yes' if obj.did_win else 'No'" in _utils


def test_failure_threshold_removed():
    trial_sig = re.search(r"^def trial\(.*?\):", _utils, re.DOTALL | re.MULTILINE)
    assert trial_sig and "failure_threshold" not in trial_sig.group(0)
    obj_init = re.search(r"class Objective:.*?def __init__\(.*?\):", _utils, re.DOTALL)
    assert obj_init and "failure_threshold" not in obj_init.group(0)


def test_objective_exceptions_reported_as_errors():
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.ones(()))

        def forward(self):
            raise RuntimeError("boom")

    utils.trial(
        Bad(),
        None,
        None,
        utils.disabled_win_condition,
        steps=1,
        opt=torch.optim.SGD,
        weight_decay=0,
        trials=1,
        device="cpu",
    )
    assert "objective exception" in utils._trial_result["error"]
    assert "RuntimeError:boom" in utils._trial_result["error"]


def test_callback_results_attribute():
    assert 'hasattr(self, "test_accuracies")' not in _utils
    assert "self.callback_results = []" in _utils


def test_device_parameterized():
    for cls in ("Validator",):
        m = re.search(rf"class {cls}:.*?(?=\nclass |\Z)", _utils, re.DOTALL)
        assert m
        for line in m.group(0).split("\n"):
            if line.lstrip().startswith("def "):
                continue
            assert 'device="cuda"' not in line, f"{cls} hardcodes cuda: {line.strip()}"


def test_optimizer_aliases_resolve():
    assert _resolve_optim_callable("ForeachSOAP").__name__ == "SOAP"
    assert _resolve_optim_callable("adamw").__name__ == "AdamW"


def test_resume_all_complete_returns_before_cuda(tmp_path):
    row = {
        "benchmark": "absolute_varying_scale",
        "difficulty": 0.5,
        "opt": "AdamW",
        "dtype": "fp32",
        "success": False,
        "runtime": 0.0,
        "attempts": 1,
        "timed_out": False,
        "loss": 1.0,
        "error": "",
        "steps": 1,
        "trials": 1,
        "timeout": 1,
        "winning": None,
    }
    log = tmp_path / "done.jsonl"
    log.write_text(json.dumps(row) + "\n")
    result = CliRunner().invoke(
        runner_app,
        [
            "--opt", "AdamW",
            "--steps", "1",
            "--trials", "1",
            "--timeout", "1",
            "--min-evals", "1",
            "--max-evals", "1",
            "--include-benchmark", "absolute_varying_scale",
            "--progress-log", str(log),
            "--resume",
            "--no-irt",
        ],
    )
    assert result.exit_code == 0
    assert "No pending tasks." in result.output


def test_dtype_gating_checks_ancestors():
    assert _is_dtype_gated("bf16", {"bf16+ecc": True})
    assert _is_dtype_gated("fp16", {"fp16+ecc": True})
    assert not _is_dtype_gated("bf16+ecc", {"bf16": True})


# --- Packaging ---------------------------------------------------------------


def test_runtime_deps():
    toml = _pkg.parent / "pyproject.toml"
    if not toml.exists():
        pytest.skip("not in source tree")
    text = toml.read_text()
    for dep in ("torchvision", "requests"):
        assert dep in text, f"missing: {dep}"
