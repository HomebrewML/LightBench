import copy
import functools
import gc
import inspect
import os
import random
import sys
import time
import warnings
from typing import Callable, Optional

import heavyball.utils
import numpy as np
import optuna
import torch
from heavyball import chainable as C
from heavyball.helpers import AutoSampler
from heavyball.utils import PrecondInitError
from torch import nn
from torch._dynamo import config as dynamo_config
from torch.func import functional_call, vmap

from lightbench import resolve_dtype

dynamo_config.cache_size_limit = 2**16
np.warnings = warnings


# --- Optimizer setup ---

BASE_OPTIM_ARGS = {
    "betas": (0.9, 0.999),
    "precondition_frequency": 16,
    "merge_dims": True,
    "warmup_steps": 10,
    "max_precond_dim": 2**16,
    "beta": 0.9,
    "max_size_triangular": 2**16,
    "split": False,
    "precond_grad_accum": False,
    "momentum_into_precond_update": True,
    "eps": 1e-8,
    "weight_decay": 0,
    "precond_update_power_iterations": 16,
    "dampening": 2**-24,
    "preconditioner_update_probability": 1.0,
    "precond_init_scale": 1.0,
    "precondition_1d": True,
    "store_triu_as_line": False,
    "update_clipping": None,
    "delayed": True,
}

_OPT_PREFIXES = [
    ("unscaled_cautious-", {"caution": True, "_unscale_caution": True}),
    ("cautious-", {"caution": True}),
    ("mars-", {"mars": True}),
    ("unscaled-", {"unscaled": True}),
    ("adaptive-", {"adaptive": True}),
    ("ortho-", {"ortho_method": "newtonschulz-graft"}),
]


def get_optim(optim: str | C.BaseOpt, params, **kwargs) -> C.BaseOpt:
    args = {**BASE_OPTIM_ARGS, **kwargs}
    if isinstance(optim, str):
        optim = getattr(heavyball, optim)
    sig = inspect.signature(optim)
    return optim(params, **{k: v for k, v in args.items() if k in sig.parameters})


def parse_opt_name(name: str) -> tuple[str, dict]:
    kwargs = {"caution": False, "mars": False}
    for prefix, kw in _OPT_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            kwargs.update(kw)
    return name, kwargs


# --- Validator ---

def _fc_compare(inp, other, minimal_improvement):
    return other.reshape(1, 1, -1) * (1 - minimal_improvement.reshape(-1, 1, 1)) < inp.reshape(1, -1, 1)


def _fc_step(comparison, consecutive_failures, max_consecutive_failures, failure_scale):
    failed = torch.any(comparison, dim=tuple(range(1, comparison.ndim)))
    new_cf = torch.where(failed, consecutive_failures + 1, torch.zeros_like(consecutive_failures))
    threshold = (max_consecutive_failures.unsqueeze(-1) * failure_scale).flatten()
    return torch.any(new_cf >= threshold), new_cf


def _validator_step(
    loss,
    step, seen_until, ema_states, update_factor,
    global_min_loss, gm_cf, gm_mcf, gm_mi,
    global_avg_loss, global_avg_step, ga_cf, ga_mcf, ga_mi,
    ema_cf, ema_mcf, ema_mi,
    weighting, indices, triu_mask, warmup, ema_index, ema_patience,
):
    new_step = step + 1
    new_seen_until = torch.maximum(seen_until, new_step - 1)

    beta = 1 - update_factor
    uf = update_factor / (1 - beta ** new_step)
    new_ema = ema_states + uf * (loss - ema_states)
    ema_loss = new_ema[ema_index]

    gm_comp = _fc_compare(ema_loss, global_min_loss, gm_mi)
    gm_failed, new_gm_cf = _fc_step(gm_comp.view(-1, 1), gm_cf, gm_mcf, weighting)
    step_ge = indices >= (new_step - 1)
    update_mask = step_ge & (ema_loss < global_min_loss) & torch.isfinite(ema_loss)
    new_gml = torch.where(update_mask, ema_loss.expand_as(global_min_loss), global_min_loss)

    step_eq = indices == (new_step - 1)
    new_gas = global_avg_step + step_eq.to(global_avg_step.dtype)
    lerp_w = torch.where(step_eq, 1.0 / new_gas.clamp(min=1), torch.zeros_like(new_gas))
    new_gal = global_avg_loss + lerp_w * (ema_loss - global_avg_loss)
    ga_comp = _fc_compare(ema_loss, new_gal, ga_mi).view(-1, 1)
    ga_comp = ga_comp & (indices < (new_seen_until - 1)).view(-1, 1)
    ga_failed, new_ga_cf = _fc_step(ga_comp, ga_cf, ga_mcf, weighting)

    lc_comp = _fc_compare(new_ema, new_ema, ema_mi)
    lc_flat = lc_comp[:, triu_mask]
    lc_failed, new_ema_cf = _fc_step(lc_flat, ema_cf, ema_mcf, ema_patience)

    result = (gm_failed | ga_failed | lc_failed) & (new_step >= warmup)
    return (result, new_step, new_seen_until, new_ema,
            new_gm_cf, new_ga_cf, new_ema_cf,
            new_gml, new_gal, new_gas)


_compiled_validator_step = torch.compile(_validator_step, mode="max-autotune-no-cudagraphs", fullgraph=True)


class Validator:
    ema_index: int = 4
    warmup: int = 256
    ema_patience: float = 2
    ema_start: int = 0

    def __init__(self, ema_mapping, global_mapping, steps, emas: int = 20, device: torch.device | str = "cuda"):
        self.emas = emas
        self.device = torch.device(device)
        self.steps = steps

        self.ema_states = torch.zeros(emas, dtype=torch.float64, device=self.device)
        es = self.ema_start + 1
        self.update_factor = 2.0 ** (-torch.arange(es, 20 + es, dtype=torch.float64, device=self.device))

        ema_mcf, ema_mi = zip(*ema_mapping.items())
        self.ema_mcf = torch.tensor(ema_mcf, dtype=torch.float64, device=self.device)
        self.ema_mi = torch.tensor(ema_mi, dtype=torch.float64, device=self.device)
        self.ema_cf = torch.zeros(len(ema_mi), dtype=torch.int64, device=self.device)

        triu_idx = torch.triu_indices(emas, emas, offset=1)
        self.triu_mask = torch.zeros(emas, emas, dtype=torch.bool, device=self.device)
        self.triu_mask[triu_idx[0], triu_idx[1]] = True

        self.global_min_loss = torch.full((steps,), float("inf"), dtype=torch.float64, device=self.device)
        self.gm_mcf = torch.tensor([1.0], dtype=torch.float64, device=self.device)
        self.gm_mi = torch.tensor([0.0], dtype=torch.float64, device=self.device)
        self.gm_cf = torch.zeros(steps, dtype=torch.int64, device=self.device)

        self.global_avg_loss = torch.zeros(steps, dtype=torch.float64, device=self.device)
        self.global_avg_step = torch.zeros(steps, dtype=torch.float64, device=self.device)
        self.ga_mcf = torch.tensor([1.0], dtype=torch.float64, device=self.device)
        self.ga_mi = torch.tensor([0.0], dtype=torch.float64, device=self.device)
        self.ga_cf = torch.zeros(steps, dtype=torch.int64, device=self.device)

        w = torch.arange(1, 1 + steps, device=self.device, dtype=torch.float64).clamp(min=self.warmup).unsqueeze(0)
        self.weighting = functools.reduce(torch.minimum, [w ** p * f for f, p in global_mapping.items()])

        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.seen_until = torch.zeros((), dtype=torch.int64, device=self.device)
        self.indices = torch.arange(steps, dtype=torch.int64, device=self.device)

    def new(self):
        n = copy.copy(self)
        n.ema_states = torch.zeros_like(self.ema_states)
        n.ema_cf = torch.zeros_like(self.ema_cf)
        n.gm_cf = torch.zeros_like(self.gm_cf)
        n.ga_cf = torch.zeros_like(self.ga_cf)
        n.step = torch.zeros((), dtype=torch.int64, device=self.device)
        return n

    def _step_args(self, loss, step, seen_until, ema_states, gm_cf, ga_cf, ema_cf, gml, gal, gas):
        return (
            loss,
            step, seen_until, ema_states, self.update_factor,
            gml, gm_cf, self.gm_mcf, self.gm_mi,
            gal, gas, ga_cf, self.ga_mcf, self.ga_mi,
            ema_cf, self.ema_mcf, self.ema_mi,
            self.weighting, self.indices, self.triu_mask,
            self.warmup, self.ema_index, self.ema_patience,
        )

    def _write_shared(self, seen_until, gml, gal, gas):
        self.seen_until.copy_(seen_until)
        self.global_min_loss.copy_(gml)
        self.global_avg_loss.copy_(gal)
        self.global_avg_step.copy_(gas)

    def __call__(self, loss):
        (result, self.step, seen_until, self.ema_states,
         self.gm_cf, self.ga_cf, self.ema_cf,
         gml, gal, gas
        ) = _compiled_validator_step(*self._step_args(
            loss, self.step, self.seen_until, self.ema_states,
            self.gm_cf, self.ga_cf, self.ema_cf,
            self.global_min_loss, self.global_avg_loss, self.global_avg_step,
        ))
        self._write_shared(seen_until, gml, gal, gas)
        return result

    def validate_batch(self, losses, n):
        step, seen_until, ema_states = self.step, self.seen_until, self.ema_states
        gm_cf, ga_cf, ema_cf = self.gm_cf, self.ga_cf, self.ema_cf
        gml, gal, gas = self.global_min_loss, self.global_avg_loss, self.global_avg_step

        triggered = torch.tensor(False, device=self.device)
        for k in range(n):
            (result, step, seen_until, ema_states, gm_cf, ga_cf, ema_cf, gml, gal, gas
            ) = _compiled_validator_step(*self._step_args(
                losses[k], step, seen_until, ema_states, gm_cf, ga_cf, ema_cf, gml, gal, gas,
            ))
            triggered = triggered | result

        self.step, self.ema_states = step, ema_states
        self.gm_cf, self.ga_cf, self.ema_cf = gm_cf, ga_cf, ema_cf
        self._write_shared(seen_until, gml, gal, gas)
        return triggered


# --- Exceptions ---

class Stop(Exception):
    pass


class SkipConfig(ValueError):
    pass


class WinConditionMet(ValueError):
    pass


# --- Plotting ---

class Plotter(nn.Module):
    def __init__(self, objective_fn, x_limits=(-5, 5), y_limits=(-5, 5),
                 resolution=300, transform=None, inverse_transform=None,
                 should_normalize: bool = True):
        super().__init__()
        self.should_normalize = should_normalize
        self.objective = objective_fn
        self.initial = objective_fn.param.data.clone()
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.resolution = resolution
        self.transform = transform or (lambda x: x)
        self.inverse_transform = inverse_transform or (lambda x: x)
        self.param = objective_fn.param

        with torch.no_grad():
            x = torch.linspace(x_limits[0], x_limits[1], resolution)
            y = torch.linspace(y_limits[0], y_limits[1], resolution)
            self.X, self.Y = torch.meshgrid(x, y, indexing="ij")
            dev = objective_fn.param.device
            base = objective_fn.param.data
            if not base.numel():
                raise ValueError("can only plot 2d functions")
            X, Y = self.X.to(dev), self.Y.to(dev)
            flat_params = torch.stack([X.flatten(), Y.flatten()], dim=-1)
            flat_params = flat_params.to(dtype=base.dtype).reshape(-1, *base.shape)
            transform = self.transform

            def _eval_at(params):
                return transform(functional_call(objective_fn, {"param": params}, ()))

            self.Z = vmap(_eval_at)(flat_params).reshape(resolution, resolution)

        self.trajectory = [self.initial.detach().cpu().numpy()]

    def forward(self, *args):
        value = self.objective(*args)
        with torch.no_grad():
            self.trajectory.append(self.param.cpu().detach().numpy())
        return self.transform(value)

    def plot(self, title=None, save_path=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        z = self.Z
        if self.should_normalize:
            z = z - z.min()
            z = z / z.max()
            z = z + 1e-8
        z = z.cpu()

        plt.contourf(self.X.numpy(), self.Y.numpy(), z.log().numpy(), levels=1000)
        plt.xlim(self.x_limits)
        plt.ylim(self.y_limits)

        trajectory = np.array(self.trajectory).reshape(-1, 2)
        plt.plot(trajectory[:, 0], trajectory[:, 1], "r.-", label="Optimization path")
        plt.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")

        label = "Log(NormalizedObjectiveValue)" if self.should_normalize else "Log(ObjectiveValue)"
        plt.colorbar(label=label)
        plt.xlabel("x")
        plt.ylabel("y")
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()


class MultiPlotter:
    def __init__(self, plotters):
        if not plotters:
            raise ValueError("MultiPlotter requires at least one Plotter instance.")
        self.plotters = plotters
        self._reference = plotters[0][1]

    def plot(self, title=None, save_path=None):
        import matplotlib.colors
        import matplotlib.pyplot as plt

        ref = self._reference
        fig, ax = plt.subplots(figsize=(10, 8))

        z = ref.Z
        if ref.should_normalize:
            z = z - z.min()
            z = z / z.max()
            z = z + 1e-8
        z = z.cpu()
        contour = ax.contourf(ref.X.numpy(), ref.Y.numpy(), z.log().numpy(), levels=1000)
        ax.set_xlim(ref.x_limits)
        ax.set_ylim(ref.y_limits)

        colors = list(plt.rcParams.get("axes.prop_cycle", {}).by_key().get("color", []))
        if not colors:
            colors = list(matplotlib.colors.TABLEAU_COLORS.values())

        for idx, (label, plotter) in enumerate(self.plotters):
            color = colors[idx % len(colors)]
            trajectory = np.array(plotter.trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], ".-", color=color, label=label)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color=color,
                       edgecolors="black", linewidths=0.8, s=70, marker="o")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color,
                       linewidths=1.6, s=90, marker="x")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True)

        label = "Log(NormalizedObjectiveValue)" if ref.should_normalize else "Log(ObjectiveValue)"
        fig.colorbar(contour, ax=ax, label=label)
        if save_path:
            fig.savefig(save_path)
        plt.close(fig)


# --- Hyperparameter search ---

_HPARAM_SPACE = {
    "lr": optuna.distributions.FloatDistribution(1e-7, 100, log=True),
    "1mbeta1": optuna.distributions.FloatDistribution(1e-5, 1, log=True),
    "1mbeta2": optuna.distributions.FloatDistribution(1e-7, 1, log=True),
    "1mshampoo_beta": optuna.distributions.FloatDistribution(1e-7, 1, log=True),
}

_HPARAM_NAMES = tuple(_HPARAM_SPACE.keys())


class Objective:
    def __init__(self, model, opt, steps, group, data, loss_fn, win_condition,
                 weight_decay, warmup_trials, eval_callback,
                 ema_index: int = 0, device: torch.device | str = "cuda",
                 dtype: torch.dtype | None = None, **kwargs):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = self._move(model)
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.RNNBase):
                mod.flatten_parameters()
        self.opt = opt
        self.steps = steps
        self.group = group
        self.data = data
        self.loss_fn = loss_fn
        self._win_condition = win_condition
        self.weight_decay = weight_decay
        self.warmup_trials = int(warmup_trials)
        self.kwargs = kwargs
        self.ema_index = ema_index
        self.eval_callback = eval_callback
        self.hyperparam_log_history: list[np.ndarray] = []
        self.callback_results = []

        self.validator = Validator(
            {1024: -1e-4, 256: -1e-3, 64: -0.01, 16: -0.1, 8: -0.33, 4: -0.5, 2: -0.75, 1: -0.99},
            {100: 2, 1600: 1},
            steps,
            device=self.device,
        )
        self.m = None
        self.attempt = 0
        self.best_loss = None
        self.best_at = 0
        self.avg = None
        self.did_win = False
        self.set_precond_init_scale = False
        self.end_time = int(os.environ.get("HEAVYBALL_BENCHMARK_TIMEOUT", 3600 * 4)) + time.time()
        self._last_loss = None
        self.disable_early_stopping = False

    def _move(self, module):
        if not hasattr(module, "to"):
            return module
        if self.dtype is not None:
            return module.to(device=self.device, dtype=self.dtype)
        return module.to(device=self.device)

    def win_condition(self, loss=None):
        if loss is not None:
            self._last_loss = loss
        won, _ = self._win_condition(self.m, self._last_loss)
        self.did_win |= won
        return self.did_win

    def _inner(self, params, model_override=None, *,
               allow_validator_stop: bool = True, allow_win_condition: bool = True):
        input_kwargs = locals()
        input_kwargs.pop("self")
        allow_validator_stop = allow_validator_stop and not self.disable_early_stopping
        allow_win_condition = allow_win_condition and not self.disable_early_stopping
        params = tuple(float(p) for p in params)
        params_dict = self._format_params(params)
        model_template = self.model if model_override is None else model_override
        self.m = self._clone_model(model_template)
        o = get_optim(self.opt, self.m.parameters(), **params_dict,
                      weight_decay=self.weight_decay, **self.kwargs)
        validator = self.validator.new()
        loss_buf = torch.empty(self.steps, dtype=torch.float64, device=self.device)
        buf_idx = 0
        last_sync = time.perf_counter()
        n_groups = self.steps // self.group

        for i in range(n_groups):
            if hasattr(o, "train"):
                o.train()
            if self.eval_callback is not None:
                self.callback_results.append(self.eval_callback(self.m))

            for j in range(self.group):
                inp, tgt = self.data()

                def _closure():
                    loss = self.m() if inp is None else self.m(inp)
                    if self.loss_fn is not None:
                        loss = self.loss_fn(loss, tgt)
                    loss.backward()
                    return loss

                try:
                    loss = o.step(_closure)
                except PrecondInitError:
                    self.set_precond_init_scale = True
                    return self._inner(**input_kwargs)

                o.zero_grad()

                with torch.no_grad():
                    loss_buf[buf_idx] = loss.detach()
                    buf_idx += 1

            if hasattr(o, "eval"):
                o.eval()

            now = time.perf_counter()
            if now - last_sync >= 1.0 or i == n_groups - 1:
                with torch.no_grad():
                    pending = loss_buf[:buf_idx]
                    all_finite = torch.isfinite(pending).all()
                    triggered = validator.validate_batch(pending, buf_idx)
                    last_loss = pending[-1]

                    if not all_finite.item():
                        idx = (~torch.isfinite(pending)).nonzero(as_tuple=True)[0][0].item()
                        return validator.ema_states.min().item(), self.m, pending[idx].item()

                    if allow_win_condition:
                        min_loss = pending.min().item()
                        if self.win_condition(min_loss):
                            return validator.ema_states.min().item(), self.m, min_loss

                    if allow_validator_stop and triggered.item():
                        return validator.ema_states.min().item(), self.m, last_loss.item()

                buf_idx = 0
                last_sync = time.perf_counter()

        return validator.ema_states.min().item(), self.m, last_loss.item()

    def objective(self, params):
        self.attempt += 1
        target, _, loss = self._inner(params)
        self.hyperparam_log_history.append(np.log10(np.array(params, dtype=np.float64)))
        if self.best_loss is None or loss < self.best_loss or not np.isfinite(self.best_loss):
            self.best_loss = loss
            self.best_at = self.attempt
            self.avg = np.log(np.array(params))
        if not self.disable_early_stopping:
            if self.best_at * 8 < self.attempt and self.attempt - self.best_at > self.warmup_trials:
                raise Stop
            if time.time() > self.end_time:
                raise Stop
        return target

    def run_study(self, n_trials):
        sampler = AutoSampler(seed=0x123125, search_space=_HPARAM_SPACE)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        winning = {k: float("nan") for k in _HPARAM_NAMES}
        best_value = float("inf")

        def _objective(trial):
            nonlocal best_value
            set_seed(0x12312)
            params = tuple(trial.suggest_float(k, d.low, d.high, log=d.log)
                           for k, d in _HPARAM_SPACE.items())
            result = self.objective(params)
            if result < best_value:
                best_value = result
                winning.update(dict(zip(_HPARAM_NAMES, params)))
            if self.win_condition():
                raise WinConditionMet
            return result

        stdout, sys.stdout = sys.stdout, sys.stderr
        try:
            set_seed()
            study.optimize(_objective, n_trials=n_trials)
        except WinConditionMet:
            pass
        finally:
            sys.stdout = stdout

        return winning

    def _format_params(self, params: tuple) -> dict:
        d = {
            "lr": params[0],
            "betas": (1 - params[1], 1 - params[2]),
            "beta": 1 - params[1],
            "shampoo_beta": 1 - params[3],
            "sam_step_size": params[3],
            "eps": 1e-8,
            "precond_lr": params[3],
        }
        if self.set_precond_init_scale:
            d["precond_init_scale"] = 0.1
        return d

    def _clone_model(self, model_template):
        cloned = copy.deepcopy(model_template)
        cloned = self._move(cloned)
        if hasattr(cloned, "trajectory"):
            initial = getattr(cloned, "initial", None)
            param_tensor = getattr(cloned, "param", None)
            if initial is not None:
                cloned.trajectory = [initial.detach().cpu().numpy()]
            elif param_tensor is not None:
                cloned.trajectory = [param_tensor.detach().cpu().numpy()]
            else:
                cloned.trajectory = []
        return cloned

    def _replay_params(self, params, model_override=None, *, callback_period: int | None = None):
        params = tuple(float(p) for p in params)
        params_dict = self._format_params(params)
        model_template = self.model if model_override is None else model_override
        replay_model = self._clone_model(model_template)
        optimizer = get_optim(self.opt, replay_model.parameters(), **params_dict,
                              weight_decay=self.weight_decay, **self.kwargs)
        period = callback_period if callback_period is not None else max(1, self.group)

        for step_idx in range(self.steps):
            if hasattr(optimizer, "train"):
                optimizer.train()
            if self.eval_callback is not None and step_idx % period == 0:
                self.callback_results.append(self.eval_callback(replay_model))

            inp, tgt = self.data()

            def _closure():
                loss = replay_model() if inp is None else replay_model(inp)
                if self.loss_fn is not None:
                    loss = self.loss_fn(loss, tgt)
                loss.backward()
                return loss

            try:
                optimizer.step(_closure)
            except PrecondInitError:
                self.set_precond_init_scale = True
                params_dict = self._format_params(params)
                optimizer = get_optim(self.opt, replay_model.parameters(), **params_dict,
                                      weight_decay=self.weight_decay, **self.kwargs)
                optimizer.step(_closure)

            optimizer.zero_grad()
            if hasattr(optimizer, "eval"):
                optimizer.eval()

        return replay_model

    def evaluate_params(self, params, model_override=None):
        params = tuple(float(p) for p in params)
        saved = self.group
        try:
            self.group = 1
            return self._replay_params(params, model_override=model_override, callback_period=saved)
        finally:
            self.group = saved

    def compute_hparam_log_ema(self, beta: float = 0.9):
        if not 0 <= beta < 1:
            raise ValueError("EMA beta must be in [0, 1).")
        if not self.hyperparam_log_history:
            return None
        ema = np.zeros_like(self.hyperparam_log_history[0])
        for t, log_vals in enumerate(self.hyperparam_log_history, 1):
            ema = beta * ema + (1 - beta) * log_vals
        return ema / (1 - beta ** t)

    def get_best(self):
        saved = self.group
        try:
            self.group = 1
            return self._replay_params(tuple(np.exp(self.avg)), callback_period=saved)
        finally:
            self.group = saved


# --- Win conditions ---

def disabled_win_condition(*_args, **_kwargs):
    return False, {}


def loss_win_condition(target):
    def win(_model, loss: float):
        return loss <= target, {}
    return win


def param_norm_win_condition(target, offset):
    def win(model, loss):
        with torch.no_grad():
            return model.param.add(offset).square().mean().sqrt().item() < target, {}
    return win


def param0_win_condition(target):
    def win(model, loss):
        with torch.no_grad():
            return model.param[0].item() < target, {}
    return win


# --- Utilities ---

def set_seed(seed: int = 0x1239121):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cleanup():
    gc.enable()
    gc.collect()
    gc.disable()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _none_data():
    return None, None


def evaluate_test_accuracy(test_loader):
    def _fn(model):
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(dev), target.to(dev)
                output = model(data)
                if output.dim() > 2:
                    pred = output.argmax(dim=-1).view(-1)
                    target = target.view(-1)
                else:
                    pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.numel()
        model.train(was_training)
        return correct / total
    return _fn


# --- Main entry point ---

_SUPPRESS = [
    "logei_candidates_func is experimental",
    "BoTorchSampler is experimental",
    "It will be set to log2(param_count). This requires `params` to be of type list.",
    "rank was set to",
    "The given NumPy array is not writable",
]


def trial(
    model, data, loss_fn, win_condition, steps, opt, weight_decay,
    trials=10, group=256, return_best: bool = False,
    warmup_trial_pct: float = 1, random_trials: int = 10,
    eval_callback: Optional[Callable] = None, ema_beta: float = 0.9,
    *, device: torch.device | str = "cuda", dtype: torch.dtype | str | None = None,
):
    if data is None:
        data = _none_data
    group = min(group, steps)
    heavyball.utils.set_torch(einsum_strategy="heavyball")
    if not 0 <= ema_beta < 1:
        raise ValueError("ema_beta must be in [0, 1).")

    opt_list = opt if isinstance(opt, list) else [opt]
    if not opt_list:
        raise ValueError("At least one optimizer must be provided.")

    target_device = torch.device(device)
    target_dtype = resolve_dtype(dtype)

    def _prepare(module):
        if not hasattr(module, "to"):
            return module
        return module.to(device=target_device, dtype=target_dtype) if target_dtype else module.to(target_device)

    model = _prepare(model)
    plotter_template = model if isinstance(model, Plotter) else None
    base_template = _prepare(copy.deepcopy(plotter_template.objective if plotter_template else model))

    for msg in _SUPPRESS:
        heavyball.utils._ignore_warning(msg)

    baseline_cautioning = getattr(heavyball.utils, "_compilable_cautioning", None)
    results = []

    for opt_entry in opt_list:
        cleanup()
        set_seed()

        if isinstance(opt_entry, str):
            opt_label = opt_entry
            opt_name, extra_kwargs = parse_opt_name(opt_entry)
            unscale = extra_kwargs.pop("_unscale_caution", False)
            if unscale:
                heavyball.utils.disable_caution_scaling()
            opt_callable = getattr(heavyball, opt_name)
        else:
            opt_label = getattr(opt_entry, "__name__", repr(opt_entry))
            opt_callable = opt_entry
            extra_kwargs = {"caution": False, "mars": False}
            unscale = False

        obj = Objective(
            copy.deepcopy(base_template), opt_callable, steps, group,
            data, loss_fn, win_condition, weight_decay,
            max(trials * warmup_trial_pct, 1 + random_trials),
            eval_callback, device=target_device, dtype=target_dtype, **extra_kwargs,
        )

        if target_device.type == "cuda":
            torch.cuda.synchronize()

        set_seed()
        t0 = time.time()
        try:
            winning = obj.run_study(trials)
        finally:
            if unscale and baseline_cautioning is not None:
                heavyball.utils._compilable_cautioning = baseline_cautioning

        if target_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        ema_logs = obj.compute_hparam_log_ema(beta=ema_beta)
        ema_params = np.power(10.0, ema_logs) if ema_logs is not None else None

        _print_result(opt_label, elapsed, obj, winning, ema_beta, ema_params)

        if return_best:
            if ema_params is None:
                model_result = obj.get_best()
            else:
                model_result = obj.evaluate_params(
                    ema_params, model_override=plotter_template if plotter_template else None)
            if isinstance(model_result, Plotter):
                print(f"[{opt_label}] Replay trajectory length: {len(model_result.trajectory)}")
            results.append({"label": opt_label, "model": model_result,
                            "best_loss": obj.best_loss, "callback_results": obj.callback_results or None})

    if not return_best:
        return
    models = [(r["label"], r["model"]) for r in results if r["model"] is not None]
    if not models:
        return None
    if len(models) == 1:
        return models[0][1]
    if all(isinstance(m, Plotter) for _, m in models):
        return MultiPlotter(models)
    return models[0][1]


def _print_result(opt_label, elapsed, obj, winning, ema_beta, ema_params):
    fmt = lambda v: f"{v:.5f}" if np.isfinite(v) else "nan"
    w = winning
    parts = [
        f"[{opt_label}] Took: {elapsed} | Attempt: {obj.attempt}",
        f"| {opt_label}(lr={fmt(w['lr'])}, betas=({1 - w['1mbeta1']:.3f}, {1 - w['1mbeta2']:.4f}), "
        f"shampoo_beta={1 - w['1mshampoo_beta']:.3f})",
    ]
    if ema_params is not None:
        e = ema_params
        parts.append(
            f"| EMA\u03b2={ema_beta:.2f}(lr={e[0]:.5f}, betas=({1 - e[1]:.3f}, {1 - e[2]:.4f}), "
            f"shampoo_beta={1 - e[3]:.3f})"
        )
    parts.append(f"| Best Loss: {obj.best_loss}")
    if obj.callback_results:
        parts.append(f"| Callback Results: {obj.callback_results}")
    if obj.did_win:
        parts.append("| Win: Yes")
    print(" ".join(parts))
