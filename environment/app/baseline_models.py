"""Shared physics and training for the baseline comparison (no training on import)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
# Must be set before the first CUDA BLAS operation for seeded CUDA runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

MODEL_SPECS = {
    "StandardPINN": {"domains": 1, "adaptive": False, "xpinn": False},
    "MDPINN": {"domains": 2, "adaptive": False, "xpinn": False},
    "V5": {"domains": 2, "adaptive": True, "xpinn": False},
    "StandardPINN_V5Controls": {"domains": 1, "adaptive": True, "xpinn": False},
    "XPINN": {"domains": 2, "adaptive": False, "xpinn": True},
}
TERMS = ("pde", "dir", "neu", "inter_T", "inter_flux")
S_BOUNDS = ((-2., 2.), (-4., 0.), (-3., 1.), (-4., 0.), (-2., 2.))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_samples(cfg, seed):
    """Same CPU torch.rand distribution/order as V5; independent of model init."""
    p, s = cfg["physics"], cfg["sampling"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def rand(n):
        return torch.rand(n, 1, generator=gen)
    n, cu, ox, h = s["N_b"], p["R_Cu"], p["R_out"], p["H"]
    out = {}
    def put(key, r, z):
        out[key] = torch.cat([r, z], 1).numpy()
    r = rand(s["N_f_cu"]) * (cu - s["r_min_cu"]) + s["r_min_cu"]
    put("cu", r, rand(s["N_f_cu"]) * h)
    r = rand(s["N_f_ox"]) * (ox - cu) + cu
    put("ox", r, rand(s["N_f_ox"]) * h)
    rtcu, rtox = rand(n) * cu, rand(n) * (ox-cu) + cu
    rbcu, rbox = rand(n) * cu, rand(n) * (ox-cu) + cu
    put("top_cu", rtcu, torch.full((n, 1), h))
    put("top_ox", rtox, torch.full((n, 1), h))
    put("bottom_cu", rbcu, torch.zeros(n, 1))
    put("bottom_ox", rbox, torch.zeros(n, 1))
    put("outer", torch.full((n, 1), ox), rand(n) * h)
    put("axis", torch.zeros(n, 1), rand(n) * h)
    c = s["corner_each"]
    zi = torch.cat([rand(n-2*c)*h, rand(c)*s["corner_depth"],
                    h-rand(c)*s["corner_depth"]], 0)
    put("interface", torch.full((n, 1), cu), zi)
    return out


def samples_hash(samples):
    digest = hashlib.sha256()
    for key in sorted(samples):
        digest.update(key.encode())
        digest.update(np.asarray(samples[key], dtype="<f4").tobytes())
    return digest.hexdigest()


class TemperatureNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p, n = cfg["physics"], cfg["network"]
        if n["activation"] != "SiLU":
            raise ValueError("This controlled comparison fixes activation to SiLU")
        layers = []
        for index in range(n["hidden_layers"]):
            layers.extend([nn.Linear(2 if index == 0 else n["width"], n["width"]), nn.SiLU()])
        layers.append(nn.Linear(n["width"], 1))
        self.net = nn.Sequential(*layers)
        self.r_out, self.height = p["R_out"], p["H"]
        self.scale = (p["T_top"]-p["T_bottom"]) / 2
        self.offset = (p["T_top"]+p["T_bottom"]) / 2
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Preserve the exact normalization/output scale in the supplied V5.
        normal = torch.cat([x[:, :1]/self.r_out, x[:, 1:2]/self.height], 1)
        return self.net(normal)*self.scale + self.offset


class BaselineModel(nn.Module):
    def __init__(self, name, cfg, seed, device):
        super().__init__()
        self.name, self.cfg, self.spec = name, cfg, MODEL_SPECS[name]
        seed_everything(seed)
        self.cu = TemperatureNet(cfg)
        if self.spec["domains"] == 2:
            self.ox = TemperatureNet(cfg)
        else:
            self.ox = self.cu  # One shared temperature network over both media.
        if self.spec["adaptive"]:
            self.log_vars = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in TERMS})
        else:
            self.log_vars = nn.ParameterDict()
        self.to(device)

    def net_parameters(self):
        # Module.parameters() deduplicates the single-network shared module.
        return list(self.cu.parameters()) + ([] if self.ox is self.cu else list(self.ox.parameters()))

    def predict(self, r, z):
        shape = np.asarray(r).shape
        coords = np.column_stack([np.asarray(r).ravel(), np.asarray(z).ravel()])
        device = next(self.parameters()).device
        result = np.empty(len(coords), dtype=np.float64)
        self.eval()
        with torch.no_grad():
            for start in range(0, len(coords), 8192):
                block = torch.as_tensor(coords[start:start+8192], dtype=torch.float32, device=device)
                mask = block[:, 0] <= self.cfg["physics"]["R_Cu"]
                pred = torch.empty(len(block), device=device)
                if mask.any():
                    pred[mask] = self.cu(block[mask]).flatten()
                if (~mask).any():
                    pred[~mask] = self.ox(block[~mask]).flatten()
                result[start:start+len(block)] = pred.cpu().numpy()
        return result.reshape(shape)


def derivative(value, variable):
    return torch.autograd.grad(value, variable, grad_outputs=torch.ones_like(value), create_graph=True)[0]


def pde_residual(net, r, z, eps):
    t = net(torch.cat([r, z], 1))
    dr, dz = derivative(t, r), derivative(t, z)
    return derivative(dr, r) + derivative(dz, z)


def top_residual(net, r, z, physics):
    temperature = net(torch.cat([r, z], 1))
    return physics["H"] * derivative(temperature, z) + (physics["T_top"] - physics["T_bottom"])


class PhysicsLoss:
    def __init__(self, model, samples):
        self.model = model
        device = next(model.parameters()).device
        self.points = {}
        for key, coords in samples.items():
            self.points[key] = tuple(torch.tensor(coords[:, i:i+1], device=device,
                                                 requires_grad=True) for i in (0, 1))

    def clear_coordinate_grads(self):
        for pair in self.points.values():
            for coord in pair:
                coord.grad = None

    def components(self):
        model, x, p = self.model, self.points, self.model.cfg["physics"]
        a, b = model.cu, model.ox
        fc = pde_residual(a, *x["cu"], p["pde_eps"])
        fo = pde_residual(b, *x["ox"], p["pde_eps"])
        losses = {"pde": fc.square().mean()+fo.square().mean()}
        losses["dir"] = sum((net(torch.cat(x[key], 1))-p["T_bottom"]).square().mean()
                            for net, key in ((a, "bottom_cu"), (b, "bottom_ox")))
        losses["dir"] += sum(top_residual(net, *x[key], p).square().mean()
                             for net, key in ((a, "top_cu"), (b, "top_ox")))
        losses["neu"] = sum(derivative(net(torch.cat(x[key], 1)), x[key][0]).square().mean()
                            for net, key in ((a, "axis"), (b, "outer")))
        r, z = x["interface"]
        ti, tj = a(torch.cat([r, z], 1)), b(torch.cat([r, z], 1))
        if model.spec["xpinn"]:
            avg = (ti+tj)/2
            # Pair-average matching, rescaled to the same temperature-jump weight.
            losses["inter_T"] = 2*((ti-avg).square().mean()+(tj-avg).square().mean())
        else:
            losses["inter_T"] = (ti-tj).square().mean()
        # Same +r derivative on both sides: k_cu*dT_cu/dr = k_ox*dT_ox/dr.
        losses["inter_flux"] = (p["k_Cu"]*derivative(ti, r)-p["k_SiO2"]*derivative(tj, r)).square().mean()
        losses["inter_residual"] = (pde_residual(a, r, z, p["pde_eps"])-
                                    pde_residual(b, r, z, p["pde_eps"])).square().mean() if model.spec["xpinn"] else ti.new_zeros(())
        return losses

    def __call__(self):
        losses = self.components()
        if self.model.spec["adaptive"]:
            total = sum(0.5*(torch.exp(-self.model.log_vars[k])*losses[k] +
                             F.softplus(self.model.log_vars[k])) for k in TERMS).squeeze()
        else:
            total = sum(losses[k] for k in TERMS)
        total = total + self.model.cfg["training"]["xpinn_residual_weight"]*losses["inter_residual"]
        return total, losses


def train_model(model, samples, output, seed, metadata):
    """15000 Adam updates + 3000 L-BFGS outer calls by default; no early stopping."""
    cfg, output = model.cfg, Path(output)
    tr = cfg["training"]
    physics = PhysicsLoss(model, samples)
    nets, params = model.net_parameters(), list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=tr["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tr["scheduler_factor"],
        patience=tr["scheduler_patience"], min_lr=tr["min_lr"])
    history, evaluations = [], 0
    start = time.perf_counter()
    initial_s = {k: 0. for k in TERMS}

    def evaluate(backward=False):
        nonlocal evaluations
        physics.clear_coordinate_grads()
        total, terms = physics()
        evaluations += 1
        if not torch.isfinite(total):
            raise FloatingPointError(f"{model.name}: non-finite loss at evaluation {evaluations}")
        if backward:
            total.backward()
        return total, terms

    def record(stage, outer_step, total, terms, current_lr):
        row = {"iteration": len(history)+1, "stage": stage, "stage_step": outer_step,
               "objective": float(total.detach()),
               "physics_loss": float(sum(terms[k] for k in TERMS).detach()),
               "lr": current_lr, "loss_evaluations": evaluations}
        row.update({"loss_"+k: float(v.detach()) for k, v in terms.items()})
        row.update({"s_"+k: float(model.log_vars[k].detach()) if k in model.log_vars else initial_s[k]
                    for k in TERMS})
        history.append(row)
        if outer_step == 1 or outer_step % tr["log_every"] == 0:
            print(f"[{model.name}][{stage} {outer_step}] objective={row['objective']:.5e} "
                  f"physics={row['physics_loss']:.5e}", flush=True)

    def checkpoint(stage, step, optimizer_now):
        torch.save({"model_name": model.name, "seed": seed, "config": cfg,
                    "model_state": model.state_dict(), "stage": stage, "stage_step": step,
                    "optimizer_state": optimizer_now.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "history": history,
                    "loss_evaluations": evaluations, "metadata": metadata}, output/"checkpoint_latest.pt")

    model.train()
    for step in range(1, tr["adam_steps"]+1):
        optimizer.zero_grad(set_to_none=True)
        total, terms = evaluate(backward=True)
        # Record the exact objective/terms that produced this update (pre-update s).
        record("Adam", step, total, terms, optimizer.param_groups[0]["lr"])
        torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step(total.detach())
        with torch.no_grad():
            for key, bounds in zip(TERMS, S_BOUNDS):
                if key in model.log_vars:
                    model.log_vars[key].clamp_(*bounds)
        if step % tr["checkpoint_every"] == 0 or step == tr["adam_steps"]:
            checkpoint("Adam", step, optimizer)

    # V5 L-BFGS optimizes network parameters only; freeze adaptive weights explicitly.
    for value in model.log_vars.values():
        value.requires_grad_(False)
    lbfgs = torch.optim.LBFGS(nets, lr=tr["lbfgs_lr"], max_iter=tr["lbfgs_max_iter"],
                             max_eval=tr["lbfgs_max_eval"], history_size=tr["lbfgs_history_size"],
                             tolerance_grad=tr["tolerance_grad"],
                             tolerance_change=tr["tolerance_change"], line_search_fn="strong_wolfe")
    for step in range(1, tr["lbfgs_steps"]+1):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            return evaluate(backward=True)[0]
        lbfgs.step(closure)
        # torch LBFGS.step returns its initial objective. Re-evaluate at accepted
        # parameters so the total, all components, and saved state agree.
        total, terms = evaluate()
        record("L-BFGS", step, total, terms, tr["lbfgs_lr"])
        if step % tr["checkpoint_every"] == 0 or step == tr["lbfgs_steps"]:
            checkpoint("L-BFGS", step, lbfgs)
    if len(history) != tr["adam_steps"]+tr["lbfgs_steps"]:
        raise RuntimeError("Training record length does not match configured budget")
    elapsed = time.perf_counter()-start
    torch.save({"model_name": model.name, "seed": seed, "config": copy.deepcopy(cfg),
                "model_state": model.state_dict(), "metadata": metadata}, output/"model_final.pt")
    return history, elapsed, evaluations
