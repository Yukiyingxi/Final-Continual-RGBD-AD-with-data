"""Replay-free joint CFNS + DSNS consolidation.

This module does *not* apply two projections in sequence.  It computes the
orthogonal projection of an optimiser candidate onto the intersection of:

1. Cumulative Functional Null-Space (CFNS) constraints built from old input
   activations and old output-error directions; and
2. Dual-sided Null-Space (DSNS) constraints built from old input activations
   and old output-activation directions.

For a stage-anchored update ``D = Wcandidate - Wanchor`` the joint projection
is

    DF = No_f D Ni_f
    D_joint = DF - Bo (Bo.T DF Bi) Bi.T

where ``Bo = orth(No_f Uo_dsns)`` and ``Bi = orth(Ni_f Ui_dsns)``.  This is a
single closed-form projection onto the intersection.  It therefore satisfies
the CFNS left/right null constraints and the DSNS protected block constraint
simultaneously, without order-dependent alternating projections.

Only compact aggregate SVD sketches are retained; no samples or per-example
features are stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn as nn


@dataclass
class _Sketch:
    basis: torch.Tensor
    singular: torch.Tensor
    count: int


def _empty_sketch(dim: int) -> _Sketch:
    return _Sketch(torch.empty(dim, 0), torch.empty(0), 0)


def _finite_rows(value: torch.Tensor, dim: int, normalise: bool = False) -> torch.Tensor:
    rows = value.detach().float().reshape(-1, dim)
    rows = rows[torch.isfinite(rows).all(dim=1)]
    if normalise and rows.numel():
        rows = rows[rows.norm(dim=1) > 1e-12]
        if rows.numel():
            rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return rows


def _compact_svd(
    previous: _Sketch,
    new_rows: torch.Tensor,
    energy: float,
    max_rank: int,
) -> _Sketch:
    if new_rows.numel() == 0:
        return previous
    new_count = int(new_rows.shape[0])
    rows = new_rows.cpu().float()
    if previous.basis.numel():
        rows = torch.cat((previous.singular[:, None] * previous.basis.T, rows), dim=0)
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    power = singular.square()
    if float(power.sum()) <= 0.0:
        rank = 0
    else:
        cumulative = power.cumsum(0) / power.sum()
        threshold = torch.tensor(float(energy), dtype=cumulative.dtype)
        rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
        rank = min(rank, int(max_rank), int(vh.shape[0]))
    return _Sketch(
        basis=vh[:rank].T.contiguous(),
        singular=singular[:rank].contiguous(),
        count=previous.count + new_count,
    )


def _residual_basis(protected: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Return orth(candidate directions outside span(protected))."""
    if candidate.numel() == 0:
        return candidate[:, :0]
    residual = candidate
    if protected.numel():
        residual = residual - protected @ (protected.T @ residual)
    if residual.numel() == 0:
        return candidate[:, :0]
    u, singular, _ = torch.linalg.svd(residual.float(), full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 0.0:
        return candidate[:, :0]
    keep = singular > singular[0] * 1e-6
    return u[:, keep].to(device=candidate.device, dtype=candidate.dtype).contiguous()


class JointCFNSDSNS:
    """Collect aggregate bases and hard-project cumulative updates jointly."""

    def __init__(
        self,
        model: nn.Module,
        cfns_energy: float = 0.995,
        cfns_rank: int = 128,
        cfns_rows: int = 512,
        dsns_energy: float = 0.97,
        dsns_rank: int = 32,
        dsns_rows: int = 128,
        rows_per_call: int = 32,
        seed: int = 1,
        exclude_prefixes: tuple[str, ...] = ("encoder.", "classifier."),
    ) -> None:
        for name, energy in (("cfns_energy", cfns_energy), ("dsns_energy", dsns_energy)):
            if not 0.0 < energy <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self.model = model
        self.cfns_energy = float(cfns_energy)
        self.cfns_rank = int(cfns_rank)
        self.cfns_rows = int(cfns_rows)
        self.dsns_energy = float(dsns_energy)
        self.dsns_rank = int(dsns_rank)
        self.dsns_rows = int(dsns_rows)
        self.rows_per_call = int(rows_per_call)
        self.seed = int(seed)
        self.exclude_prefixes = tuple(exclude_prefixes)
        self.modules: Dict[str, nn.Linear] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and module.weight.requires_grad
            and not any(name == prefix[:-1] or name.startswith(prefix) for prefix in self.exclude_prefixes)
        }
        self.cfns_input = {name: _empty_sketch(m.in_features) for name, m in self.modules.items()}
        self.cfns_error = {name: _empty_sketch(m.out_features) for name, m in self.modules.items()}
        self.dsns_input = {name: _empty_sketch(m.in_features) for name, m in self.modules.items()}
        self.dsns_output = {name: _empty_sketch(m.out_features) for name, m in self.modules.items()}
        self._buffers: Dict[str, Dict[str, list[torch.Tensor]]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._anchors: Dict[str, torch.Tensor] = {}
        self._device_bases: Dict[str, tuple[torch.Tensor, ...]] = {}
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    @property
    def layer_count(self) -> int:
        return len(self.modules)

    def _sample(self, rows: torch.Tensor, remaining: int) -> torch.Tensor:
        take = min(int(rows.shape[0]), self.rows_per_call, int(remaining))
        if take <= 0:
            return rows[:0].cpu()
        if rows.shape[0] == take:
            return rows.cpu()
        indices = torch.randperm(rows.shape[0], generator=self._generator)[:take]
        return rows[indices.to(rows.device)].cpu()

    def begin_collection(self) -> None:
        if self._handles:
            raise RuntimeError("collection is already active")
        self._buffers = {name: {"in": [], "out": [], "err": []} for name in self.modules}
        for name, module in self.modules.items():
            def hook(mod, inputs, output, layer_name=name):
                if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                    return
                bucket = self._buffers[layer_name]
                input_cap = max(self.cfns_rows, self.dsns_rows)
                used_in = sum(item.shape[0] for item in bucket["in"])
                used_out = sum(item.shape[0] for item in bucket["out"])
                if used_in < input_cap:
                    rows = _finite_rows(inputs[0], mod.in_features)
                    bucket["in"].append(self._sample(rows, input_cap - used_in))
                if used_out < self.dsns_rows:
                    rows = _finite_rows(output, mod.out_features)
                    bucket["out"].append(self._sample(rows, self.dsns_rows - used_out))
                if output.requires_grad:
                    def capture_error(gradient, captured_name=layer_name, out_dim=mod.out_features):
                        captured = self._buffers.get(captured_name)
                        if captured is None:
                            return gradient
                        used = sum(item.shape[0] for item in captured["err"])
                        if used < self.cfns_rows:
                            rows = _finite_rows(gradient, out_dim, normalise=True)
                            captured["err"].append(self._sample(rows, self.cfns_rows - used))
                        return gradient
                    output.register_hook(capture_error)
            self._handles.append(module.register_forward_hook(hook))

    def end_collection(self) -> dict:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        summary = {}
        for name, module in self.modules.items():
            bucket = self._buffers[name]
            input_rows = torch.cat(bucket["in"], 0) if bucket["in"] else torch.empty(0, module.in_features)
            output_rows = torch.cat(bucket["out"], 0) if bucket["out"] else torch.empty(0, module.out_features)
            error_rows = torch.cat(bucket["err"], 0) if bucket["err"] else torch.empty(0, module.out_features)
            self.cfns_input[name] = _compact_svd(
                self.cfns_input[name], input_rows[: self.cfns_rows], self.cfns_energy, self.cfns_rank
            )
            self.cfns_error[name] = _compact_svd(
                self.cfns_error[name], error_rows[: self.cfns_rows], self.cfns_energy, self.cfns_rank
            )
            self.dsns_input[name] = _compact_svd(
                self.dsns_input[name], input_rows[: self.dsns_rows], self.dsns_energy, self.dsns_rank
            )
            self.dsns_output[name] = _compact_svd(
                self.dsns_output[name], output_rows[: self.dsns_rows], self.dsns_energy, self.dsns_rank
            )
            summary[name] = {
                "cfns_input_rank": int(self.cfns_input[name].basis.shape[1]),
                "cfns_error_rank": int(self.cfns_error[name].basis.shape[1]),
                "dsns_input_rank": int(self.dsns_input[name].basis.shape[1]),
                "dsns_output_rank": int(self.dsns_output[name].basis.shape[1]),
                "input_rows": int(input_rows.shape[0]),
                "output_rows": int(output_rows.shape[0]),
                "error_rows": int(error_rows.shape[0]),
            }
        self._buffers = {}
        self._device_bases.clear()
        return summary

    @torch.no_grad()
    def set_stage_anchor(self) -> None:
        self._anchors = {name: module.weight.detach().clone() for name, module in self.modules.items()}
        self._device_bases = {}
        for name, module in self.modules.items():
            device, dtype = module.weight.device, module.weight.dtype
            uif = self.cfns_input[name].basis.to(device=device, dtype=dtype)
            uof = self.cfns_error[name].basis.to(device=device, dtype=dtype)
            uid = self.dsns_input[name].basis.to(device=device, dtype=dtype)
            uod = self.dsns_output[name].basis.to(device=device, dtype=dtype)
            bi = _residual_basis(uif, uid)
            bo = _residual_basis(uof, uod)
            self._device_bases[name] = (uif, uof, uid, uod, bi, bo)

    @staticmethod
    def project_delta(
        delta: torch.Tensor,
        uif: torch.Tensor,
        uof: torch.Tensor,
        bi: torch.Tensor,
        bo: torch.Tensor,
    ) -> torch.Tensor:
        projected = delta
        if uof.numel():
            projected = projected - uof @ (uof.T @ projected)
        if uif.numel():
            projected = projected - (projected @ uif) @ uif.T
        if bo.numel() and bi.numel():
            projected = projected - bo @ (bo.T @ projected @ bi) @ bi.T
        return projected

    @torch.no_grad()
    def hard_project_parameters(self, compute_diagnostics: bool = False) -> dict:
        if not self._anchors:
            raise RuntimeError("set_stage_anchor() must be called before training")
        diagnostics = {}
        for name, module in self.modules.items():
            anchor = self._anchors[name]
            uif, uof, uid, uod, bi, bo = self._device_bases[name]
            candidate = module.weight - anchor
            projected = self.project_delta(candidate, uif, uof, bi, bo)
            module.weight.copy_(anchor + projected)
            if compute_diagnostics:
                norm = candidate.norm().clamp_min(1e-12)
                left = (uof.T @ projected).norm() if uof.numel() else projected.new_zeros(())
                right = (projected @ uif).norm() if uif.numel() else projected.new_zeros(())
                dsns = (
                    (uod.T @ projected @ uid).norm()
                    if uod.numel() and uid.numel()
                    else projected.new_zeros(())
                )
                diagnostics[name] = {
                    "cfns_left_residual": float(left / norm),
                    "cfns_right_residual": float(right / norm),
                    "dsns_block_residual": float(dsns / norm),
                    "removed_fraction": float((candidate - projected).norm() / norm),
                    "cfns_input_rank": int(uif.shape[1]),
                    "cfns_error_rank": int(uof.shape[1]),
                    "dsns_input_rank": int(uid.shape[1]),
                    "dsns_output_rank": int(uod.shape[1]),
                    "joint_input_rank": int(bi.shape[1]),
                    "joint_output_rank": int(bo.shape[1]),
                }
        return diagnostics

    def state_dict(self) -> dict:
        def encode(items: Dict[str, _Sketch]) -> dict:
            return {
                name: {"basis": sketch.basis.cpu(), "singular": sketch.singular.cpu(), "count": int(sketch.count)}
                for name, sketch in items.items()
            }
        return {
            "cfns_energy": self.cfns_energy,
            "cfns_rank": self.cfns_rank,
            "cfns_rows": self.cfns_rows,
            "dsns_energy": self.dsns_energy,
            "dsns_rank": self.dsns_rank,
            "dsns_rows": self.dsns_rows,
            "rows_per_call": self.rows_per_call,
            "cfns_input": encode(self.cfns_input),
            "cfns_error": encode(self.cfns_error),
            "dsns_input": encode(self.dsns_input),
            "dsns_output": encode(self.dsns_output),
        }

    def load_state_dict(self, state: dict) -> None:
        def decode(source: dict, target: Dict[str, _Sketch]) -> None:
            for name, value in source.items():
                if name in target:
                    target[name] = _Sketch(
                        value["basis"].cpu().float(),
                        value["singular"].cpu().float(),
                        int(value.get("count", 0)),
                    )
        decode(state.get("cfns_input", {}), self.cfns_input)
        decode(state.get("cfns_error", {}), self.cfns_error)
        decode(state.get("dsns_input", {}), self.dsns_input)
        decode(state.get("dsns_output", {}), self.dsns_output)
        self._device_bases.clear()

    def storage_bytes(self) -> int:
        tensors: Iterable[torch.Tensor] = (
            tensor
            for collection in (self.cfns_input, self.cfns_error, self.dsns_input, self.dsns_output)
            for sketch in collection.values()
            for tensor in (sketch.basis, sketch.singular)
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
