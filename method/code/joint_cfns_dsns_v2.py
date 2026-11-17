"""Numerically hardened joint CFNS+DSNS projector."""

from __future__ import annotations

import torch

from joint_cfns_dsns import JointCFNSDSNS as _JointCFNSDSNS


def _orthonormal_basis(candidate: torch.Tensor) -> torch.Tensor:
    if candidate.numel() == 0:
        return candidate[:, :0]
    u, singular, _ = torch.linalg.svd(candidate.float(), full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 0.0:
        return candidate[:, :0]
    keep = singular > singular[0] * 1e-6
    return u[:, keep].to(device=candidate.device, dtype=candidate.dtype).contiguous()


def _residual_basis(protected: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if candidate.numel() == 0:
        return candidate[:, :0]
    residual = candidate
    if protected.numel():
        residual = residual - protected @ (protected.T @ residual)
        residual = residual - protected @ (protected.T @ residual)
    basis = _orthonormal_basis(residual)
    if protected.numel() and basis.numel():
        basis = basis - protected @ (protected.T @ basis)
        basis = basis - protected @ (protected.T @ basis)
        basis = _orthonormal_basis(basis)
    return basis


class JointCFNSDSNS(_JointCFNSDSNS):
    @torch.no_grad()
    def set_stage_anchor(self) -> None:
        self._anchors = {name: module.weight.detach().clone() for name, module in self.modules.items()}
        self._device_bases = {}
        for name, module in self.modules.items():
            device, dtype = module.weight.device, module.weight.dtype
            uif = _orthonormal_basis(self.cfns_input[name].basis.to(device=device, dtype=dtype))
            uof = _orthonormal_basis(self.cfns_error[name].basis.to(device=device, dtype=dtype))
            uid = _orthonormal_basis(self.dsns_input[name].basis.to(device=device, dtype=dtype))
            uod = _orthonormal_basis(self.dsns_output[name].basis.to(device=device, dtype=dtype))
            bi = _residual_basis(uif, uid)
            bo = _residual_basis(uof, uod)
            self._device_bases[name] = (uif, uof, uid, uod, bi, bo)

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
                    "cfns_left_absolute": float(left),
                    "cfns_right_absolute": float(right),
                    "dsns_block_absolute": float(dsns),
                    "candidate_norm": float(candidate.norm()),
                    "projected_norm": float(projected.norm()),
                    "removed_fraction": float((candidate - projected).norm() / norm),
                    "cfns_input_rank": int(uif.shape[1]),
                    "cfns_error_rank": int(uof.shape[1]),
                    "dsns_input_rank": int(uid.shape[1]),
                    "dsns_output_rank": int(uod.shape[1]),
                    "joint_input_rank": int(bi.shape[1]),
                    "joint_output_rank": int(bo.shape[1]),
                }
        return diagnostics
