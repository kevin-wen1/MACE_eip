###########################################################################################
# Implementation of Evidential Deep Learning loss functions for EIP (Evidential Interatomic Potential)
# Based on: "Evidential deep learning for interatomic potentials" (Nature Communications, 2025)
# This module implements the complete EIP framework with NIG (Normal-Inverse-Gamma) priors
###########################################################################################

from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from mace.modules.loss import reduce_loss


def tilted_loss(error: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
    """
    Tilted loss (quantile loss) for quantile regression.
    
    ρ_q(e) = max(q*e, (q-1)*e)
    
    Args:
        error: Prediction error [...]
        quantile: Quantile parameter (default 0.5 for median)
    
    Returns:
        Tilted loss values
    """
    return torch.where(error >= 0, quantile * error, (quantile - 1) * error)


def evidential_nll_loss(
    pred_gamma: torch.Tensor,
    pred_nu: torch.Tensor,
    pred_alpha: torch.Tensor,
    pred_beta: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Evidential NLL loss for quantile regression (EIP Equation 10).
    
    This implements the negative log-likelihood of the Normal-Inverse-Gamma distribution
    with quantile regression via asymmetric Laplace distribution.
    
    Args:
        pred_gamma: Predicted mean (γ) [n_atoms, 3]
        pred_nu: Nu parameter (ν) [n_atoms, 3]
        pred_alpha: Alpha parameter (α) [n_atoms, 3]
        pred_beta: Beta parameter (β) [n_atoms, 3]
        target: True values [n_atoms, 3]
        quantile: Quantile for regression (default 0.5 for median)
    
    Returns:
        NLL loss per component
    """
    # Compute auxiliary variables for quantile regression
    tau = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    omega = 2.0 / (quantile * (1.0 - quantile))
    z = pred_beta / (pred_alpha - 1.0 + 1e-10)
    
    # Compute Omega
    Omega = 4.0 * pred_beta * (1.0 + omega * z * pred_nu)
    
    # Compute error
    error = target - (pred_gamma + tau * z)
    
    # NLL formula (Equation 10)
    nll = (
        0.5 * torch.log(torch.tensor(torch.pi, device=pred_gamma.device) / (pred_nu + 1e-10))
        - pred_alpha * torch.log(Omega + 1e-10)
        + (pred_alpha + 0.5) * torch.log(error ** 2 * pred_nu + Omega + 1e-10)
        + torch.lgamma(pred_alpha + 1e-10)
        - torch.lgamma(pred_alpha + 0.5 + 1e-10)
    )
    
    return nll


def evidential_regularizer(
    pred_gamma: torch.Tensor,
    pred_nu: torch.Tensor,
    pred_alpha: torch.Tensor,
    pred_beta: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Evidence regularizer (EIP Equation 11).
    
    This regularizer penalizes high confidence when predictions are wrong,
    preventing the model from being overconfident.
    
    L_R = ρ_q(target - gamma) * Φ
    where Φ = 2ν + α + 1/β (confidence)
    
    Args:
        pred_gamma: Predicted mean (γ) [n_atoms, 3]
        pred_nu: Nu parameter (ν) [n_atoms, 3]
        pred_alpha: Alpha parameter (α) [n_atoms, 3]
        pred_beta: Beta parameter (β) [n_atoms, 3]
        target: True values [n_atoms, 3]
        quantile: Quantile for regression (default 0.5 for median)
    
    Returns:
        Regularization loss per component
    """
    # Tilted loss on prediction error
    error = target - pred_gamma
    rho_q = tilted_loss(error, quantile)
    
    # Confidence (higher when model is more certain)
    confidence = 2.0 * pred_nu + pred_alpha + 1.0 / (pred_beta + 1e-10)
    
    return rho_q * confidence


def evidential_forces_loss(
    ref: Batch, 
    pred: TensorDict, 
    ddp: Optional[bool] = None,
    quantile: float = 0.5,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """
    Evidential loss for forces using complete EIP formulation.
    
    L = L_NLL + λ * L_R
    
    where:
    - L_NLL: Negative log-likelihood (Equation 10)
    - L_R: Evidence regularizer (Equation 11)
    - λ: Regularization coefficient
    
    Args:
        ref: Reference batch with true forces
        pred: Predictions with NIG parameters
        ddp: Distributed data parallel flag
        quantile: Quantile for regression (default 0.5)
        lambda_reg: Regularization coefficient (default 0.01)
    
    Returns:
        Total evidential loss for forces
    """
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    
    forces_ref = ref["forces"]
    forces_pred = pred["forces"]
    
    # Check if NIG parameters are available
    if (
        "node_nu" in pred and pred["node_nu"] is not None
        and "node_alpha" in pred and pred["node_alpha"] is not None
        and "node_beta" in pred and pred["node_beta"] is not None
    ):
        # EIP: Use NIG parameters to compute evidential loss
        # Note: gamma is the force prediction itself
        pred_gamma = forces_pred
        
        # Expand node-level parameters to force dimensions [n_atoms, 3]
        pred_nu = pred["node_nu"].unsqueeze(-1).expand(-1, 3)
        pred_alpha = pred["node_alpha"].unsqueeze(-1).expand(-1, 3)
        pred_beta = pred["node_beta"].unsqueeze(-1).expand(-1, 3)
        
        # Compute NLL loss (Equation 10)
        nll = evidential_nll_loss(
            pred_gamma, pred_nu, pred_alpha, pred_beta, forces_ref, quantile
        )
        
        # Compute Evidence Regularizer (Equation 11)
        reg = evidential_regularizer(
            pred_gamma, pred_nu, pred_alpha, pred_beta, forces_ref, quantile
        )
        
        # Total loss: NLL + λ * Regularizer
        raw_loss = configs_weight * configs_forces_weight * (nll + lambda_reg * reg)
    else:
        # Fallback: Standard MSE if NIG parameters not available
        from mace.modules.loss import mean_squared_error_forces
        return mean_squared_error_forces(ref, pred, ddp)
    
    return reduce_loss(raw_loss, ddp)


def evidential_energy_loss(
    ref: Batch, 
    pred: TensorDict, 
    ddp: Optional[bool] = None,
    quantile: float = 0.5,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """
    Evidential loss for energy using complete EIP formulation.
    
    Similar to forces loss but for system-level energy predictions.
    
    Args:
        ref: Reference batch with true energies
        pred: Predictions with uncertainty
        ddp: Distributed data parallel flag
        quantile: Quantile for regression (default 0.5)
        lambda_reg: Regularization coefficient (default 0.01)
    
    Returns:
        Total evidential loss for energy
    """
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    energy_pred = pred["energy"]
    energy_ref = ref["energy"]
    
    if "energy_uncertainty" in pred and pred["energy_uncertainty"] is not None:
        # For energy, we use a simplified loss based on uncertainty
        # This is because energy is a system-level property aggregated from atoms
        energy_std = pred["energy_uncertainty"]
        energy_std = torch.clamp(energy_std, min=1e-6)
        energy_var = energy_std ** 2
        
        # Per-atom energy for fair comparison
        energy_pred_per_atom = energy_pred / num_atoms
        energy_ref_per_atom = energy_ref / num_atoms
        
        squared_error = torch.square(energy_ref_per_atom - energy_pred_per_atom)
        weighted_error = squared_error / (2.0 * energy_var)
        log_var_term = 0.5 * torch.log(energy_var)
        
        # Add regularization term
        error = energy_ref_per_atom - energy_pred_per_atom
        reg_term = torch.abs(error) / (energy_std + 1e-10)
        
        raw_loss = ref.weight * ref.energy_weight * (weighted_error + log_var_term + lambda_reg * reg_term)
    else:
        # Fallback to standard weighted MSE
        raw_loss = (
            ref.weight
            * ref.energy_weight
            * torch.square((energy_ref - energy_pred) / num_atoms)
        )
    
    return reduce_loss(raw_loss, ddp)


def uncertainty_weighted_stress_loss(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """
    Standard MSE loss for stress (no uncertainty weighting).
    
    Stress is a system-level property computed from forces, so it doesn't have
    its own uncertainty estimate. We use standard MSE loss for stress.
    """
    from mace.modules.loss import weighted_mean_squared_stress
    return weighted_mean_squared_stress(ref, pred, ddp)


class UncertaintyWeightedEnergyForcesLoss(torch.nn.Module):
    """
    Combined energy and force loss with EIP evidential uncertainty.
    
    This loss function implements the complete EIP framework where:
    - Energy and force predictions include NIG uncertainty parameters
    - Loss combines NLL with evidence regularizer
    - The loss automatically balances based on prediction confidence
    """
    
    def __init__(
        self, 
        energy_weight: float = 1.0, 
        forces_weight: float = 1.0,
        use_uncertainty: bool = True,
        quantile: float = 0.5,
        lambda_reg: float = 0.01,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.use_uncertainty = use_uncertainty
        self.quantile = quantile
        self.lambda_reg = lambda_reg

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        if self.use_uncertainty:
            loss_energy = evidential_energy_loss(ref, pred, ddp, self.quantile, self.lambda_reg)
            loss_forces = evidential_forces_loss(ref, pred, ddp, self.quantile, self.lambda_reg)
        else:
            # Fallback to standard loss
            from mace.modules.loss import (
                weighted_mean_squared_error_energy,
                mean_squared_error_forces,
            )
            loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
        
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, use_uncertainty={self.use_uncertainty}, "
            f"quantile={self.quantile:.2f}, lambda_reg={self.lambda_reg:.4f})"
        )


class UncertaintyWeightedEnergyForcesStressLoss(torch.nn.Module):
    """
    Combined energy, force, and stress loss with EIP evidential uncertainty.
    
    Note: Stress currently uses standard MSE loss (no uncertainty weighting yet).
    """
    
    def __init__(
        self, 
        energy_weight: float = 1.0, 
        forces_weight: float = 1.0,
        stress_weight: float = 1.0,
        use_uncertainty: bool = True,
        quantile: float = 0.5,
        lambda_reg: float = 0.01,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.use_uncertainty = use_uncertainty
        self.quantile = quantile
        self.lambda_reg = lambda_reg

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        if self.use_uncertainty:
            loss_energy = evidential_energy_loss(ref, pred, ddp, self.quantile, self.lambda_reg)
            loss_forces = evidential_forces_loss(ref, pred, ddp, self.quantile, self.lambda_reg)
            loss_stress = uncertainty_weighted_stress_loss(ref, pred, ddp)
        else:
            # Fallback to standard loss
            from mace.modules.loss import (
                weighted_mean_squared_error_energy,
                mean_squared_error_forces,
                weighted_mean_squared_stress,
            )
            loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
            loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        
        return (
            self.energy_weight * loss_energy 
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, "
            f"use_uncertainty={self.use_uncertainty}, quantile={self.quantile:.2f}, "
            f"lambda_reg={self.lambda_reg:.4f})"
        )
