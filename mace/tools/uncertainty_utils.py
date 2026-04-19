###########################################################################################
# Utility functions for loading pretrained models with uncertainty quantification
# Handles parameter initialization and shape matching
###########################################################################################

import logging
from typing import Dict, Any

import torch


EIP_UNCERTAINTY_PARAM_NAMES = ("linear_nu", "linear_alpha", "linear_beta")
LEGACY_UNCERTAINTY_PARAM_NAMES = ("linear_uncertainty",)
ALL_UNCERTAINTY_PARAM_NAMES = (
    EIP_UNCERTAINTY_PARAM_NAMES + LEGACY_UNCERTAINTY_PARAM_NAMES
)


def _has_eip_uncertainty(module: torch.nn.Module) -> bool:
    return all(hasattr(module, name) for name in EIP_UNCERTAINTY_PARAM_NAMES)


def _has_any_uncertainty(module: torch.nn.Module) -> bool:
    return _has_eip_uncertainty(module) or hasattr(module, "linear_uncertainty")


def _parameter_name_has_uncertainty(name: str) -> bool:
    return any(param_name in name for param_name in ALL_UNCERTAINTY_PARAM_NAMES)


def initialize_uncertainty_parameters(model: torch.nn.Module) -> None:
    """
    Initialize uncertainty parameters in the model with Kaiming initialization.
    
    This function should be called after loading a pretrained model to add
    uncertainty quantification capability.
    
    Args:
        model: MACE model with use_uncertainty=True
    """
    for name, module in model.named_modules():
        if _has_eip_uncertainty(module):
            logging.info(f"Initializing EIP uncertainty parameters in {name}")
            for param_name, init_bias in [
                ("linear_nu", 1.0),
                ("linear_alpha", 1.0),
                ("linear_beta", -3.0),
            ]:
                linear_layer = getattr(module, param_name)
                if hasattr(linear_layer, "weight") and linear_layer.weight is not None:
                    torch.nn.init.kaiming_normal_(
                        linear_layer.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                if hasattr(linear_layer, "bias") and linear_layer.bias is not None:
                    torch.nn.init.constant_(linear_layer.bias, init_bias)
        elif hasattr(module, "linear_uncertainty"):
            logging.info(f"Initializing legacy uncertainty parameters in {name}")
            linear_layer = module.linear_uncertainty
            if hasattr(linear_layer, "weight") and linear_layer.weight is not None:
                torch.nn.init.kaiming_normal_(
                    linear_layer.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
            if hasattr(linear_layer, "bias") and linear_layer.bias is not None:
                torch.nn.init.constant_(linear_layer.bias, -1.0)


def load_pretrained_with_uncertainty(
    pretrained_path: str,
    use_uncertainty: bool = True,
    device: str = 'cpu',
    strict: bool = False,
) -> torch.nn.Module:
    """
    Load a pretrained MACE model and optionally add uncertainty quantification.
    
    Args:
        pretrained_path: Path to pretrained model
        use_uncertainty: Whether to enable uncertainty quantification
        device: Device to load model on
        strict: Whether to strictly match all parameters (set False for adding uncertainty)
    
    Returns:
        Loaded model with uncertainty capability
    """
    logging.info(f"Loading pretrained model from {pretrained_path}")
    
    # Load the pretrained model
    pretrained_model = torch.load(pretrained_path, map_location=device)
    
    if not use_uncertainty:
        logging.info("Loaded model without uncertainty quantification")
        return pretrained_model
    
    # Check if model already has uncertainty
    has_uncertainty = any(
        _parameter_name_has_uncertainty(name)
        for name, _ in pretrained_model.named_parameters()
    )
    
    if has_uncertainty:
        logging.info("Model already has uncertainty parameters")
        return pretrained_model
    
    # If we need to add uncertainty to a pretrained model without it
    logging.warning(
        "Pretrained model does not have uncertainty parameters. "
        "You need to create a new model with use_uncertainty=True and "
        "load the pretrained weights manually."
    )
    
    return pretrained_model


def transfer_weights_with_uncertainty(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    initialize_uncertainty: bool = True,
) -> None:
    """
    Transfer weights from a source model to a target model with uncertainty.
    
    This is useful when you have a pretrained model without uncertainty
    and want to add uncertainty quantification.
    
    Args:
        source_model: Pretrained model without uncertainty
        target_model: New model with use_uncertainty=True
        initialize_uncertainty: Whether to initialize uncertainty parameters
    """
    logging.info("Transferring weights from source to target model")
    
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    # Transfer matching parameters
    transferred = 0
    skipped = 0
    
    for name, param in source_dict.items():
        if name in target_dict:
            if param.shape == target_dict[name].shape:
                target_dict[name].copy_(param)
                transferred += 1
            else:
                logging.warning(
                    f"Shape mismatch for {name}: "
                    f"source {param.shape} vs target {target_dict[name].shape}"
                )
                skipped += 1
        else:
            # This is expected for uncertainty parameters
            if not _parameter_name_has_uncertainty(name):
                logging.warning(f"Parameter {name} not found in target model")
            skipped += 1
    
    logging.info(f"Transferred {transferred} parameters, skipped {skipped}")
    
    # Initialize uncertainty parameters
    if initialize_uncertainty:
        initialize_uncertainty_parameters(target_model)
    
    logging.info("Weight transfer completed")


def check_model_compatibility(
    model: torch.nn.Module,
    use_uncertainty: bool,
) -> Dict[str, Any]:
    """
    Check if a model is compatible with uncertainty quantification settings.
    
    Args:
        model: Model to check
        use_uncertainty: Expected uncertainty setting
    
    Returns:
        Dictionary with compatibility information
    """
    has_uncertainty = any(
        _parameter_name_has_uncertainty(name) for name, _ in model.named_parameters()
    )
    
    model_has_flag = hasattr(model, 'use_uncertainty') and model.use_uncertainty
    
    info = {
        'has_uncertainty_parameters': has_uncertainty,
        'has_uncertainty_flag': model_has_flag,
        'expected_uncertainty': use_uncertainty,
        'compatible': has_uncertainty == use_uncertainty,
    }
    
    if not info['compatible']:
        if use_uncertainty and not has_uncertainty:
            info['message'] = (
                "Model does not have uncertainty parameters but use_uncertainty=True. "
                "You need to create a new model with use_uncertainty=True and "
                "transfer weights using transfer_weights_with_uncertainty()."
            )
        elif not use_uncertainty and has_uncertainty:
            info['message'] = (
                "Model has uncertainty parameters but use_uncertainty=False. "
                "The uncertainty parameters will be ignored during inference."
            )
    else:
        info['message'] = "Model is compatible with uncertainty settings."
    
    return info


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Example: Loading pretrained model with uncertainty")
    print("=" * 60)
    
    # Example 1: Load pretrained model
    # model = load_pretrained_with_uncertainty(
    #     pretrained_path='pretrained.model',
    #     use_uncertainty=True,
    #     device='cuda',
    # )
    
    # Example 2: Transfer weights
    # source_model = torch.load('pretrained_without_uncertainty.model')
    # target_model = create_model_with_uncertainty()  # Your model creation function
    # transfer_weights_with_uncertainty(source_model, target_model)
    
    # Example 3: Check compatibility
    # model = torch.load('model.model')
    # info = check_model_compatibility(model, use_uncertainty=True)
    # print(info['message'])
    
    print("\nSee function docstrings for detailed usage")
