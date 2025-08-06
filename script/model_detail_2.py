import os
import sys
import torch
import torch.nn as nn
import shutil

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lib.models.siamtpn_st.track_st import build_network
from lib.config.tpn_st import cfg, update_config_from_file

def clear_and_make_directory(path):
    """Clear and recreate the directory"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_summary(model, train_input, test_input, filepath):
    """Save a summary of the model, including parameter counts and layer output shapes"""
    total_params = 0  # Initialize total_params

    def hook(module, input, output):
        nonlocal total_params  # Declare total_params as nonlocal to modify it
        layer_name = f"{module.__class__.__name__}"

        # Check if output is a dictionary, tuple, or list
        if isinstance(output, dict):  # If output is a dictionary
            output_shape = {key: value.shape for key, value in output.items()}
        elif isinstance(output, tuple):  # If output is a tuple
            output_shape = [o.shape for o in output]
        elif isinstance(output, list):  # If output is a list
            output_shape = [o.shape for o in output]
        else:  # If output is a single tensor
            output_shape = output.shape

        layer_params = 0
        if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
            layer_params += torch.numel(module.weight.data)
        if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
            layer_params += torch.numel(module.bias.data)

        total_params += layer_params

        # Write layer information to the file
        file.write(f"{layer_name}: {layer_params} parameters\n")
        file.write(f"Output shape: {output_shape}\n\n")

    with open(filepath, 'w') as file:
        # Register hooks and perform forward pass
        hooks = [layer.register_forward_hook(hook) for layer in model.modules() if isinstance(layer, nn.Module)]
        model(train_input, test_input)  # Ensure your model accepts two inputs
        for hook in hooks:
            hook.remove()  # Remove hooks after usage

        file.write("\nModel Summary:\n")
        file.write("--------------------------------------------------------------------------------\n")
        file.write(f"Total parameters in model: {total_params}\n")
        file.write("--------------------------------------------------------------------------------\n")

def model_detail(local_rank=-1, save_dir=None):
    # Set absolute path for the YAML file
    yaml_file_path = os.path.join(parent_dir, 'experiments', 'tpn_st.yaml')

    # Update configuration with the config file
    update_config_from_file(cfg, yaml_file_path)
    
    # Build the network
    net = build_network(cfg)  # Model remains on CPU

    # Clear and create results folder
    clear_and_make_directory(save_dir)
    model_summary_path = os.path.join(save_dir, "model_summary.txt")

    # Prepare dummy inputs for the model on CPU
    dummy_train_input = [torch.randn(1, 3, 224, 224), 
                         torch.randn(1, 3, 224, 224), 
                        #  torch.randn(1, 3, 224, 224),
                        #  torch.randn(1, 3, 224, 224),
                        #  torch.randn(1, 3, 224, 224),
                         ]
    dummy_test_input = torch.randn(1, 3, 224, 224)

    # Save model summary
    save_model_summary(net, dummy_train_input, dummy_test_input, model_summary_path)

# Example usage:
model_detail(local_rank=-1, save_dir="./results/model_detail_2")
