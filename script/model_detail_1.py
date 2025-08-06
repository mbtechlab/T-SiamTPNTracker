import os
import sys
import shutil
import torch

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lib.config.tpn_st import cfg, update_config_from_file
from lib.models.siamtpn_st.track_st import build_network
from torch.autograd import profiler


def clear_and_make_directory(path):
    """Clear and recreate the directory"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_details(model, filepath):
    """Save model details to a text file"""
    with open(filepath, 'w') as file:
        file.write("Model Details:\n")
        file.write("--------------------------------------------------------------------------------\n")
        total_params = 0
        total_trainable_params = 0
        for name, module in model.named_modules():
            for param_name, parameter in module.named_parameters(recurse=False):
                param = parameter.numel()
                trainable = parameter.requires_grad
                trainability = "Trainable" if trainable else "Frozen"
                file.write(f"{name}.{param_name:30} | Count: {param:10} | {trainability}\n")
                total_params += param
                if trainable:
                    total_trainable_params += param

        file.write("--------------------------------------------------------------------------------\n")
        file.write(f"Total parameters: {total_params}\n")
        file.write(f"Trainable parameters: {total_trainable_params}\n")
        file.write("--------------------------------------------------------------------------------\n")

def save_model_structure(model, filepath):
    """Save model structure to a text file"""
    with open(filepath, 'w') as file:
        def print_module(module, indent=0):
            for name, submodule in module.named_children():
                file.write(' ' * indent + f"{name}: {submodule.__class__.__name__}\n")
                print_module(submodule, indent + 4)

        file.write("Model Structure:\n")
        file.write("--------------------------------------------------------------------------------\n")
        print_module(model)
        file.write("--------------------------------------------------------------------------------\n")

def profile_model_performance(model, train_input, test_input, filepath):
    """Profile model performance and save the results"""
    with profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True) as prof:
        model(train_input, test_input)
    
    with open(filepath, 'a') as file:  # Append to the existing file
        file.write(prof.key_averages().table(sort_by="cuda_time_total"))
        file.write("\n--------------------------------------------------------------------------------\n")

def model_detail(local_rank=-1, save_dir=None, base_seed=None):
    # Set absolute path for the YAML file
    yaml_file_path = os.path.join(parent_dir, 'experiments', 'tpn_st.yaml')

    # Update configuration with the config file
    update_config_from_file(cfg, yaml_file_path)
    
    # Build the network model
    net = build_network(cfg)
    net.cuda()  # Transfer the model to CUDA

    # Prepare the storage directory
    clear_and_make_directory(save_dir)

    # Save model information
    model_params_path = os.path.join(os.path.abspath(save_dir), "model_params.txt")
    model_structure_path = os.path.join(os.path.abspath(save_dir), "model_structure.txt")
    
    if local_rank in [-1, 0]:  # Save only in the main process
        save_model_details(net, model_params_path)
        save_model_structure(net, model_structure_path)

# Run the main function
model_detail(local_rank=-1, save_dir="./results/model_detail_1")
