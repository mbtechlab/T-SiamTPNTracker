import os
import sys
import shutil
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
from tabulate import tabulate

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lib.config.tpn_st import cfg, update_config_from_file
from lib.models.siamtpn_st.track_st import build_network

def clear_and_make_directory(path):
    """Clear and recreate the directory"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_details(model, filepath):
    """Save model parameter count details to a text file"""
    with open(filepath, 'w') as file:
        # Calculate parameter counts for each layer
        params = parameter_count_table(model)
        file.write("Parameter Count per Module:\n")
        file.write("--------------------------------------------------------------------------------\n")
        file.write(params)
        file.write("\n--------------------------------------------------------------------------------\n")

def save_flops_details(model, inputs, filepath):
    """Calculate and save FLOPs for each layer"""
    flops = FlopCountAnalysis(model, inputs)
    flop_table = flop_count_table(flops, max_depth=4)  # Adjust depth as needed
    with open(filepath, 'w') as file:
        file.write("FLOPs per Module:\n")
        file.write("--------------------------------------------------------------------------------\n")
        file.write(flop_table)
        file.write("\n--------------------------------------------------------------------------------\n")

def save_detailed_report(model, inputs, filepath):
    """Save a detailed report of FLOPs and parameters as a table"""
    # Calculate FLOPs
    flops = FlopCountAnalysis(model, inputs)
    # Get FLOPs data as a dictionary
    flops_per_module = flops.by_module()
    # Calculate parameter counts
    params = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
              for name, module in model.named_modules()}
    # Prepare data for the table
    table_data = []
    for name in flops_per_module.keys():
        flops_value = flops_per_module[name]
        params_value = params.get(name, 0)
        flops_str = f"{flops_value / 1e6:.3f}M"
        params_str = f"{params_value / 1e3:.3f}K" if params_value < 1e6 else f"{params_value / 1e6:.3f}M"
        table_data.append([name, params_str, flops_str])

    # Sort data by module name
    table_data.sort(key=lambda x: x[0])

    # Create table
    table = tabulate(table_data, headers=["Module", "#Parameters", "#FLOPs"], tablefmt="github")

    # Save table to file
    with open(filepath, 'w') as file:
        file.write(table)

def model_detail(local_rank=-1, save_dir=None, base_seed=None):
    # Set absolute path for the YAML file
    yaml_file_path = os.path.join(parent_dir, 'experiments', 'tpn_st.yaml')

    # Update configuration with the config file
    update_config_from_file(cfg, yaml_file_path)
    
    # Build the network model
    net = build_network(cfg)
    net.eval()
    net.cuda()  # Transfer model to CUDA

    # Prepare storage directory
    clear_and_make_directory(save_dir)

    # Create dummy inputs for the model with required dimensions
    train_input = [torch.randn(1, 3, 128, 128).cuda()]  # Transfer inputs to CUDA
    test_input = torch.randn(1, 3, 256, 256).cuda()  # Dummy test input
    inputs = (train_input, test_input)

    # Define file paths
    model_flops_path = os.path.join(os.path.abspath(save_dir), "model_flops.txt")
    detailed_report_path = os.path.join(os.path.abspath(save_dir), "detailed_report.txt")

    if local_rank in [-1, 0]:  # Save only in the main process
        # Save FLOPs for each layer
        save_flops_details(net, inputs, model_flops_path)
        # Save detailed report
        save_detailed_report(net, inputs, detailed_report_path)

# Run the main function
if __name__ == "__main__":
    model_detail(local_rank=-1, save_dir="./results/model_detail_flops")
