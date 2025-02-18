import torch
import torch.nn as nn
import os
from adaptive_network import AdaptiveMLP, data_loader
from train_adaptive_configure import load_robot_data

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: The trained AdaptiveMLP model.
        test_loader: DataLoader for test data.
        device: Device to run inference on.
    
    Returns:
        Average test loss.
    """
    model.eval()
    model.to(device)
    
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for joint_states, targets, gt_dynamic_joint_description, gt_general_policy_state in test_loader:
            # Forward pass
            dynamic_joint_description_pred, general_policy_state_pred = model(joint_states, targets)

            # Compute loss
            loss = criterion(dynamic_joint_description_pred, gt_dynamic_joint_description) + \
                   criterion(general_policy_state_pred, gt_general_policy_state)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Evaluation Loss: {avg_loss:.6f}")

    return avg_loss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    h5py_file = "logs/rsl_rl/Genhexapod2_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/2024-12-18_10-38-06/h5py_record/obs_actions_00001.h5"
    dynamic_joint_state, targets, dynamic_joint_description, general_policy_state = load_robot_data(h5py_file, device)

    # Prepare test DataLoader
    test_loader = data_loader(dynamic_joint_state, targets, dynamic_joint_description, general_policy_state, batch_size=16, device=device)

    # Load trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "logs/adaptive_configure_model.pth")
    
    model = AdaptiveMLP()
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # Evaluate model
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
