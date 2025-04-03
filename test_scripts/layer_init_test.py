import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Custom initialization function that scales weights so that:
# bias + 3 * desired_std ≈ max_q
def layer_init(layer, bias_const=0.0, max_q=2.0):
    # Standard Kaiming initialization
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)

    # Compute the desired standard deviation such that bias_const + 3*desired_std ≈ max_q.
    desired_std = (max_q - bias_const) / 3.0

    # Get current standard deviation of the weights
    current_std = layer.weight.data.std()

    # Scale weights if possible
    if current_std > 0:
        scaling_factor = current_std / desired_std
        layer.weight.data.mul_(scaling_factor)

    return layer


# Parameters
in_features = 64
out_features = 1
num_samples = 10000

# Generate random input data (assuming roughly unit variance)
inputs = torch.randn(num_samples, in_features)

# --- Standard Initialization ---
layer_standard = nn.Linear(in_features, out_features)
nn.init.kaiming_normal_(layer_standard.weight, nonlinearity='relu')
torch.nn.init.constant_(layer_standard.bias, 0.0)

with torch.no_grad():
    outputs_standard = layer_standard(inputs)

# --- Custom Initialization with max_q=2 ---
layer_custom = nn.Linear(in_features, out_features)
layer_custom = layer_init(layer_custom, bias_const=0.0, max_q=0.5)

with torch.no_grad():
    outputs_custom = layer_custom(inputs)

# Convert outputs to numpy arrays for plotting
outputs_standard_np = outputs_standard.numpy().flatten()
outputs_custom_np = outputs_custom.numpy().flatten()

# Plot the distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(outputs_standard_np, bins=50, color='blue', alpha=0.7)
plt.title("Standard Initialization")
plt.xlabel("Q-value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(outputs_custom_np, bins=50, color='green', alpha=0.7)
plt.title("Custom Initialization (max_q = 2)")
plt.xlabel("Q-value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
